"""Native OpenRouter tool calls for an isolated GTA Workflow inference process."""

import copy
import hashlib
import json
import random
import re
import time
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import Lock
from uuid import uuid4

SYSTEM = (
    "Complete the user's task. When a tool is needed, call one available "
    "function at a time and wait for its actual result. Use the exact input "
    "paths and returned output paths. A failed or skipped call did not "
    "complete its work. If the task asks for an artifact, create it with the "
    "available tools and identify the actual result. For a text answer, "
    "give the requested content directly. Describe any unfinished work "
    "accurately. Use assistant text for your final answer."
)
_OPENROUTER = 'https://openrouter.ai/api/v1/chat/completions'
_KEY_LOCK = Lock()
_WIRE_NAME = re.compile(r'[A-Za-z0-9_-]{1,64}')
_TEXT_CALL = re.compile(r'^\s*(?:Thought:[\s\S]*?\n)?(?:Tool|Action):[^\n]+\n'
                        r'\s*(?:Tool Input|Action Input):', re.I)
_PARAM_TYPES = {
    'STRING': 'string', 'NUMBER': 'integer', 'INTEGER': 'integer',
    'FLOAT': 'number', 'BOOLEAN': 'boolean', 'ARRAY': 'array', 'OBJECT': 'object',
}


class NativeError(RuntimeError):
    """Contains only local diagnostics suitable for the prediction record."""


class ProviderError(NativeError):
    def __init__(self, message, category, http_status=None, upstream_failure=None, wire_http_status=None):
        super().__init__(message)
        self.category = category
        self.http_status = http_status
        self.wire_http_status = wire_http_status
        self.upstream_failure = upstream_failure


class _RejectedBudget(ProviderError):
    def __init__(self, source, status, kind, sent, numbers, wire_status=None):
        super().__init__('Unable to fit request within the ' + kind + ' budget', source, status, kind, wire_status)
        self.sent = sent
        self.numbers = numbers


class _RequiredUnsupported(ProviderError):
    pass


def _effective_error_status(wire_status, detail):
    if 200 <= wire_status < 300 and isinstance(detail, dict):
        code = detail.get('code')
        if type(code) in (int, str) and re.fullmatch(r'[45][0-9]{2}', str(code)):
            return int(code)
    return wire_status


def _transient_transport(exc, errors):
    # SSLError is a ConnectionError subclass. Explicitly distinguish EOF
    # during TLS handshakes/reads from bad certificates and TLS configuration.
    if isinstance(exc, (errors.ConnectionError, errors.Timeout)):
        text = str(exc).lower()
        if any(s in text for s in ('certificate verify failed', 'certificate_verify_failed',
                                   'proxy authentication required', 'wrong version number')):
            return None
        if any(s in text for s in ('unexpected_eof_while_reading', 'unexpected eof while reading',
                                   'eof occurred in violation of protocol')):
            return 'tls_unexpected_eof'
        if isinstance(exc, errors.SSLError):
            return None
        return 'timeout' if isinstance(exc, errors.Timeout) else 'connection_error'
    if isinstance(exc, errors.ChunkedEncodingError):
        return 'incomplete_transfer'
    return None


def _retry_after(headers):
    if not hasattr(headers, 'get'):
        return None
    value = headers.get('Retry-After') or headers.get('retry-after')
    if not isinstance(value, str) or len(value) > 80:
        return None
    value = value.strip()
    if re.fullmatch(r'[0-9]+', value):
        return min(60, int(value))
    try:
        date = parsedate_to_datetime(value)
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return min(60, max(0, date.timestamp() - time.time()))
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def _wait_for_retry(journal, turn, attempt, reason, retry_after=None):
    base = min(60, 2 ** min(attempt, 6))
    floor = max(base, retry_after or 0)
    delay = floor + random.uniform(0, min(base, 60 - floor))
    journal.emit('request_backoff', turn=turn, attempt=attempt, next_attempt=attempt + 1,
                 reason=reason, delay_seconds=round(delay, 3), retry_after_seconds=retry_after)
    time.sleep(delay)


_COUNT = r'([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{1,9})(?![a-zA-Z0-9]|[.,]\d)'
_LIMIT_PATTERNS = {
    'context_tokens': (
        r'(?:(?:max(?:imum)?[ _-])?context[ _-](?:length|window)(?:[ _-](?:limit|size))?'
        r'|context[ _-](?:token[ _-])?limit)\s*(?:is|of|:|=)?\s*\(?\s*' + _COUNT,
    ),
    'input_tokens': (
        _COUNT + r'\s+(?:tokens?\s+)?(?:in (?:the |your )?(?:messages?|prompt)|of text input|input tokens|prompt tokens)',
        r'(?:input[ _-](?:length|tokens)|prompt[ _-](?:length|tokens)|messages resulted in)'
        r'\s*(?:is|of|:|=)?\s*\(?\s*(?:at least\s+)?' + _COUNT,
        r'(?:your )?(?:input|prompt|messages)\s+(?:contains?|have|has)\s+(?:at least\s+)?' + _COUNT + r'\s+tokens',
    ),
    'requested_total_lower_bound': (
        r'(?:request(?:ed| requires)|requested total(?: is)?)\s+at least\s+' + _COUNT + r'\s+tokens',
    ),
    'max_output_tokens': (
        r'(?:max(?:imum)? (?:output|completion) (?:length|tokens)|completion token limit)'
        r'\s*(?:is|of|:|=)?\s*\(?\s*' + _COUNT,
        r'max[_ ](?:completion[_ ])?tokens[^.\n]{0,40}?(?:at most|less than or equal to|<=|up to)\s*' + _COUNT,
        r'max[_ ](?:completion[_ ])?tokens[^.\n]{0,35}?between\s+\d+\s+and\s+' + _COUNT,
    ),
}


def _budget_numbers(detail):
    """Read only documented numeric fields and token counts in error prose."""
    aliases = {'max_context_length': 'context_tokens', 'context_length_limit': 'context_tokens',
               'max_model_len': 'context_tokens', 'prompt_tokens': 'input_tokens',
               'input_tokens': 'input_tokens', 'max_output_tokens': 'max_output_tokens'}
    numbers, messages, pending = {}, [], [detail]
    for _ in range(12):
        if not pending:
            break
        obj = pending.pop()
        if isinstance(obj, str):
            snippet = obj[:16000]
            try:
                nested = json.loads(snippet)
            except ValueError:
                nested = None
            if isinstance(nested, dict):
                pending.append(nested)
            else:
                messages.append(snippet)
        if not isinstance(obj, dict):
            continue
        for key, label in aliases.items():
            value = obj.get(key)
            if isinstance(value, str) and re.fullmatch(_COUNT, value):
                value = int(value.replace(',', ''))
            if type(value) is int and 0 < value < 1_000_000_000:
                numbers.setdefault(label, value)
        pending.extend(obj[k] for k in ('message', 'error', 'metadata', 'raw', 'details')
                       if isinstance(obj.get(k), (str, dict)))
    for label, patterns in _LIMIT_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, '\n'.join(messages), flags=re.I)
            if match:
                value = int(match[1].replace(',', ''))
                if 0 < value < 1_000_000_000:
                    numbers.setdefault(label, value)
                    break
    return numbers


def _failure_kind(status, detail):
    """Classify known upstream diagnostics without logging their raw content."""
    if not isinstance(detail, dict):
        detail = {}
    meta = detail.get('metadata')
    meta = meta if isinstance(meta, dict) else {}
    if (meta.get('reason') == 'in_flight_budget_exhausted'
            or meta.get('limit_source') == 'openrouter_in_flight_budget'):
        return 'in_flight_budget'
    fields = [detail.get('code'), detail.get('message'), meta.get('reason'), meta.get('raw')]
    text = ' '.join(str(part)[:16000] for part in fields if part).lower()
    text = re.sub(r'[-_]', ' ', text)
    for kind, patterns in (
        ('in_flight_budget', ('in flight budget',)),
        ('thought_signature', ('thought signature', 'thoughtsignature', 'thinking signature')),
        ('invalid_tool_schema', ('invalid function schema', 'invalid tool schema', 'invalid schema',
                                 'function declaration', 'function.parameters', 'function parameters')),
        ('context_limit', ('context length', 'context window', 'maximum context', 'context size',
                           'context limit', 'prompt is too long', 'input token limit', 'max model len')),
        ('output_limit', ('maximum output tokens', 'maximum completion tokens', 'maximum output length',
                          'maximum completion length', 'completion token limit')),
        ('request_size', ('payload too large', 'request too large', 'request body too large',
                          'request size limit', 'entity too large')),
        ('insufficient_credits', ('insufficient credits', 'insufficient balance', 'insufficient quota',
                                  'insufficient funds', 'can only afford', 'requires more credits',
                                  'exceeded your current quota', 'credit balance is too low',
                                  'billing hard limit', 'spending limit reached')),
        ('account_quota_limit', ('account quota exceeded', 'account quota exhausted',
                                 'daily limit for your account', 'monthly limit for your account')),
        ('unsupported_tool_capability', ('no endpoints found that support', 'does not support tools',
                                         'does not support tool', 'tool use is not supported',
                                         'tool calling is not supported', 'unsupported parameter')),
    ):
        if any(pattern in text for pattern in patterns):
            return kind
    if (status in (400, 413, 422) and re.search(
            r'\bmax (?:completion )?tokens\b[^.\n]{0,60}?(?:at most|less than or equal to|<=|'
            r'between\s+\d+\s+and|too (?:large|high)|exceeds (?:the |its )?limit)', text)):
        return 'output_limit'
    return {402: 'payment_required', 413: 'request_size'}.get(status, 'unknown')


def _unsupported_required(detail):
    if not isinstance(detail, dict):
        return False
    meta = detail.get('metadata')
    raw = meta.get('raw', '') if isinstance(meta, dict) else ''
    text = (str(detail.get('message', ''))[:8000] + ' ' + str(raw)[:16000]).lower().replace('_', ' ')
    return ('tool choice' in text and any(s in text for s in
            ('not support', 'unsupported', 'only support', 'only allow', 'must be auto',
             'must be either', 'no endpoints found that support', "doesn't support",
             'not available', 'not implemented', 'not allowed', 'not permitted')))


def _gemini_error_diagnostics(detail, messages):
    """Keep allowlisted provider hints and counts, never error text or signatures."""
    detail = detail if isinstance(detail, dict) else {}
    meta = detail.get('metadata')
    meta = meta if isinstance(meta, dict) else {}
    raw = meta.get('raw')
    nested = raw
    raw_format = 'none'
    if isinstance(raw, str):
        raw_format = 'text'
        try:
            nested = json.loads(raw[:32000])
        except ValueError:
            nested = None
    if isinstance(nested, dict):
        raw_format = 'object'
        nested = nested.get('error', nested)
    nested = nested if isinstance(nested, dict) else {}
    fields = [detail.get('message'), nested.get('message'), nested.get('details')]
    if raw_format == 'text':
        fields.append(raw)
    text = ' '.join(str(value)[:16000] for value in fields if value).lower()
    text = re.sub(r'[_\s-]+', ' ', text)
    signals = []
    for label, phrases in (
        ('thought_signature', ('thought signature', 'thoughtsignature', 'thinking signature')),
        ('function_response', ('function response', 'functionresponse', 'function result', 'tool call id')),
        ('turn_order', ('immediately after', 'immediately follow', 'must be followed', 'alternate',
                        'consecutive', 'last turn', 'matching function call')),
        ('tool_schema', ('functiondeclarations', 'function declarations', 'function declaration',
                         'invalid schema', 'parameters.properties')),
        ('empty_content', ('must not be empty', 'must be non empty', 'empty parts', 'empty content')),
        ('token_budget', ('maxoutputtokens', 'max output tokens', 'output token', 'context length',
                          'input token', 'token count', 'maximum tokens')),
        ('payload_size', ('payload size', 'request size', 'too large', 'size limit')),
        ('malformed_payload', ('invalid json', 'unknown name', 'unknown field', 'cannot find field')),
        ('invalid_argument', ('invalid argument', 'invalidargument')),
        ('safety_filter', ('safety rating', 'safety filter', 'blocked for safety')),
    ):
        if any(phrase in text for phrase in phrases):
            signals.append(label)
    allowed_statuses = {'INVALID_ARGUMENT', 'FAILED_PRECONDITION', 'OUT_OF_RANGE', 'RESOURCE_EXHAUSTED',
                        'PERMISSION_DENIED', 'UNAUTHENTICATED', 'UNAVAILABLE', 'INTERNAL', 'NOT_FOUND'}
    status = nested.get('status') or detail.get('status')
    provider = meta.get('provider_name')
    allowed_providers = ('Google AI Studio', 'Google Vertex', 'Google Vertex AI', 'Vertex AI', 'OpenRouter')
    matched, pending = 0, set()
    tool_rounds = calls = signed = reasoning_rounds = unmatched = 0
    for msg in messages:
        if msg.get('role') == 'assistant' and msg.get('tool_calls'):
            tool_rounds += 1
            reasoning_rounds += bool(msg.get('reasoning_details'))
            for call in msg['tool_calls']:
                calls += 1
                if isinstance(call.get('id'), str):
                    pending.add(call['id'])
                extra = call.get('extra_content')
                google = extra.get('google') if isinstance(extra, dict) else None
                signed += isinstance(google, dict) and bool(google.get('thought_signature'))
        elif msg.get('role') == 'tool':
            ident = msg.get('tool_call_id')
            if ident in pending:
                pending.remove(ident)
                matched += 1
            else:
                unmatched += 1
    return dict(raw_format=raw_format, status=status if isinstance(status, str) and status in allowed_statuses else None,
                provider=provider if provider in allowed_providers else None, signals=signals,
                token_limits=_budget_numbers(detail),
                request_messages=len(messages), tool_call_rounds=tool_rounds, calls=calls,
                calls_with_google_signature=signed, tool_rounds_with_reasoning_details=reasoning_rounds,
                matched_tool_results=matched, unmatched_tool_results=unmatched, pending_tool_results=len(pending))


def _wire_name(name):
    if _WIRE_NAME.fullmatch(name):
        return name
    stem = re.sub(r'[^A-Za-z0-9_-]', '_', name)[:48]
    return stem + '_' + hashlib.sha256(name.encode()).hexdigest()[:12]


def _functions(wrapper, resources):
    from opencompass.models.lagent import DummyTool

    actions, names, schemas, unavailable = {}, {}, [], []
    for name in dict.fromkeys(item['name'] for item in resources if item['type'] == 'tool'):
        action = wrapper.tools.get(name)
        if action is None or isinstance(getattr(action, 'tool', None), DummyTool):
            unavailable.append(name)
            continue
        alias = _wire_name(name)
        if alias in names:
            raise NativeError('Two tool names map to the same provider function')
        desc = action.description
        properties = {}
        for item in desc.get('parameters', ()):
            kind = str(item['type']).upper()
            if kind not in _PARAM_TYPES:
                raise NativeError('Unknown tool parameter type: ' + kind)
            prop = {'type': _PARAM_TYPES[kind]}
            if item.get('description'):
                prop['description'] = item['description']
            properties[item['name']] = prop
        names[alias], actions[name] = name, action
        schemas.append({'type': 'function', 'function': {
            'name': alias, 'description': desc.get('description') or name,
            'parameters': {'type': 'object', 'properties': properties,
                           'required': list(desc.get('required', [])),
                           'additionalProperties': False}}})
    return actions, names, schemas, unavailable


class _Journal:
    def __init__(self, root):
        self.root = root
        self.run_id = uuid4().hex
        self.http_attempts = self.native_calls = self.executions = 0

    def emit(self, event, **values):
        if event == 'provider':
            self.http_attempts += 1
            self.native_calls += values.get('tool_call_count', 0)
        if event == 'tool' and values.get('executed'):
            self.executions += 1
        if event == 'terminal':
            values.update(http_attempts=self.http_attempts, native_calls=self.native_calls,
                          execution_attempts=self.executions)
        if self.root is not None:
            with (self.root / 'native_events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(event=event, invocation=self.run_id,
                                             time=time.time(), **values), ensure_ascii=False) + '\n')

    def files(self):
        if self.root is None:
            return []
        roots = [self.root / 'files', *(self.root / 'calls').glob('*/generated')]
        return sorted(str(path) for root in roots for path in root.rglob('*')
                      if path.is_file() and not path.is_symlink()
                      and path.resolve().is_relative_to(self.root) and path.stat().st_size)


def _request_once(llm, messages, schemas, final_only, journal, turn, output_limit, trimmed, required):
    import requests

    if llm.url != _OPENROUTER:
        raise NativeError('Configured URL is not the OpenRouter chat completions endpoint')
    options = dict(llm.gen_params)
    temperature = llm.temperature if llm.temperature is not None else 0.7
    token_key = 'max_completion_tokens' if 'max_completion_tokens' in options else 'max_tokens'
    requested = int(options.pop(token_key, 512))
    if output_limit is not None:
        requested = min(requested, output_limit)
    options.pop('max_tokens', None)
    estimate = llm.get_token_len(json.dumps([messages, schemas], ensure_ascii=False))
    context_window = llm.max_seq_len or 4096
    budget = context_window - estimate - 100
    if budget < 1:
        raise _RejectedBudget('local_context_budget', None, 'context_limit', requested,
                              {'context_tokens': context_window, 'input_tokens': estimate})
    payload = dict(n=1, temperature=temperature)
    payload.update(options)
    payload.update(model=llm.path, messages=messages, **{token_key: min(requested, budget)})
    # Stop strings and tool settings from a text-protocol adapter are inapplicable here.
    for key in ('stop', 'tools', 'tool_choice', 'parallel_tool_calls'):
        payload.pop(key, None)
    if schemas:
        payload.update(tools=schemas, tool_choice='none' if final_only else 'auto',
                       parallel_tool_calls=False)
        if required and not final_only:
            payload['tool_choice'] = 'required'
    sent = dict(turn=turn, sent_max_tokens=payload[token_key], temperature=payload.get('temperature'),
                tool_choice=payload.get('tool_choice'))
    if trimmed:
        sent['trimmed_observations'] = trimmed

    timeout = getattr(llm, 'request_timeout', (30, 600))
    category, status, last_error = 'configuration', None, 'no configured attempts'
    wire_status = None
    upstream_failure = None
    # Rotate on a new logical request. Keep its selected key/org on every
    # transport retry of the same payload.
    with getattr(llm, '_key_lock', _KEY_LOCK):
        for _ in range(len(llm.keys)):
            llm.key_ctr = (llm.key_ctr + 1) % len(llm.keys)
            key = llm.keys[llm.key_ctr]
            if key and key not in llm.invalid_keys:
                break
        else:
            raise NativeError('No configured provider credential is available')
        headers = {'Authorization': 'Bearer ' + key}
        if llm.orgs:
            llm.org_ctr = (llm.org_ctr + 1) % len(llm.orgs)
            headers['OpenAI-Organization'] = llm.orgs[llm.org_ctr]
    for attempt in range(1, llm.retry + 1):
        upstream_failure = None
        llm.wait()
        try:
            response = requests.post(llm.url, headers=headers, json=payload,
                                     timeout=timeout, allow_redirects=False)
        except requests.RequestException as exc:
            reason = _transient_transport(exc, requests.exceptions)
            category = 'transport_error' if reason else 'transport_configuration_error'
            status, last_error = None, type(exc).__name__
            wire_status = None
            sent['wire_http_status'] = None
            journal.emit('provider', **sent, attempt=attempt, status=category,
                         error=last_error, http_status=None, transport_reason=reason)
            if not reason or attempt == llm.retry:
                break
            _wait_for_retry(journal, turn, attempt, reason)
            continue
        wire_status = response.status_code
        try:
            outer = response.json()
            data = outer.get('data', outer) if isinstance(outer, dict) else {}
            if not isinstance(data, dict):
                data = {}
        except ValueError:
            outer = data = {}
        choices = data.get('choices')
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        top_error = 'error' in data or (isinstance(outer, dict) and 'error' in outer)
        detail = data.get('error') or (outer.get('error') if isinstance(outer, dict) else None) or choice.get('error')
        status = _effective_error_status(wire_status, detail)
        sent['wire_http_status'] = wire_status
        if not (200 <= status < 300) or top_error:
            category = 'api_error'
            code = str(detail.get('code', '')) if isinstance(detail, dict) else ''
            if (not re.fullmatch(r'[0-9]{3}', code) and code not in
                    ('insufficient_quota', 'rate_limit_exceeded', 'context_length_exceeded',
                     'invalid_request_error', 'invalid_api_key', 'model_not_found')):
                code = ''
            upstream_failure = _failure_kind(status, detail)
            last_error = 'HTTP ' + str(status) + (' ' + code if code else '')
            diagnostics = {}
            if llm.path == 'google/gemini-2.5-pro' and status in (400, 422) and upstream_failure == 'unknown':
                diagnostics['gemini_validation'] = _gemini_error_diagnostics(detail, messages)
            if status in (400, 413, 422) and upstream_failure in ('context_limit', 'output_limit'):
                numbers = _budget_numbers(detail)
                journal.emit('provider', **sent, attempt=attempt, status='api_error', http_status=status,
                             code=code, upstream_failure=upstream_failure, token_limits=numbers)
                raise _RejectedBudget(category, status, upstream_failure, payload[token_key], numbers, wire_status)
            journal.emit('provider', **sent, attempt=attempt, status='api_error',
                         http_status=status, code=code, upstream_failure=upstream_failure, **diagnostics)
            if (payload.get('tool_choice') == 'required' and status in (400, 404, 422)
                    and _unsupported_required(detail)):
                raise _RequiredUnsupported(last_error, category, status, 'unsupported_required_tool_choice', wire_status)
            if code == 'insufficient_quota' and len(llm.keys) > 1:
                with getattr(llm, '_key_lock', _KEY_LOCK):
                    llm.invalid_keys.add(key)
            transient_statuses = (408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524)
            transient_code = code in ('', 'rate_limit_exceeded') or (code.isdigit() and int(code) in transient_statuses)
            if (attempt < llm.retry and status in transient_statuses and transient_code
                    and upstream_failure == 'unknown'):
                retry_after = _retry_after(getattr(response, 'headers', {}))
                metadata = detail.get('metadata') if isinstance(detail, dict) else None
                if retry_after is None and isinstance(metadata, dict):
                    retry_after = _retry_after(metadata.get('headers'))
                _wait_for_retry(journal, turn, attempt, 'http_' + str(status), retry_after)
                continue
            break
        try:
            if choice.get('error') or choice.get('finish_reason') == 'error':
                category, last_error = 'provider_error', 'Provider failed during generation'
                upstream_failure = _failure_kind(status, choice.get('error'))
                journal.emit('provider', **sent, attempt=attempt, status=category,
                             http_status=status, finish_reason='error', upstream_failure=upstream_failure)
                if upstream_failure != 'unknown':
                    break
                continue
            message = choice['message']
            if not isinstance(message, dict):
                raise TypeError
        except (IndexError, KeyError, TypeError):
            category, last_error = 'malformed', 'Malformed provider response'
            journal.emit('provider', **sent, attempt=attempt, status='malformed', http_status=status)
            continue
        calls = message.get('tool_calls')
        reason = choice.get('finish_reason')
        journal.emit('provider', **sent, attempt=attempt, status='ok', http_status=status,
                     response_id=data.get('id'), model=data.get('model'), provider=data.get('provider'),
                     finish_reason=reason, tool_call_count=len(calls) if isinstance(calls, list) else 0)
        record_usage = getattr(llm, '_record_gta_usage', None)
        if record_usage:
            source = data if isinstance(data.get('usage'), dict) else outer
            record_usage(source, choice, max_out_len=payload[token_key],
                         temperature=payload['temperature'])
        return message, reason
    raise ProviderError('OpenRouter generation failed: ' + last_error, category, status, upstream_failure, wire_status)


def _wire_history(messages, max_chars):
    if max_chars is None:
        return messages, []
    wire, trimmed = [], []
    for i, msg in enumerate(messages):
        content = msg.get('content')
        if msg['role'] != 'tool' or not isinstance(content, str) or len(content) <= max_chars:
            wire.append(msg)
            continue
        kept = max_chars - 100
        head = kept * 2 // 3
        marker = f'\n[... {len(content) - kept} characters omitted from this tool observation ...]\n'
        excerpt = content[:head] + marker + content[-(kept - head):]
        wire.append(dict(msg, content=excerpt))
        trimmed.append({'index': i, 'tool_call_id': msg.get('tool_call_id'), 'original_chars': len(content),
                        'wire_chars': len(excerpt), 'omitted_chars': len(content) - kept})
    return wire, trimmed


def _request_budgeted(llm, messages, schemas, final_only, journal, turn, limits, required):
    # Each logical turn gets at most four changed requests after a verified
    # token-limit rejection. A changed request generates no duplicate tool actions.
    previous_context_width = object()
    for recovery in range(5):
        wire, trimmed = _wire_history(messages, limits.get('observation_chars'))
        try:
            return _request_once(llm, wire, schemas, final_only, journal, turn, limits.get('output'),
                                 trimmed, required)
        except _RejectedBudget as exc:
            n = exc.numbers
            context = n.get('context_tokens')
            inputs = n.get('input_tokens')
            remaining = context - inputs if context and inputs else None
            output = exc.sent
            width = limits.get('observation_chars')
            previous_width = width
            is_context = exc.upstream_failure == 'context_limit'
            largest = max((len(m['content']) for m in wire if m['role'] == 'tool'
                           and isinstance(m.get('content'), str)), default=0) if is_context else 0
            # Some gateways stop counting input as soon as input + requested
            # output exceeds the context. On an unchanged rejected history,
            # another arithmetic adjustment can repeat the same failure.
            reduce_wire = (largest > 512 and (exc.category == 'local_context_budget'
                           or previous_context_width == width or (recovery > 0 and bool(trimmed))))
            if not reduce_wire and exc.category != 'local_context_budget':
                reduced_output = max(1, min(4096, output // 2))
                if n.get('max_output_tokens', output) < output:
                    output = n['max_output_tokens']
                    if is_context and exc.sent - output <= max(128, exc.sent // 100):
                        output = min(output, reduced_output)
                elif remaining is not None and 0 < remaining < output:
                    if output - remaining <= max(128, output // 100):
                        output = reduced_output
                    else:
                        output = max(1, remaining - min(64, remaining // 8))
                elif exc.upstream_failure == 'output_limit' or (recovery == 0 and output > 1):
                    output = reduced_output
            if is_context:
                previous_context_width = previous_width
                if output == exc.sent and largest > 512:
                    width = max(512, min(16384, largest // 2))
            changed = output < exc.sent or width != limits.get('observation_chars')
            if not changed or recovery == 4:
                journal.emit('context_recovery_exhausted', turn=turn, source=exc.category,
                             http_status=exc.http_status, wire_http_status=exc.wire_http_status,
                             token_limits=n, sent_max_tokens=exc.sent,
                             trimmed_observations=trimmed, reason='recovery_limit_reached' if changed
                             else 'no_further_safe_reduction')
                raise
            limits.update(output=output, observation_chars=width)
            journal.emit('context_recovery', turn=turn, retry=recovery + 1, source=exc.category,
                         upstream_failure=exc.upstream_failure, http_status=exc.http_status,
                         wire_http_status=exc.wire_http_status,
                         action='trim_tool_observations' if width != previous_width else 'reduce_output',
                         token_limits=n, previous_max_tokens=exc.sent, next_max_tokens=output,
                         observation_chars=width, trimmed_observations=_wire_history(messages, width)[1])


def _request(llm, messages, schemas, final_only, journal, turn, limits, required=False):
    required = bool(required and schemas and not final_only and not limits.get('required_unsupported'))
    try:
        return _request_budgeted(llm, messages, schemas, final_only, journal, turn, limits, required)
    except _RequiredUnsupported as exc:
        limits['required_unsupported'] = True
        journal.emit('tool_choice_fallback', turn=turn, http_status=exc.http_status,
                     original='required', retry='auto', reason=exc.upstream_failure)
        return _request_budgeted(llm, messages, schemas, final_only, journal, turn, limits, False)


def _tool_result(executor, actual, arguments, skipped):
    if skipped:
        return None, arguments, skipped, False
    if actual is None:
        return None, arguments, 'Unknown or unavailable function.', False
    try:
        args = json.loads(arguments)
        if not isinstance(args, dict):
            raise ValueError
    except (TypeError, ValueError):
        return None, arguments, 'Arguments must be a JSON object.', False
    try:
        action = executor(actual, arguments)
    except Exception as exc:
        return None, args, 'Tool raised ' + type(exc).__name__, True
    error = getattr(action, 'errmsg', None)
    if not error and int(getattr(action, 'state', 0)) != 0:
        error = 'Tool finished with status ' + str(action.state)
    effective = action.args
    if error and isinstance(effective, dict):
        effective = effective.get('inputs', effective)
    return action.result, effective, error, True


def _process_calls(message, calls, executor, names, seen, successful, steps, messages,
                   journal, turn, final_only):
    # Validate the entire envelope before side effects. A missing or repeated ID
    # cannot be repaired locally while retaining the provider's call identity.
    ids = set()
    normalized = []
    for call in calls:
        if not isinstance(call, dict):
            break
        ident, func = call.get('id'), call.get('function')
        if (not isinstance(ident, str) or not ident or ident in ids or ident in seen
                or call.get('type') != 'function' or not isinstance(func, dict)
                or not isinstance(func.get('name'), str)):
            break
        ids.add(ident)
        args = func.get('arguments')
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        normalized.append({**call, 'function': {**func, 'arguments': args}})
    if not normalized or len(normalized) != len(calls):
        journal.emit('invalid_reply', turn=turn, reason='malformed_or_repeated_tool_call_id')
        messages.append({'role': 'user', 'content': 'The tool-call envelope was invalid. '
                         'No tool ran. Send one new native function call or your final answer.'})
        return
    # OpenRouter reasoning models may require unaltered reasoning_details
    # (including provider signatures) on subsequent rounds. Keep them in memory.
    assistant = {k: message[k] for k in ('content', 'reasoning_details', 'reasoning',
                                         'reasoning_content') if k in message}
    messages.append(dict(assistant, role='assistant', tool_calls=normalized))
    seen.update(ids)
    for index, call in enumerate(normalized):
        ident = call['id']
        func = call['function']
        alias = func['name']
        actual = names.get(alias)
        skipped = None
        if final_only:
            skipped = 'Final-answer turn; tool was not executed.'
        elif index:
            skipped = 'Send one tool call per turn; this additional call was not executed.'
        result, args, error, executed = _tool_result(executor, actual, func['arguments'], skipped)
        ok = executed and not error
        if ok:
            successful.add(actual)
        journal.emit('tool', turn=turn, name=actual or alias, executed=executed, ok=ok,
                     call_id=ident)
        step = {'role': 'assistant', 'tool_calls': [{
            'id': ident, 'type': 'function', 'function': {'name': actual or alias, 'arguments': args}}]}
        if error:
            step['error'] = {'type': 'tool_error' if executed else 'not_executed', 'msg': error}
        steps.extend((step, {'role': 'tool', 'tool_call_id': ident, 'content': result}))
        observation = {'error': error, 'executed': executed} if error else result
        messages.append({'role': 'tool', 'tool_call_id': ident,
                         'content': json.dumps(observation, ensure_ascii=False)})


def _text(message):
    content = message.get('content')
    if isinstance(content, list):
        content = ''.join(part.get('text', '') for part in content
                          if isinstance(part, dict) and part.get('type') == 'text'
                          and isinstance(part.get('text'), str))
    return content.strip() if isinstance(content, str) else ''


def native_chat(self, query, memory=None, resources=None):
    from opencompass.models.tool_output import _TASK_ROOT
    from .native_guard import final_guard

    resources = resources or []
    input_files = [item['path'] for item in resources if item['type'] == 'file']
    if memory is None:
        memory = []
        if input_files:
            memory.append({'role': 'user', 'content': 'The related files are at '
                           + ', '.join(f'`{file}`' for file in input_files)})
    memory = [*memory, {'role': 'user', 'content': query}]
    system_prompt = getattr(self, 'native_system_prompt', SYSTEM)
    messages = [{'role': 'system', 'content': system_prompt}, *memory]
    root = _TASK_ROOT.get()
    journal = _Journal(Path(root).resolve() if root is not None else None)
    steps, seen, successful = [], set(), set()
    limits = {}
    status, final_text, issue = 'incomplete', '', 'Turn limit reached'
    failure = {}
    try:
        actions, names, schemas, unavailable = _functions(self, resources)
        journal.emit('start', model=self.agent._llm.path, tools=list(actions), unavailable=unavailable)
        executor = copy.copy(self.agent._action_executor)
        executor.actions = actions
        max_turn = getattr(self.agent, '_max_turn', None)
        if max_turn is None:
            max_turn = self.agent.max_turn
        if max_turn < 1:
            raise NativeError('Invalid configured turn limit')
        require_next = False
        for turn in range(1, max_turn + 1):
            final_only = turn == max_turn and (turn > 1 or not schemas)
            if final_only:
                messages.append({'role': 'user', 'content': 'Give the final answer now. '
                                 'Report any incomplete deliverables accurately.'})
            message, reason = _request(self.agent._llm, messages, schemas, final_only, journal, turn,
                                       limits, required=require_next)
            require_next = False
            calls = message.get('tool_calls')
            # A truncated/incomplete tool envelope must never reach an executor.
            if reason not in ('stop', 'tool_calls'):
                issue = 'Incomplete provider reply: ' + str(reason)
                if not calls:
                    final_text = _text(message) or final_text
                journal.emit('invalid_reply', turn=turn, reason=issue)
                messages.append({'role': 'user', 'content': issue + '. Send one complete call or answer.'})
                continue
            if calls:
                if isinstance(calls, list):
                    _process_calls(message, calls, executor, names, seen, successful, steps,
                                   messages, journal, turn, final_only)
                else:
                    journal.emit('invalid_reply', turn=turn, reason='malformed_tool_calls')
                    messages.append({'role': 'user', 'content': 'The call envelope was invalid. No tool ran.'})
                continue
            text = _text(message)
            if not text or reason == 'tool_calls':
                issue = 'Provider supplied no complete visible answer or native call'
                journal.emit('invalid_reply', turn=turn, reason=issue)
                messages.append({'role': 'user', 'content': issue + '. Use a native call or visible final text.'})
                continue
            issue = final_guard(prompt=query, text=text, used_tools=sorted(successful), files=journal.files(),
                                input_files=input_files, available_tools=list(actions), tool_steps=steps)
            if issue and schemas and turn + 1 < max_turn and not limits.get('required_unsupported'):
                require_next = True
                journal.emit('require_evidence_tool', turn=turn, next_turn=turn + 1)
            if _TEXT_CALL.match(text):
                journal.emit('invalid_reply', turn=turn, reason='textual_tool_call')
                issue = issue or 'The text described a function call. No tool ran. Use a native call or give your answer.'
                messages.append({'role': 'user', 'content': issue})
                continue
            final_text = text
            if not issue:
                status = 'completed'
                break
            journal.emit('guard_retry', turn=turn, reason=issue)
            messages.extend(({'role': 'assistant', 'content': text}, {'role': 'user', 'content': issue}))
    except Exception as exc:
        status = 'error'
        issue = str(exc) if isinstance(exc, NativeError) else type(exc).__name__
        failure = {'error_category': getattr(exc, 'category', 'adapter_error'),
                   'http_status': getattr(exc, 'http_status', None),
                   'wire_http_status': getattr(exc, 'wire_http_status', None),
                   'upstream_failure': getattr(exc, 'upstream_failure', None)}
        final_text = '[LagentAgentError] ' + type(exc).__name__ + ': ' + issue
    meta = {'native_status': status}
    if status != 'completed':
        meta.update(type='error' if status == 'error' else 'incomplete', reason=issue or 'Turn limit reached')
    journal.emit('terminal', status=status, reason=meta.get('reason'), **failure)
    steps.append({'role': 'assistant', 'content': final_text, 'meta': meta})
    return steps, memory
