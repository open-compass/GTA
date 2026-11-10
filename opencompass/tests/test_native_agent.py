import copy
import json
import pathlib
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from email.utils import format_datetime
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from opencompass.models.native_workflow import NativeLagentAgent, native_agent
from opencompass.models.lagent import LagentAgent


def call(ident, arguments='{"filename":"report.pdf"}'):
    return {'id': ident, 'type': 'function',
            'function': {'name': 'Writer', 'arguments': arguments}}


def reply(content=None, calls=None, finish='stop', **fields):
    message = dict(role='assistant', content=content, **fields)
    if calls is not None:
        message['tool_calls'] = calls
    return NS(status_code=200, json=lambda: {
        'id': 'generation-id', 'choices': [{'finish_reason': finish, 'message': message}]})


def reject(message, status=400):
    return NS(status_code=status, json=lambda: {'error': {'code': status, 'metadata': {
        'raw': json.dumps({'error': {'message': message}})}}})


class Executor:
    def __init__(self, root):
        self.actions, self.arguments, self.root = {}, [], root

    def __call__(self, name, command):
        assert list(self.actions) == [name]
        args = json.loads(command)
        self.arguments.append(args)
        path = self.root / 'files' / args['filename']
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'example content')
        return NS(result=[{'type': 'file', 'content': str(path)}], state=0, errmsg=None, args=args)


class NativeAgentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.executor = Executor(self.root)
        self.llm = NS(path='example/model', url=native_agent._OPENROUTER,
                      keys=['test-only'], orgs=[], invalid_keys=set(), key_ctr=0,
                      gen_params={'reasoning': {'effort': 'low'}, 'max_tokens': 8192},
                      retry=2, temperature=None, wait=Mock(), max_seq_len=100000,
                      get_token_len=lambda _: 10, _record_gta_usage=Mock())
        self.real = NS(description=dict(name='Writer', description='Write a report',
                                        parameters=[dict(name='filename', type='STRING')],
                                        required=['filename']))
        self.dummy = type('FixtureDummy', (), {})
        self.model = NS(tools={'Writer': self.real, 'Unserved': NS(tool=self.dummy())},
                        agent=NS(_max_turn=3, _action_executor=self.executor, _llm=self.llm))
        self.history = []
        self.pending = []

        def post(url, **options):
            self.history.append(copy.deepcopy(options))
            return self.pending.pop(0)

        modules = {'opencompass.models.tool_output': NS(_TASK_ROOT=NS(get=lambda: self.root)),
                   'opencompass.models.lagent': NS(DummyTool=self.dummy),
                   'requests': NS(post=post, RequestException=OSError)}
        self.fake_modules = patch.dict(sys.modules, modules)
        self.fake_modules.start()
        self.addCleanup(self.fake_modules.stop)
        self.resources = [{'type': 'tool', 'name': 'Writer'}, {'type': 'tool', 'name': 'Unserved'}]

    def events(self):
        return [json.loads(line) for line in (self.root / 'native_events.jsonl').read_text().splitlines()]

    def test_optional_class_preserves_ordinary_agent_and_constructor(self):
        self.assertTrue(issubclass(NativeLagentAgent, LagentAgent))
        self.assertIsNot(LagentAgent.chat, native_agent.native_chat)
        with patch.object(LagentAgent, '__init__', return_value=None) as build:
            wrapper = NativeLagentAgent(protocol='configured', native_system_prompt='Evaluate the task.')
        build.assert_called_once_with(protocol='configured')
        self.assertIs(wrapper.chat.__func__, native_agent.native_chat)
        self.assertEqual(wrapper.native_system_prompt, 'Evaluate the task.')

    def test_explicit_system_prompt_is_sent(self):
        self.model.native_system_prompt = 'Evaluate the task.'
        self.pending = [reply('Four.')]
        native_agent.native_chat(self.model, 'What is two plus two?', resources=[])
        self.assertEqual(self.history[0]['json']['messages'][0],
                         {'role': 'system', 'content': 'Evaluate the task.'})

    def test_real_execution_one_per_turn_linked_results_reasoning_and_guard(self):
        signed = dict(call('a'), extra_content={'google': {'thought_signature': 'test-signature'}})
        self.pending = [reply(calls=[signed, call('b')], finish='tool_calls',
                              reasoning_details=[{'opaque': 'provider-only'}]),
                        reply('I created the report.pdf document.')]
        steps, memory = native_agent.native_chat(self.model, 'Create a PDF.', resources=self.resources)
        self.assertEqual(self.executor.arguments, [{'filename': 'report.pdf'}])
        self.assertEqual(self.executor.actions, {})
        self.assertEqual(memory, [{'role': 'user', 'content': 'Create a PDF.'}])
        self.assertEqual(steps[-1]['meta']['native_status'], 'completed')
        self.assertEqual([s['tool_call_id'] for s in steps if s['role'] == 'tool'], ['a', 'b'])
        self.assertEqual([s['executed'] for s in self.events() if s['event'] == 'tool'], [True, False])
        first = self.history[0]['json']
        self.assertFalse(first['parallel_tool_calls'])
        self.assertEqual(first['reasoning'], self.llm.gen_params['reasoning'])
        self.assertEqual(first['max_tokens'], 8192)
        self.assertEqual([t['function']['name'] for t in first['tools']], ['Writer'])
        followup = self.history[1]['json']['messages']
        self.assertIn('reasoning_details', followup[-3])
        self.assertEqual(followup[-3]['tool_calls'][0], signed)
        self.assertEqual([s['tool_call_id'] for s in followup[-2:]], ['a', 'b'])
        self.assertIn('not executed', followup[-1]['content'])
        self.assertNotIn('provider-only', json.dumps(self.events()))
        self.assertNotIn('test-signature', json.dumps(self.events()))
        self.assertNotIn('test-only', json.dumps(self.events()))
        self.assertEqual(self.llm._record_gta_usage.call_count, 2)

    def test_guard_requests_retry_but_does_not_starve_text(self):
        self.pending = [reply('I created the PDF document.'), reply(calls=[call('x')]),
                        reply('Here is your report.pdf.')]
        result, _ = native_agent.native_chat(self.model, 'Create a PDF.', resources=self.resources)
        self.assertEqual(self.history[-1]['json']['tool_choice'], 'none')
        self.assertEqual(result[-1]['meta']['native_status'], 'completed')
        self.assertTrue(any(x['event'] == 'guard_retry' for x in self.events()))
        self.pending = [reply('Four.')]
        result, _ = native_agent.native_chat(self.model, 'What is two plus two?', resources=[])
        self.assertNotIn('tools', self.history[-1]['json'])
        self.assertEqual(result[-1]['content'], 'Four.')

    def test_repeated_id_truncation_and_textual_action_do_not_execute(self):
        self.pending = [reply(calls=[call('same'), call('same')]),
                        reply(calls=[call('partial')], finish='length'),
                        reply('Tool: Writer\nTool Input: {}')]
        result, _ = native_agent.native_chat(self.model, 'Create a PDF.', resources=self.resources)
        self.assertEqual(result[-1]['meta']['native_status'], 'incomplete')
        self.assertEqual(self.executor.arguments, [])

    def test_http_credit_failure_is_immediately_distinguishable(self):
        self.pending = [NS(status_code=402, json=lambda: {'error': {'code': 402,
                           'message': 'insufficient credits, provider detail test-only'}})]
        result, _ = native_agent.native_chat(self.model, 'Create a PDF.', resources=self.resources)
        self.assertEqual(result[-1]['meta']['native_status'], 'error')
        self.assertEqual(self.events()[-1]['http_status'], 402)
        self.assertEqual(self.events()[-1]['error_category'], 'api_error')
        self.assertEqual(self.events()[-1]['upstream_failure'], 'insufficient_credits')
        provider = next(e for e in self.events() if e['event'] == 'provider')
        self.assertEqual((provider['sent_max_tokens'], provider['temperature']), (8192, 0.7))
        self.assertEqual(len(self.history), 1)
        self.assertNotIn('test-only', json.dumps([result, self.events()]))

    def test_only_named_upstream_failure_categories_are_published(self):
        for code, detail, expected in [
            (400, {'metadata': {'raw': '{"message":"missing thought_signature"}'}}, 'thought_signature'),
            (402, {'metadata': {'reason': 'in_flight_budget_exhausted'}}, 'in_flight_budget'),
            (400, {'message': 'maximum context length of 4096'}, 'context_limit'),
            (413, {}, 'request_size'),
            (400, {'message': 'Invalid schema for function Writer'}, 'invalid_tool_schema'),
            (404, {'message': 'No endpoints found that support tool use'}, 'unsupported_tool_capability'),
            (502, {'metadata': {'raw': 'unclassified provider detail'}}, 'unknown'),
        ]:
            with self.subTest(code=code, expected=expected):
                self.assertEqual(native_agent._failure_kind(code, detail), expected)

    def test_guard_exhaustion_keeps_last_answer_marked_incomplete(self):
        self.pending = [reply('I created the PDF document.') for _ in range(3)]
        result, _ = native_agent.native_chat(self.model, 'Create a PDF.', resources=self.resources)
        self.assertEqual(result[-1]['content'], 'I created the PDF document.')
        self.assertEqual(result[-1]['meta']['native_status'], 'incomplete')

    def test_numeric_rejection_changes_only_output_for_chat(self):
        message = 'Maximum context length is 45,000 tokens; requested 39,000 tokens in the messages.'
        self.pending = [reject(message), reply('A.'), reply('B.')]
        journal = native_agent._Journal(self.root)
        messages = [{'role': 'system', 'content': 'Original instruction'},
                    {'role': 'user', 'content': 'Original question'}]
        limits = {}
        native_agent._request(self.llm, messages, [], False, journal, 1, limits)
        native_agent._request(self.llm, messages, [], False, journal, 2, limits)
        first, second, third = [x['json'] for x in self.history]
        self.assertEqual(first['max_tokens'], 8192)
        self.assertEqual(second['max_tokens'], 5936)
        self.assertEqual(second, dict(first, max_tokens=5936))
        self.assertEqual(third, second)
        event = next(e for e in self.events() if e['event'] == 'context_recovery')
        self.assertEqual(event['token_limits'], {'context_tokens': 45000, 'input_tokens': 39000})
        self.assertEqual(event['trimmed_observations'], [])
        self.assertEqual(self.llm.gen_params['max_tokens'], 8192)

    def test_repeated_rejection_trims_only_wire_results_and_preserves_raw_record(self):
        content = 'original-start ' + 'original words ' * 5000 + ' original-end'
        class LargeReader:
            def __call__(self, name, args):
                return NS(state=0, args=json.loads(args), errmsg=None,
                          result=[{'type': 'text', 'content': content}])
        self.model.agent._action_executor = LargeReader()
        self.pending = [reply(calls=[call('read')], reasoning_details=[{'opaque': 'xyz'}]),
                        reject('context length exceeded'), reject('context length exceeded'),
                        reply('Text answer.')]
        question = 'Summarize the supplied material.'
        steps, _ = native_agent.native_chat(self.model, question, resources=self.resources)
        self.assertEqual(steps[-1]['meta']['native_status'], 'completed')
        self.assertEqual(steps[1]['content'][0]['content'], content)
        original, reduced, excerpt = [x['json'] for x in self.history[1:]]
        self.assertEqual(reduced, dict(original, max_tokens=4096))
        self.assertEqual(excerpt['messages'][:-1], original['messages'][:-1])
        self.assertEqual(excerpt['messages'][-1]['tool_call_id'], 'read')
        sent = excerpt['messages'][-1]['content']
        for text in ('characters omitted from this tool observation', 'original-start', 'original-end'):
            self.assertIn(text, sent)
        self.assertLess(len(sent), len(content))
        recovery = [e for e in self.events() if e['event'] == 'context_recovery'][-1]
        self.assertEqual(recovery['trimmed_observations'][0]['wire_chars'], len(sent))
        self.assertNotIn('original-start', json.dumps(self.events()))
        self.assertEqual(sum(e['event'] == 'tool' for e in self.events()), 1)

    def test_unrecoverable_context_is_unscored_and_bounded(self):
        self.pending = [reject('context window exceeded'), reject('context window exceeded')]
        steps, _ = native_agent.native_chat(self.model, 'This user message cannot be shortened.')
        self.assertEqual(steps[-1]['meta']['native_status'], 'error')
        self.assertEqual(len(self.history), 2)
        for sent in self.history:
            self.assertEqual(sent['json']['messages'][1]['content'], 'This user message cannot be shortened.')
        self.assertEqual(self.events()[-1]['upstream_failure'], 'context_limit')

    def test_four_pdf_observations_recover_against_smaller_provider_window(self):
        self.llm.max_seq_len = 131072
        self.llm.gen_params['max_tokens'] = 37267
        self.llm.get_token_len = lambda s: len(s) // 4 + 1
        messages = [{'role': 'system', 'content': 'Unchanged instruction.'},
                    {'role': 'user', 'content': 'Compare the attached PDFs.'}]
        for i in range(4):
            ident = 'pdf_' + str(i)
            messages.append({'role': 'assistant', 'tool_calls': [
                dict(call(ident), function={'name': 'ReadPDF', 'arguments': '{"file":"example.pdf"}'})]})
            messages.append({'role': 'tool', 'tool_call_id': ident, 'content': 'source text ' * 7700})
        before = copy.deepcopy(messages)
        seen = []
        def enforce_context(_url, **kwargs):
            data = kwargs['json']
            inputs = self.llm.get_token_len(json.dumps(data['messages']))
            seen.append((data['max_tokens'], inputs, copy.deepcopy(data['messages'])))
            if inputs + data['max_tokens'] > 65536:
                return reject(f"Maximum context length is 65536 tokens; your messages resulted in {inputs} tokens.")
            return reply('Comparison based on source excerpts.')
        with patch.object(sys.modules['requests'], 'post', enforce_context):
            result, _ = native_agent._request(self.llm, messages, [], False, native_agent._Journal(self.root), 5, {})
        self.assertEqual(result['content'], 'Comparison based on source excerpts.')
        self.assertEqual([r[0] for r in seen], [37267, 4096, 4096])
        self.assertEqual(seen[0][2], seen[1][2])
        self.assertEqual(messages, before)
        for stored, projected in zip(before, seen[-1][2]):
            if stored['role'] == 'tool':
                self.assertEqual(stored['tool_call_id'], projected['tool_call_id'])
                self.assertIn('characters omitted', projected['content'])
            else:
                self.assertEqual(stored, projected)

    def test_qwen_gateway_stops_counting_at_first_excess_token(self):
        limit = 131072
        def detail(inputs):
            return (f'Context length limit: {limit:,} tokens. You requested at least 131,073 tokens. '
                    f'Input length is at least {inputs:,} tokens.')
        # The five previously recorded attempts had unchanged messages, yet
        # the provider reported an increasing input count as output decreased.
        for sent, reported in [(37410, 93663), (37345, 93728), (37280, 93793),
                               (37215, 93858), (37150, 93923)]:
            values = native_agent._budget_numbers({'message': detail(reported)})
            self.assertEqual(values, {'context_tokens': limit, 'input_tokens': reported,
                                      'requested_total_lower_bound': limit + 1})
            self.assertEqual(reported + sent, limit + 1)
        self.llm.gen_params['max_tokens'] = 37410
        messages = [{'role': 'system', 'content': 'Read this document.'},
                    {'role': 'user', 'content': 'List the results.'}]
        for i in range(4):
            ident = 'pdf' + str(i)
            messages.extend([{'role': 'assistant', 'tool_calls': [call(ident)]},
                             {'role': 'tool', 'tool_call_id': ident, 'content': 'pdf text ' * 14000}])
        before = copy.deepcopy(messages)
        seen = []
        def gateway(_url, **kwargs):
            data = kwargs['json']
            seen.append((data['max_tokens'], copy.deepcopy(data['messages'])))
            if data['messages'] == before:
                return reject(detail(limit + 1 - data['max_tokens']))
            return reply('Done.')
        with patch.object(sys.modules['requests'], 'post', gateway):
            msg, _ = native_agent._request(self.llm, messages, [], False, native_agent._Journal(self.root), 5, {})
        self.assertEqual(msg['content'], 'Done.')
        self.assertEqual([sent for sent, _ in seen], [37410, 4096, 4096])
        self.assertEqual(seen[0][1], seen[1][1])
        self.assertEqual(messages, before)
        corrections = [e for e in self.events() if e['event'] == 'context_recovery']
        self.assertEqual([e['action'] for e in corrections], ['reduce_output', 'trim_tool_observations'])
        self.assertEqual(len(corrections[-1]['trimmed_observations']), 4)
        for old, new in zip(before, seen[-1][1]):
            if old['role'] == 'tool':
                self.assertEqual(new['tool_call_id'], old['tool_call_id'])
                self.assertIn('omitted from this tool observation', new['content'])
            else:
                self.assertEqual(new, old)

    def test_scout_embedded_context_400_on_turn_nine_recovers_for_each_envelope(self):
        self.llm.gen_params['max_tokens'] = 37208
        messages = [{'role': 'user', 'content': 'Compare the PDFs.'}]
        for index in range(4):
            ident = 'read_' + str(index)
            messages.extend([{'role': 'assistant', 'tool_calls': [call(ident)]},
                             {'role': 'tool', 'tool_call_id': ident, 'content': 'PDF text ' * 5000}])
        for location in ('outer', 'wrapped', 'choice'):
            with self.subTest(location=location):
                detail = {'code': 400 if location == 'choice' else '400',
                          'message': 'Maximum context length is 131072 tokens.'}
                data = {'error': detail}
                if location == 'wrapped':
                    data = {'data': data}
                elif location == 'choice':
                    data = {'choices': [{'finish_reason': 'error', 'error': detail,
                                         'message': {'content': 'unusable incomplete response'}}]}
                previous = len(self.events()) if (self.root / 'native_events.jsonl').exists() else 0
                self.pending = [NS(status_code=200, json=lambda: data), reply('Compared.')]
                result, _ = native_agent._request(self.llm, messages, [], False,
                                                   native_agent._Journal(self.root), 9, {})
                self.assertEqual(result['content'], 'Compared.')
                events = self.events()[previous:]
                self.assertEqual([e['event'] for e in events], ['provider', 'context_recovery', 'provider'])
                self.assertEqual((events[0]['status'], events[0]['http_status'], events[0]['wire_http_status']),
                                 ('api_error', 400, 200))
                self.assertEqual((events[1]['http_status'], events[1]['wire_http_status']), (400, 200))
                self.assertEqual((events[0]['sent_max_tokens'], events[2]['sent_max_tokens']), (37208, 4096))
                self.assertEqual(self.history[-1]['json']['messages'], self.history[-2]['json']['messages'])

    def test_noncontext_embedded_error_is_unscored_and_actual_wire_failures_keep_precedence(self):
        self.pending = [NS(status_code=200, json=lambda: {'choices': [{'error': {
            'code': '402', 'message': 'Insufficient credits'}, 'message': {'content': 'Do not accept.'}}]})]
        steps, _ = native_agent.native_chat(self.model, 'Summarize.')
        self.assertEqual(steps[-1]['meta']['native_status'], 'error')
        self.assertNotIn('Do not accept', steps[-1]['content'])
        self.assertEqual((self.events()[-1]['http_status'], self.events()[-1]['wire_http_status']), (402, 200))
        self.assertEqual(len(self.history), 1)
        self.assertEqual(native_agent._effective_error_status(503, {'code': '400'}), 503)
        for code in ('400 bad request', '399', '600', 400.0, True, None):
            self.assertEqual(native_agent._effective_error_status(200, {'code': code}), 200)

    def test_gemini_unknown_400_saves_only_structural_diagnostic_without_retry(self):
        self.llm.path = 'google/gemini-2.5-pro'
        secret_marker = 'private-fixture-data'
        actual = dict(call('call-1'), extra_content={'google': {'thought_signature': secret_marker}})
        messages = [{'role': 'user', 'content': secret_marker},
                    {'role': 'assistant', 'tool_calls': [actual], 'reasoning_details': [{'data': secret_marker}]},
                    {'role': 'tool', 'tool_call_id': 'call-1', 'content': secret_marker}]
        nested = {'error': {'status': 'INVALID_ARGUMENT', 'message': 'FunctionResponse must come immediately '
                            'after a matching function call: ' + secret_marker}}
        detail = {'code': 400, 'message': 'Provider returned error',
                  'metadata': {'provider_name': 'Google AI Studio', 'raw': json.dumps(nested),
                               'credential': secret_marker}}
        self.pending = [NS(status_code=400, json=lambda: {'error': detail})]
        with self.assertRaises(native_agent.ProviderError) as raised:
            native_agent._request(self.llm, messages, [], False, native_agent._Journal(self.root), 9, {})
        self.assertEqual(raised.exception.upstream_failure, 'unknown')
        self.assertEqual(len(self.history), 1)
        event, = self.events()
        diagnostics = event['gemini_validation']
        self.assertEqual(diagnostics['provider'], 'Google AI Studio')
        self.assertEqual(diagnostics['status'], 'INVALID_ARGUMENT')
        self.assertEqual(diagnostics['raw_format'], 'object')
        self.assertEqual(diagnostics['signals'], ['function_response', 'turn_order'])
        self.assertEqual(diagnostics['calls_with_google_signature'], 1)
        self.assertEqual(diagnostics['tool_rounds_with_reasoning_details'], 1)
        self.assertEqual(diagnostics['matched_tool_results'], 1)
        self.assertEqual(diagnostics['pending_tool_results'], 0)
        self.assertNotIn(secret_marker, json.dumps(event))
        self.assertNotIn('call-1', json.dumps(event))

    def test_tls_eof_retries_identical_request_and_auth_with_jitter(self):
        class FakeConnectionError(OSError):
            pass
        class FakeSSL(FakeConnectionError):
            pass
        http = sys.modules['requests']
        http.exceptions = NS(SSLError=FakeSSL, Timeout=TimeoutError, ConnectionError=FakeConnectionError,
                             ChunkedEncodingError=type('Incomplete', (OSError,), {}))
        self.llm.keys = ['fixture-key-a', 'fixture-key-b']
        self.llm.orgs, self.llm.org_ctr = ['fixture-org-a', 'fixture-org-b'], 0
        self.llm.retry = 3
        failure = FakeSSL('UNEXPECTED_EOF_WHILE_READING, fixture-private-text')
        with patch.object(http, 'post', side_effect=[failure, failure, reply('OK')]) as send, \
                patch.object(native_agent.random, 'uniform', side_effect=[0.5, 1.5]) as jitter, \
                patch.object(native_agent.time, 'sleep') as wait:
            steps, _ = native_agent.native_chat(self.model, 'Summarize.')
        self.assertEqual(steps[-1]['meta']['native_status'], 'completed')
        self.assertEqual(send.call_count, 3)
        self.assertEqual(send.call_args_list, [send.call_args_list[0]] * 3)
        self.assertEqual([args.args for args in jitter.call_args_list], [(0, 2), (0, 4)])
        self.assertEqual([args.args for args in wait.call_args_list], [(2.5,), (5.5,)])
        self.assertNotIn('fixture-', json.dumps(self.events()))
        self.assertEqual([e['reason'] for e in self.events() if e['event'] == 'request_backoff'],
                         ['tls_unexpected_eof'] * 2)

    def test_retry_after_caps_http_dates_and_does_not_wait_after_the_last_attempt(self):
        now = 1700000000
        date = format_datetime(datetime.fromtimestamp(now + 600, timezone.utc), usegmt=True)
        response = NS(status_code=503, headers={'Retry-After': date}, json=lambda: {'error': {'code': 503}})
        with patch.object(sys.modules['requests'], 'post', return_value=response) as send, \
                patch.object(native_agent.time, 'time', return_value=now), \
                patch.object(native_agent.time, 'sleep') as wait:
            steps, _ = native_agent.native_chat(self.model, 'Compute.')
        self.assertEqual(send.call_count, self.llm.retry)
        wait.assert_called_once_with(60)
        self.assertEqual(steps[-1]['meta']['native_status'], 'error')
        self.assertEqual(self.events()[-1]['http_status'], 503)

    def test_429_retry_after_and_408_transient_but_hard_failures_do_not_wait(self):
        def http_error(code, body):
            return NS(status_code=code, headers={'Retry-After': '7'}, json=lambda: {'error': body})
        for code in (408, 429):
            with self.subTest(code=code), patch.object(sys.modules['requests'], 'post', side_effect=[
                    http_error(code, {'code': code}), reply('OK')]) as send, \
                    patch.object(native_agent.random, 'uniform', return_value=0), \
                    patch.object(native_agent.time, 'sleep') as wait:
                steps, _ = native_agent.native_chat(self.model, 'Compute.')
                self.assertEqual(steps[-1]['meta']['native_status'], 'completed')
                wait.assert_called_once_with(7)
                self.assertEqual(send.call_count, 2)
        self.llm.keys = ['test-a', 'test-b']
        for code, body in [(429, {'code': 'insufficient_quota'}), (429, {'code': 429, 'message': 'Account quota exceeded'}),
                           (402, {'code': 402}), (403, {'code': 403}), (500, {'code': 'invalid_api_key'}),
                           (501, {'code': 501})]:
            self.llm.invalid_keys.clear()
            with self.subTest(code=code, body=body), patch.object(sys.modules['requests'], 'post',
                    return_value=http_error(code, body)) as send, patch.object(native_agent.time, 'sleep') as wait:
                steps, _ = native_agent.native_chat(self.model, 'Compute.')
                self.assertEqual(steps[-1]['meta']['native_status'], 'error')
                self.assertEqual(send.call_count, 1)
                wait.assert_not_called()

    def test_deterministic_tls_failure_is_not_retried(self):
        class FakeSSL(OSError):
            pass
        http = sys.modules['requests']
        http.exceptions = NS(SSLError=FakeSSL, ConnectionError=OSError, Timeout=TimeoutError,
                             ChunkedEncodingError=EOFError)
        with patch.object(http, 'post', side_effect=FakeSSL('CERTIFICATE_VERIFY_FAILED')) as send, \
                patch.object(native_agent.time, 'sleep') as wait:
            steps, _ = native_agent.native_chat(self.model, 'Compute.')
        self.assertEqual(steps[-1]['meta']['native_status'], 'error')
        self.assertEqual(send.call_count, 1)
        wait.assert_not_called()
        self.assertEqual(self.events()[-1]['error_category'], 'transport_configuration_error')

    def test_output_cap_and_unrelated_400(self):
        self.pending = [reject('max_tokens must be between 1 and 1024'), reply('OK')]
        result, _ = native_agent.native_chat(self.model, 'Compute a sum.')
        self.assertEqual(result[-1]['meta']['native_status'], 'completed')
        self.assertEqual(self.history[-1]['json']['max_tokens'], 1024)
        self.pending = [reject('max_tokens must be an integer')]
        result, _ = native_agent.native_chat(self.model, 'Compute a sum.')
        self.assertEqual(result[-1]['meta']['native_status'], 'error')
        self.assertEqual(len(self.history), 3)

    def test_local_nonpositive_estimate_only_trims_tool_content(self):
        self.llm.get_token_len = len
        self.llm.max_seq_len = 10000
        messages = [{'role': 'user', 'content': 'original-user'},
                    {'role': 'tool', 'tool_call_id': 'seen', 'content': 's' * 22000}]
        self.pending = [reply('OK')]
        native_agent._request(self.llm, messages, [], False, native_agent._Journal(self.root), 1, {})
        self.assertEqual(len(self.history), 1)
        sent = self.history[0]['json']['messages']
        self.assertEqual(sent[0], messages[0])
        self.assertLess(len(sent[-1]['content']), len(messages[-1]['content']))
        self.assertEqual(messages[-1]['content'], 's' * 22000)
        result, _ = native_agent.native_chat(self.model, 'u' * 20000)
        self.assertEqual(result[-1]['meta']['native_status'], 'error')
        self.assertEqual(len(self.history), 1)
        self.assertEqual(self.events()[-1]['error_category'], 'local_context_budget')

    def test_guard_requires_only_next_action_and_capability_fallback_is_logged(self):
        def guard(**kw):
            self.assertEqual(kw['available_tools'], ['Writer'])
            return None if kw['used_tools'] else 'Use the available writer to create the file.'
        self.model.agent._max_turn = 4
        self.pending = [reply('I created the file.'), reject("tool_choice 'required' is not supported."),
                        reply(calls=[call('writer')], finish='tool_calls'), reply('Completed.')]
        with patch('opencompass.models.native_workflow.native_guard.final_guard', side_effect=guard):
            steps, _ = native_agent.native_chat(self.model, 'Write a file.', resources=self.resources)
        self.assertEqual(steps[-1]['meta']['native_status'], 'completed')
        self.assertEqual([x['json']['tool_choice'] for x in self.history], ['auto', 'required', 'auto', 'auto'])
        self.assertEqual(len(self.executor.arguments), 1)
        fallback = [e for e in self.events() if e['event'] == 'tool_choice_fallback']
        self.assertEqual(len(fallback), 1)
        self.assertEqual(fallback[0]['http_status'], 400)
        self.assertEqual(self.history[1]['json']['messages'], self.history[2]['json']['messages'])

    def test_required_does_not_retry_unrelated_error_or_execute_a_pseudo_call(self):
        self.pending = [reply('[Writer(filename="report.pdf")]'), reject('Invalid schema: expected object')]
        with patch('opencompass.models.native_workflow.native_guard.final_guard', return_value='A real tool result is required.'):
            steps, _ = native_agent.native_chat(self.model, 'Write a file.', resources=self.resources)
        self.assertEqual(steps[-1]['meta']['native_status'], 'error')
        self.assertEqual(self.executor.arguments, [])
        self.assertEqual([x['json']['tool_choice'] for x in self.history], ['auto', 'required'])
        self.assertFalse(any(e['event'] == 'tool_choice_fallback' for e in self.events()))

    def test_real_guard_rejects_scout_style_pseudo_read_then_accepts_observed_read(self):
        filepath = str(self.root / 'input_report.pdf')
        class Reader:
            def __call__(self, name, command):
                return NS(state=0, args=json.loads(command), errmsg=None,
                          result=[{'type': 'text', 'content': 'Report total: 42.'}])
        self.model.agent._action_executor = Reader()
        self.model.agent._max_turn = 4
        self.model.tools['ReadPDF'] = NS(description=dict(description='Read a PDF',
            parameters=[dict(name='file', type='STRING')], required=['file']))
        resources = [{'type': 'file', 'path': filepath}, {'type': 'tool', 'name': 'ReadPDF'}]
        real = {'id': 'read1', 'type': 'function',
                'function': {'name': 'ReadPDF', 'arguments': json.dumps({'file': filepath})}}
        self.pending = [reply('[ReadPDF(file="input_report.pdf")]'),
                        reply(calls=[real], finish='tool_calls'), reply('The report total is 42.')]
        steps, _ = native_agent.native_chat(self.model, 'Summarize the supplied input_report.pdf.',
                                            resources=resources)
        self.assertEqual(steps[-1]['meta']['native_status'], 'completed')
        self.assertEqual([x['json']['tool_choice'] for x in self.history], ['auto', 'required', 'auto'])
        self.assertEqual([s['tool_call_id'] for s in steps if s['role'] == 'tool'], ['read1'])

    def test_guard_never_forces_final_only_or_general_text_turn(self):
        self.model.agent._max_turn = 2
        self.pending = [reply('I created the file.'), reply('I could not complete it.')]
        with patch('opencompass.models.native_workflow.native_guard.final_guard', return_value='Use the available writer to create the file.'):
            native_agent.native_chat(self.model, 'Write a PDF.', resources=self.resources)
        self.assertEqual([x['json']['tool_choice'] for x in self.history], ['auto', 'none'])
        self.pending = [reply('Plain text answer.')]
        with patch('opencompass.models.native_workflow.native_guard.final_guard', return_value=None):
            result, _ = native_agent.native_chat(self.model, 'Explain a for-loop.', resources=self.resources)
        self.assertEqual(self.history[-1]['json']['tool_choice'], 'auto')
        self.assertEqual(result[-1]['meta']['native_status'], 'completed')


if __name__ == '__main__':
    unittest.main()
