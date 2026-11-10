"""Compare an explicitly requested operation with this task's observed evidence."""
import json
import re
from pathlib import Path


FORMATS = {
    'pdf': ('pdf',), 'docx': ('docx',), 'word document': ('docx', 'doc'),
    'pptx': ('pptx',), 'powerpoint': ('pptx', 'ppt'), 'xlsx': ('xlsx',),
    'excel workbook': ('xlsx', 'xls'), 'csv': ('csv',), 'mp4': ('mp4',),
    'mp3': ('mp3',), 'wav': ('wav',), 'html': ('html', 'htm'),
    'png': ('png',), 'jpeg': ('jpeg', 'jpg'), 'jpg': ('jpeg', 'jpg'), 'zip': ('zip',),
}
FORMAT = re.compile(r'\b(' + '|'.join(FORMATS) + r')\b', re.I)
INSTRUCTION_ONLY = re.compile(
    r'\b(?:example|sample|demo|syntax|pseudocode|snippet) (?:of (?:a |an |the )?)?'
    r'(?:native |tool |function |API )?(?:calls?|code|syntax|commands?|json|payload)'
    r'|\b(?:write|provide|show|give|generate|return|create)\s+(?:an? |the |some )?'
    r'(?:[\w+-]+\s+)?(?:code|script|program|snippet|instructions?|tutorial|example) (?:that |to |for )'
    r'|\b(?:explain|show|describe|demonstrate|teach)(?: me)? how (?:to|I (?:can|could))'
    r'|\b(?:in|as) (?:an? )?(?:example|code snippet|code block|pseudocode)\b'
    r'|\b(?:return|give|provide|show|write) (?:only |just )?(?:the )?(?:code|syntax|pseudo-?code)\b',
    re.I)
INLINE = re.compile(r'\b(?:inline|in (?:the |your )?(?:reply|response|chat)|as plain text)\b', re.I)
OUTPUT = re.compile(
    r'\b(?:creat(?:e|ing)|generat(?:e|ing)|produc(?:e|ing)|deliver(?:ed)?|export(?:ed)?|'
    r'compile[d]? into|save[d]?|attach|submit|provide|'
    r'(?:assembled?|integrated?|compress(?:ed)?|combined?|packaged?|compiled?)\b[^.;!\n]{0,180}?\binto|'
    r'(?:final )?(?:deliverables?|output files?)\s+(?:is|are|include[s]?|consist[s]? of|'
    r'(?:must|should) (?:be|include)))\b',
    re.I)
INPUT_BOUNDARY = re.compile(
    r'\b(?:from|using|based on|provided in|in the attached|'
    r'of (?:the |my |our |this |these |an? )?(?:attached |provided |input )?'
    r'(?:pdf|csv|xlsx|html|file|document|spreadsheet|workbook)(?:s)?\b)', re.I)
PASSIVE_FILE = re.compile(
    r'\b(?:should|must|will|is to|are to)\s+be\s+(?:saved|exported|delivered|generated|produced|'
    r'submitted|attached)\b', re.I)
FILE_DELIVERY = re.compile(
    r'\b(?:standalone|stand-alone|downloadable)\s+(?:[\w-]+\s+){0,2}'
    r'(?:report|document|presentation|handbook|spreadsheet|file)\b'
    r'|\b(?:file|report|document|workbook|presentation)\b.{0,35}\b(?:ready|available) for download\b'
    r'|\b(?:deliverables?|outputs?)\s+(?:must be|are|is|include|includes|consist of|consists of)\s+'
    r'(?:an? |the |two |three )?(?:separate |standalone |downloadable )?(?:files?|workbook|presentation)\b'
    r'|\b(?:file|document|workbook|archive|presentation)\s+(?:named|called)\b'
    r'|\b(?:deliverables?|deliver|submit|save|export|download|compress)\b[^.!?;\n]{0,95}'
    r'\b(?:files?|archive|document|workbook)\b', re.I)
CLAIM = re.compile(
    r'\b(?:I|we)(?:[’\x27]ve| have)?\s+(?:successfully\s+)?'
    r'(?:created|generated|produced|attached|saved|exported|uploaded)\b'
    r'|\b(?:files?|documents?|videos?|slides?|workbooks?)\s+(?:["“][^"\n”]+["”]\s+)?'
    r'(?:have been|has been|were|was)\s+'
    r'(?:created|generated|attached|saved|uploaded)\b|\]\(sandbox:/'
    r'|\bdownload\s+(?:your|the|my)\s+(?:file|document|video|workbook)\b', re.I)
FILE_WORDS = re.compile(r'\b(?:files?|attachments?|documents?|videos?|workbooks?|slides?)\b', re.I)
LIMIT = re.compile(r'\b(?:unable to|could(?:n[’\x27]t| not)|cannot|can[’\x27]t|failed to|'
                   r'not (?:created|generated|available|accessible)|no (?:access|attachment))\b', re.I)
WEB = re.compile(
    r'\b(?:web|internet|online) search(?:es)?\b|\bsearch(?:ing)? (?:online|(?:the )?(?:web|internet))\b'
    r'|\b(?:look up|research|verify|find|search for)\b[^.!?;\n]{0,65}\b(?:online|on the web)\b'
    r'|\b(?:consult|visit|check|retrieve from)\s+(?:the )?(?:official |public )?websites?\b', re.I)
INPUT_ACTION = re.compile(r'\b(?:analy[sz](?:e|ing|is)|read(?:ing)?|review(?:ing)?|summari[sz](?:e|ing)|'
                          r'summar(?:y|ies)|extract(?:ing)?|compar(?:e|ing|ison)|transcrib(?:e|ing)|'
                          r'convert(?:ing)?|process(?:ing)?|inspect(?:ing)?|according to|'
                          r'based on|from|using|use)\b', re.I)
INPUT_REFERENCE = re.compile(r'\b(?:attached|provided|supplied|given|input|uploaded|these|the)\s+'
                             r'(?:[\w-]+\s+){0,2}(?:files?|documents?|images?|photos?|videos?|audio|'
                             r'recordings?|spreadsheets?|workbooks?|data(?:sets?)?|PDFs?|CSVs?)\b', re.I)
PLAN = re.compile(r'^(?:(?:first(?:ly)?|next|now|to start|to begin)[,:]?\s+)?'
                  r'(?:I (?:will|shall|need to|am going to)|I[’\x27]ll|let me)\b', re.I)
EXAMPLE_LEAD = re.compile(r'\b(?:for example\b|sample (?:call|syntax)\b|illustrative (?:call|example)\b|'
                          r'(?:example|syntax|notation|code snippet|quoted call)(?:\s+is|\s+would be)?\s*:|'
                          r'(?:call|syntax|example) (?:looks|would look) like\b):?', re.I)
PLACEHOLDER = re.compile(r'\[\s*(?:insert|attach|add|link to)\s+(?:the |your |a |an )?'
                         r'(?:(?:pdf|docx|csv|xlsx|html|zip|output|download)\s+)?'
                         r'(?:file|report|document|attachment|link)(?:\s+here)?\s*\]', re.I)
PROMISE = re.compile(r'\b(?:I(?:[’\x27]ll| will)|this will|it will|we will)\s+(?:generate|create|save|'
                     r'produce|export|compress|zip)\b[^.!?\n]{0,100}', re.I)


def clauses(prompt):
    return [c.strip() for c in re.split(r'(?<=[.!?;])\s+|\n+', prompt) if c.strip()]


def requested_outputs(parts):
    """Keep source filenames out of the output check. Alternative suffixes form one group."""
    groups, generic = set(), False
    for clause in parts:
        if INSTRUCTION_ONLY.search(clause) or INLINE.search(clause):
            continue
        generic |= bool(FILE_DELIVERY.search(clause))
        objects = [INPUT_BOUNDARY.split(clause[m.end():], 1)[0] for m in OUTPUT.finditer(clause)]
        passive = PASSIVE_FILE.search(clause)
        if passive:
            # "The CSV report should be saved", in addition to "saved as CSV".
            subject = re.split(r'[,;:]', clause[:passive.start()])[-1]
            objects.append(subject)
            objects.append(INPUT_BOUNDARY.split(clause[passive.end():], 1)[0])
            generic |= bool(re.search(r'\b(?:report|document|output|file)s?\b', subject, re.I))
        for obj in objects:
            matches = {FORMATS[word.lower()] for word in FORMAT.findall(obj)}
            if re.search(r'\bor\b', obj, re.I) and len(matches) > 1:
                groups.add(tuple(sorted({ext for choice in matches for ext in choice})))
            else:
                groups.update(matches)
    return groups, generic


def input_names(parts, files):
    names = {str(p).replace('\\', '/').rsplit('/', 1)[-1].casefold() for p in files if p}
    for part in parts:
        if INSTRUCTION_ONLY.search(part) or not INPUT_ACTION.search(part):
            continue
        folded = part.casefold()
        mentions = {n for n in names if n in folded}
        if mentions or INPUT_REFERENCE.search(part):
            return mentions or names
    return set()


def strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, str):
        yield value


def input_observed(steps, used_tools, names):
    for step in steps:
        if step.get('role') != 'assistant' or step.get('error'):
            continue
        for call in step.get('tool_calls') or []:
            fn = call.get('function', {})
            tool = fn.get('name', '')
            # A path pasted into search text or document content does not open it.
            if tool not in used_tools or tool.endswith('Generator') or tool in ('GoogleSearch', 'Calculator', 'Solver'):
                continue
            args = fn.get('arguments')
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    pass
            for val in strings(args):
                basename = val.strip().replace('\\', '/').rsplit('/', 1)[-1].casefold()
                if basename in names:
                    return True
    return False


def unquoted_lines(text):
    """Mask examples locally, preserving independent operative lines elsewhere in a reply."""
    fence = None
    for line in text.splitlines():
        stripped = line.strip()
        marker = re.match(r'^(\x60{3,}|~{3,})', stripped)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            yield ''
        elif fence or stripped.startswith(('>', '"', '“', '\x27')):
            yield ''
        else:
            yield re.sub(r'(\x60+).*?\1', ' ', stripped).strip()


def json_call(body, tools):
    try:
        envelope = json.loads(body)
    except ValueError:
        return None
    if isinstance(envelope, dict) and 'tool_calls' in envelope:
        envelope = envelope['tool_calls']
    for call in envelope if isinstance(envelope, list) else [envelope]:
        if not isinstance(call, dict):
            continue
        fn = call.get('function', call)
        if isinstance(fn, dict) and 'arguments' in fn and fn.get('name') in tools:
            return fn['name']
    return None


def pseudo_tool(text, tools):
    """Recognize an operative registry call in prose without ever executing printed syntax."""
    lines = list(unquoted_lines(text))
    previous = ''
    for i, line in enumerate(lines):
        if not line:
            continue
        example = bool(EXAMPLE_LEAD.search(previous) or EXAMPLE_LEAD.search(line))
        previous = line
        if example:
            continue
        marker = re.search(r'\[([\w.-]+)\s*\(', line)
        if marker and marker.group(1) in tools:
            prefix = line[:marker.start()].strip()
            if not prefix or prefix in ('-', '*') or PLAN.match(prefix) or prefix.lower().endswith('answer is:'):
                # Calls can span lines or contain a comma/semicolon-separated group.
                if re.search(r'\)\s*\]\s*[.!]?\s*$', line) or any(
                        re.search(r'\)\s*\]\s*[.!]?\s*$', tail) for tail in lines[i+1:] if tail):
                    return marker.group(1)
        legacy = re.fullmatch(r'Tool:[ \t]*([\w.-]+)', line)
        if (legacy and legacy.group(1) in tools and i+1 < len(lines)
                and re.fullmatch(r'Tool Input:\s*\{.*\}', lines[i+1])):
            return legacy.group(1)
        if line.startswith(('{', '[')):
            name = json_call(line, tools)
            if name:
                return name
    return json_call('\n'.join(lines).strip(), tools) if lines else None


def final_guard(*, prompt, text, used_tools, files, input_files=(), available_tools=(), tool_steps=()):
    """Return a corrective turn for a concrete request lacking execution evidence."""
    parts = clauses(prompt)
    operative = [c for c in parts if not INSTRUCTION_ONLY.search(c)
                 and not re.match(r'^(?:do not|don[’\x27]t|never|avoid|optionally|if desired)\b', c, re.I)]
    if not operative:
        return None
    tools, used = set(available_tools), set(used_tools)
    named = set()
    for name in tools:
        command = r'\b(?:use|run|invoke|call)\s+(?:the\s+)?[\x60"]?' + re.escape(name) + r'\b'
        if any(re.search(command, c, re.I) for c in operative):
            named.add(name)
    if 'GoogleSearch' in tools and any(WEB.search(c.split(' without ')[0]) for c in operative):
        named.add('GoogleSearch')
    groups, file_required = requested_outputs(operative)
    refs = input_names(operative, input_files)
    present = {Path(p).suffix.lstrip('.').lower() for p in files}
    missing = sorted('/'.join(group) for group in groups if not present.intersection(group))
    claimed_file = bool(CLAIM.search(text) and (groups or FILE_WORDS.search(text) or FORMAT.search(text)))
    proposed = pseudo_tool(text, tools)
    if proposed and not (INSTRUCTION_ONLY.search(prompt) and not (groups or file_required or named or refs)):
        return (f'The printed {proposed} call did not execute. Send the function call through the native '
                'tool interface, wait for its returned observation, then continue the task.')
    visible = '\n'.join(unquoted_lines(text))
    if groups or file_required:
        gaps = [m.group() for pattern in (PLACEHOLDER, PROMISE) for m in pattern.finditer(visible)]
        if any(any(not present.intersection(FORMATS[fmt.lower()]) for fmt in FORMAT.findall(gap))
               or not files for gap in gaps):
            return ('A file placeholder or future action does not provide the requested deliverable. '
                    'Create the file with an available tool and report its actual output.')
    failed = any(step.get('error', {}).get('type') == 'tool_error' for step in tool_steps
                 if isinstance(step.get('error'), dict))
    # With a genuine attempted failure (or no real functions), a truthful
    # limitation is a meaningful final answer. A mere plan is still actionable.
    if LIMIT.search(text) and not claimed_file and (failed or used or not tools):
        return None
    if refs and not input_observed(tool_steps, used, refs):
        issue = ('Use an available tool to open and inspect the supplied input for this task. '
                 'Pass its actual file path and use the returned observation before answering.')
    elif named - used:
        issue = ('Perform the explicitly requested operation using a native function call to '
                 + ', '.join(sorted(named - used)) + '. Base your answer on the returned result.')
    elif missing or ((file_required or claimed_file) and not files):
        expected = ', '.join(missing) if missing else 'the requested file'
        issue = (f'The task workspace has no recorded output for {expected}. '
                 'Use an available tool to create and check it, or report what prevented completion.')
    else:
        return None
    return issue
