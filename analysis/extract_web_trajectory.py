#!/usr/bin/env python3
"""Extract persisted Web-search calls and responses from an OpenCompass run."""

import argparse
import ast
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_arguments(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    text = value.strip()
    candidates = [text]
    if '\n' in text:
        candidates.append(text.split('\n', 1)[0].strip())
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return {}


def messages(turn):
    if isinstance(turn, dict):
        yield turn
    elif isinstance(turn, list):
        for item in turn:
            yield from messages(item)


FIELDS = ('model', 'sample_index', 'task_id', 'prediction_file', 'step_index',
          'tool_name', 'tool_call_id', 'query', 'k', 'arguments_raw',
          'response_saved', 'search_response', 'artifact_mtime_utc')


def extract(run_dir):
    for path in sorted((run_dir / 'predictions').glob('*/*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        stamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        if not isinstance(data, dict):
            continue
        for index, sample in data.items():
            if not isinstance(sample, dict):
                continue
            turns = sample.get('prediction', [])
            if not isinstance(turns, list):
                continue
            if turns and isinstance(turns[0], dict):
                turns = [turns]
            step_index = 0
            for turn in turns:
                pending, rows = [], []
                for msg in messages(turn):
                    step_index += 1
                    role = msg.get('role')
                    if role != 'tool':
                        pending = []
                    if role == 'assistant':
                        for call in msg.get('tool_calls') or ():
                            fn = call.get('function', {})
                            name, raw = fn.get('name'), fn.get('arguments')
                            row = None
                            if name in ('GoogleSearch', 'WebSearch'):
                                args = parse_arguments(raw)
                                row = dict(model=path.parent.name, sample_index=index,
                                           task_id=sample.get('task_id', sample.get('id', '')),
                                           prediction_file=str(path.relative_to(run_dir)),
                                           step_index=step_index, tool_name=name,
                                           tool_call_id=call.get('id', ''), query=args.get('query', ''),
                                           k=args.get('k', ''), arguments_raw=raw if isinstance(raw, str)
                                           else json.dumps(raw, ensure_ascii=False),
                                           response_saved=False, search_response='',
                                           artifact_mtime_utc=stamp)
                                rows.append(row)
                            pending.append((call.get('id'), name, row))
                    elif role == 'tool':
                        ident, name = msg.get('tool_call_id'), msg.get('name')
                        if ident:
                            matches = [p for p in pending if p[0] == ident]
                        elif name:
                            matches = [p for p in pending if p[1] == name]
                        else:
                            matches = pending[:]
                        if len(matches) == 1:
                            match = matches[0]
                            pending.remove(match)
                            row, body = match[2], msg.get('content')
                            if row is not None and body is not None:
                                row['search_response'] = (body if isinstance(body, str)
                                                          else json.dumps(body, ensure_ascii=False))
                                row['response_saved'] = True
                yield from rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not (args.run_dir / 'predictions').is_dir():
        parser.error('the run must contain a predictions directory')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(extract(args.run_dir))


if __name__ == '__main__':
    main()
