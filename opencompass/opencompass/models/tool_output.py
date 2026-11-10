"""Keep ordinary GTA inference artifacts next to its predictions.

Only an explicitly configured inferencer activates routing. Prompts, ReAct
parsing, history and tool implementations are unchanged. This is artifact
routing, not a sandbox for arbitrary code executed by a tool.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
import os
from pathlib import Path
import shutil
import tempfile
import types
from urllib.parse import quote


_TASK_ROOT = ContextVar('gta_tool_output_task', default=None)
_CALL_ROOT = ContextVar('gta_tool_output_call', default=None)


def path_component(value):
    """Encode identifiers without merging distinct model/dataset/task names."""
    value = str(value)
    if not value or value in ('.', '..'):
        raise ValueError('Expected a nonempty output path component')
    return quote(value, safe='-_.')


def _install_temp_paths():
    # IO serializers import this alias in agentlego.types. Keep their original
    # encoders and extensions; only replace allocation while inside our scope.
    import agentlego.types as io_types
    original = io_types.temp_path
    if getattr(original, '_gta_output_routing', False):
        return

    @wraps(original)
    def scoped_temp_path(category, suffix, prefix='', root=None):
        call_root = _CALL_ROOT.get()
        if call_root is None or root is not None:
            return original(category, suffix, prefix=prefix, root=root)
        directory = call_root / 'generated' / path_component(category)
        directory.mkdir(parents=True, exist_ok=True)
        # Exclusive allocation removes the original four-hex-character name
        # collision risk, including multiple outputs from the same tool call.
        if any(char in suffix for char in ('/', '\\', '\x00')):
            raise ValueError('Unexpected file suffix')
        fd, path = tempfile.mkstemp(
            dir=directory, prefix=quote(prefix, safe='-_.'), suffix=suffix)
        os.close(fd)
        return path

    scoped_temp_path._gta_output_routing = True
    io_types.temp_path = scoped_temp_path


@contextmanager
def task_output_scope(output_dir, task_id):
    """Give one sample invocation a durable, non-overwriting attempt directory."""
    if output_dir is None:
        yield None
        return
    parent = Path(output_dir).resolve() / ('task_' + path_component(task_id))
    parent.mkdir(parents=True, exist_ok=True)
    task_root = Path(tempfile.mkdtemp(prefix='attempt_', dir=parent))
    _install_temp_paths()
    token = _TASK_ROOT.set(task_root)
    try:
        yield task_root
    finally:
        _TASK_ROOT.reset(token)


def scoped_tool_outputs(method):
    """Wrap an AgentInferencer sample without changing its messages or history."""
    @wraps(method)
    def wrapped(self, chat, index, output_handler):
        root = getattr(self, '_tools_output_dir', None)
        task_id = chat.get('id', index) if isinstance(chat, dict) else index
        if task_id is None:
            task_id = index
        with task_output_scope(root, task_id) as task_root:
            try:
                return method(self, chat, index, output_handler)
            finally:
                if task_root is not None and str(index) in output_handler.results_dict:
                    output_handler.results_dict[str(index)]['tools_output_dir'] = str(task_root)
    return wrapped


def _filename(value):
    # A model-supplied absolute filename would otherwise bypass output_dir in
    # os.path.join. Preserve the basename, including Unicode and extensions.
    name = str(value).replace('\\', '/').rsplit('/', 1)[-1]
    if name in ('', '.', '..') or '\x00' in name:
        raise ValueError('Expected a valid output filename')
    return name


def _retain_returned_files(action_return, call_root):
    """Retain already-materialized typed outputs (e.g. a local tool's image)."""
    for item in getattr(action_return, 'result', None) or ():
        if not isinstance(item, dict) or item.get('type') not in ('image', 'audio', 'video', 'file'):
            continue
        value = item.get('content')
        if not isinstance(value, str):
            continue
        source = Path(value)
        if not source.is_file():
            continue
        source = source.resolve()
        # Files serialized by RemoteTool already live in this unique call.
        if call_root in source.parents:
            continue
        directory = call_root / 'generated' / 'file'
        directory.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=directory, suffix=source.suffix)
        os.close(fd)
        shutil.copyfile(source, name)
        item['content'] = name


def bind_tool_output(action):
    """Route an AgentLego LagentTool at execution time; leave other actions alone."""
    from agentlego.tools.wrappers.lagent import LagentTool
    from agentlego.tools.remote import RemoteTool
    if not isinstance(action, LagentTool) or getattr(action, '_gta_output_routing', False):
        return action
    original_run = action.run

    @wraps(original_run)
    def run(self, **kwargs):
        task_root = _TASK_ROOT.get()
        if task_root is None:
            return original_run(**kwargs)
        calls = task_root / 'calls'
        calls.mkdir(parents=True, exist_ok=True)
        call_root = Path(tempfile.mkdtemp(prefix=path_component(self.name) + '_', dir=calls))
        effective = dict(kwargs)
        routing = {'tool': self.name, 'task_output_dir': str(task_root)}
        if isinstance(self.tool, RemoteTool) and 'output_dir' in self.tool.arguments:
            original_dir = task_root / 'files'
            original_dir.mkdir(parents=True, exist_ok=True)
            effective['output_dir'] = str(original_dir)
            routing['requested_output_dir'] = kwargs.get('output_dir')
            routing['effective_output_dir'] = effective['output_dir']
            if isinstance(effective.get('filename'), str):
                try:
                    effective['filename'] = _filename(effective['filename'])
                except ValueError as error:
                    # Keep the ordinary ReAct tool-error feedback path alive.
                    from lagent.schema import ActionReturn, ActionStatusCode
                    return ActionReturn(type=self.name, args=kwargs,
                                        errmsg=repr(error),
                                        state=ActionStatusCode.API_ERROR)
                routing['requested_filename'] = kwargs['filename']
                routing['effective_filename'] = effective['filename']
        # This record contains routing fields only, never credentials or full
        # tool arguments. Prediction ActionReturn.args contains effective args.
        (call_root / 'routing.json').write_text(
            json.dumps(routing, ensure_ascii=False, indent=2), encoding='utf-8')
        token = _CALL_ROOT.set(call_root)
        try:
            result = original_run(**effective)
            _retain_returned_files(result, call_root)
            return result
        finally:
            _CALL_ROOT.reset(token)

    action.run = types.MethodType(run, action)
    action._gta_output_routing = True
    return action
