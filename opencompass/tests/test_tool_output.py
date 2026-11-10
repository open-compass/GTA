"""Artifact routing tests using real AgentLego/Lagent and fake HTTP responses.

Run in the existing opencompass conda environment; no model or live tool-server
requests are made.
"""
import base64
from concurrent.futures import ThreadPoolExecutor
import copy
from io import BytesIO
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import patch


from opencompass.models import tool_output as HELPER


def make_remote(name='WriteFile', output_type='file', output_dir=True, filetype=None):
    """Construct a real RemoteTool with a minimal already-parsed OpenAPI operation."""
    from agentlego.parsers import DefaultParser
    from agentlego.schema import Parameter, ToolMeta
    from agentlego.tools.remote import RemoteTool
    from agentlego.types import File, ImageIO
    from agentlego.utils.openapi import APIResponseProperty
    result_type = File if output_type == 'file' else ImageIO
    inputs = [Parameter(name='filename', type=str, description='Output filename', optional=False)]
    if output_dir:
        inputs.append(Parameter(name='output_dir', type=str, description='Output directory',
                                optional=True, default='generated'))
    tool = RemoteTool.__new__(RemoteTool)
    tool.operation = types.SimpleNamespace(
        path_params=[], query_params=[], body_params=[p.name for p in inputs],
        request_body=types.SimpleNamespace(media_type='application/json'),
        responses={'200': types.SimpleNamespace(properties=APIResponseProperty.__new__(APIResponseProperty))})
    tool.url = 'http://fake.invalid/' + name
    tool.method = 'POST'
    tool.headers = None
    tool.auth = None
    tool.toolkit = None
    tool.toolmeta = ToolMeta(name=name, description='A test remote export tool.',
        inputs=tuple(inputs), outputs=(Parameter(name='result', type=result_type,
        description='Exported artifact', optional=False,
        filetype=filetype or ('text/html' if output_type == 'file' else 'image/png')),))
    tool._is_setup = True
    tool.set_parser(DefaultParser)
    return tool.to_lagent()


class FakeHTTP:
    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, **kwargs):
        body = dict(kwargs.get('json', {}))
        with self.lock:
            ordinal = len(self.calls)
            self.calls.append(body)
        if kwargs['url'].endswith('/MakeImage'):
            from PIL import Image
            stream = BytesIO()
            Image.new('RGB', (2, 2), color=(70, 130, 180)).save(stream, 'PNG')
            payload = stream.getvalue()
        elif kwargs['url'].endswith('/MakeVideo'):
            # The File transport preserves bytes; it does not decode a video.
            payload = b'\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom'
        else:
            payload = ('<html><body>artifact-%s</body></html>' % ordinal).encode()
        if body.get('output_dir'):
            target = Path(body['output_dir']) / body['filename']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        response = types.SimpleNamespace(status_code=200, reason='OK', headers={'Content-Type': 'application/json'})
        response.json = lambda: base64.b64encode(payload).decode('ascii')
        return response


def result_path(result):
    if getattr(result, 'errmsg', None):
        raise AssertionError(result.errmsg)
    return Path(result.result[0]['content']).resolve()


def process_job(output_dir, queue):
    fake = FakeHTTP()
    action = HELPER.bind_tool_output(make_remote())
    with patch('agentlego.tools.remote.requests.request', side_effect=fake):
        with HELPER.task_output_scope(output_dir, 'same-task') as root:
            result = action.run(filename='same.html', output_dir='ignored')
            queue.put((str(root), str(result_path(result)), str(fake.calls[-1]['output_dir'])))


class ArtifactRoutingTests(unittest.TestCase):
    def setUp(self):
        # All test writes, including legacy fallback writes, stay in one fresh temp tree.
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name).resolve()
        self.fake = FakeHTTP()
        self.http = patch('agentlego.tools.remote.requests.request', side_effect=self.fake)
        self.http.start()
        self.old_temp = patch.dict(os.environ, {'AGENTLEGO_TMPDIR': str(self.base / 'legacy')})
        self.old_temp.start()

    def tearDown(self):
        self.http.stop()
        self.old_temp.stop()
        self.temp.cleanup()

    def invoke(self, model='model-a', task='7', filename='same.html'):
        action = HELPER.bind_tool_output(make_remote())
        with HELPER.task_output_scope(self.base / 'tools_output' / model, task) as root:
            result = action.run(filename=filename, output_dir='should-be-replaced')
            return root, result, result_path(result)

    def test_real_remote_file_response_and_effective_args(self):
        root, result, path = self.invoke()
        self.assertTrue(path.is_relative_to(root))
        self.assertEqual(path.read_bytes(), (root / 'files/same.html').read_bytes())
        self.assertEqual(result.args['output_dir'], str(root / 'files'))
        self.assertEqual(self.fake.calls[-1]['output_dir'], result.args['output_dir'])
        self.assertEqual(result.args['filename'], self.fake.calls[-1]['filename'])
        self.assertEqual(path.suffix, '.html')

    def test_real_remote_image_response(self):
        action = HELPER.bind_tool_output(make_remote(name='MakeImage', output_type='image', output_dir=False))
        with HELPER.task_output_scope(self.base / 'tools_output/model', '8') as root:
            result = action.run(filename='image.png')
            path = result_path(result)
            self.assertTrue(path.is_relative_to(root))
            self.assertEqual(path.suffix, '.png')
            from PIL import Image
            with Image.open(path) as image:
                self.assertEqual(image.size, (2, 2))

    def test_real_remote_video_file_response(self):
        action = HELPER.bind_tool_output(make_remote(name='MakeVideo', filetype='video/mp4'))
        with HELPER.task_output_scope(self.base / 'tools_output/model', 'video-task') as root:
            result = action.run(filename='../clip.mp4', output_dir='ignored')
            path = result_path(result)
            self.assertTrue(path.is_relative_to(root))
            self.assertEqual(path.suffix, '.mp4')
            self.assertEqual(path.read_bytes(), (root / 'files/clip.mp4').read_bytes())
            self.assertIn(b'ftypmp42', path.read_bytes())
            self.assertEqual(result.args['filename'], 'clip.mp4')
            self.assertEqual(self.fake.calls[-1]['output_dir'], str(root / 'files'))

    def test_models_and_tasks_with_same_filename_do_not_collide(self):
        paths, roots = [], []
        for model, task in [('model-a', '7'), ('model-a', '8'), ('model-b', '7')]:
            root, _, path = self.invoke(model, task)
            roots.append(root)
            paths.append(path)
            self.assertTrue(root.is_relative_to(self.base / 'tools_output' / model))
            self.assertTrue((root / 'files/same.html').is_file())
        self.assertEqual(len(set(roots)), 3)
        self.assertEqual(len(set(paths)), 3)
        self.assertEqual(len({p.read_bytes() for p in paths}), 3)

    def test_same_task_new_attempt_preserves_old_files(self):
        root1, _, path1 = self.invoke()
        before = path1.read_bytes()
        root2, _, path2 = self.invoke()
        self.assertNotEqual(root1, root2)
        self.assertEqual(root1.parent, root2.parent)
        self.assertEqual(path1.read_bytes(), before)
        self.assertNotEqual(path1, path2)
        self.assertEqual((root1 / 'files/same.html').read_bytes(), before)

    def test_absolute_and_traversal_filenames_use_basename(self):
        for requested in ['/tmp/escape.html', '../../escape.html', r'C:\outside\escape.html', 'sub/escape.html']:
            with self.subTest(requested=requested):
                root, result, path = self.invoke(filename=requested)
                self.assertEqual(result.args['filename'], 'escape.html')
                self.assertEqual(self.fake.calls[-1]['filename'], 'escape.html')
                self.assertTrue((root / 'files/escape.html').is_file())
                self.assertTrue(path.is_relative_to(root))

    def test_invalid_filename_stops_before_http(self):
        from lagent.schema import ActionStatusCode
        action = HELPER.bind_tool_output(make_remote())
        with HELPER.task_output_scope(self.base / 'tools_output/model', 9):
            for filename in ['', '..', '/tmp/', 'bad\x00.html']:
                before = len(self.fake.calls)
                with self.subTest(filename=filename):
                    result = action.run(filename=filename, output_dir='ignored')
                    self.assertEqual(result.state, ActionStatusCode.API_ERROR)
                    self.assertTrue(result.errmsg)
                    self.assertEqual(result.args['filename'], filename)
                self.assertEqual(len(self.fake.calls), before)

    def test_real_lagent_dispatcher_uses_bound_run(self):
        action = HELPER.bind_tool_output(make_remote())
        with HELPER.task_output_scope(self.base / 'tools_output/model', 'dispatcher') as root:
            result = action(json.dumps({'filename': '../dispatch.html', 'output_dir': 'ignored'}))
            path = result_path(result)
            self.assertTrue(path.is_relative_to(root))
            self.assertEqual(self.fake.calls[-1]['filename'], 'dispatch.html')
            self.assertEqual(result.args['filename'], 'dispatch.html')
            self.assertEqual(result.args['output_dir'], str(root / 'files'))
            self.assertTrue((root / 'files/dispatch.html').is_file())

    def test_no_scope_keeps_legacy_directory_and_arguments(self):
        action = HELPER.bind_tool_output(make_remote())
        explicit = str(self.base / 'requested')
        result = action.run(filename='legacy.html', output_dir=explicit)
        self.assertEqual(result.args, {'filename': 'legacy.html', 'output_dir': explicit})
        self.assertEqual(self.fake.calls[-1], result.args)
        self.assertTrue(result_path(result).is_relative_to(self.base / 'legacy'))
        self.assertTrue((Path(explicit) / 'legacy.html').exists())

    def test_exception_restores_task_and_call_context(self):
        action = HELPER.bind_tool_output(make_remote())
        with self.assertRaises(RuntimeError):
            with HELPER.task_output_scope(self.base / 'tools_output/model', 10):
                action.run(filename='one.html', output_dir='ignored')
                raise RuntimeError('synthetic sample failure')
        self.assertIsNone(HELPER._TASK_ROOT.get())
        self.assertIsNone(HELPER._CALL_ROOT.get())
        result = action.run(filename='legacy.html', output_dir=str(self.base / 'after-error'))
        self.assertTrue(result_path(result).is_relative_to(self.base / 'legacy'))

    def test_remote_http_exception_does_not_leak_call_scope(self):
        import requests
        action = HELPER.bind_tool_output(make_remote())
        with HELPER.task_output_scope(self.base / 'tools_output/model', 11):
            with patch('agentlego.tools.remote.requests.request', side_effect=requests.ConnectionError('test only')):
                result = action.run(filename='bad.html', output_dir='ignored')
            self.assertTrue(result.errmsg)
            self.assertIsNone(HELPER._CALL_ROOT.get())
            result = action.run(filename='good.html', output_dir='ignored')
            self.assertTrue(result_path(result).is_file())

    def test_threads_keep_their_own_scope(self):
        barrier = threading.Barrier(4)
        def job(number):
            action = HELPER.bind_tool_output(make_remote())
            with HELPER.task_output_scope(self.base / f'tools_output/model-{number % 2}', number) as root:
                barrier.wait(timeout=20)
                path = result_path(action.run(filename='same.html', output_dir='ignored'))
                self.assertTrue(path.is_relative_to(root))
                return root, path
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(job, range(4)))
        self.assertEqual(len({r for r, _ in results}), 4)
        self.assertEqual(len({p for _, p in results}), 4)
        self.assertIsNone(HELPER._TASK_ROOT.get())

    def test_spawned_processes_allocate_distinct_attempts(self):
        context = multiprocessing.get_context('spawn')
        queue = context.Queue()
        procs = [context.Process(target=process_job,
                 args=(str(self.base / 'tools_output/model'), queue)) for _ in range(2)]
        for proc in procs:
            proc.start()
        try:
            values = [queue.get(timeout=60) for _ in procs]
            for proc in procs:
                proc.join(timeout=20)
                self.assertEqual(proc.exitcode, 0)
            self.assertEqual(len({v[0] for v in values}), 2)
            self.assertEqual(len({v[1] for v in values}), 2)
            for root, artifact, server_dir in values:
                self.assertTrue(Path(artifact).is_relative_to(Path(root)))
                self.assertTrue(Path(server_dir).is_relative_to(Path(root)))
        finally:
            for proc in procs:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=10)
            queue.close()

    def test_protocol_and_prompt_construction_unchanged(self):
        from lagent.actions import ActionExecutor
        from lagent.agents.react import ReActProtocol
        action = make_remote()
        executor = ActionExecutor([action])
        protocol = ReActProtocol()
        config = copy.deepcopy(protocol.__dict__)
        history = [{'role': 'user', 'content': 'Create the requested document.'}]
        before = protocol.format(chat_history=[], inner_step=history, action_executor=executor)
        description = copy.deepcopy(action.description)
        HELPER.bind_tool_output(action)
        with HELPER.task_output_scope(self.base / 'tools_output/model', 12):
            action.run(filename='doc.html', output_dir='ignored')
            after = protocol.format(chat_history=[], inner_step=history, action_executor=executor)
        self.assertEqual(before, after)
        self.assertEqual(protocol.__dict__, config)
        self.assertEqual(action.description, description)

    def test_inferencer_decorator_keeps_chat_and_return(self):
        original_chat = {'id': 'real-id', 'dialogs': [{'role': 'user', 'content': 'unchanged'}]}
        chat = copy.deepcopy(original_chat)
        handler = types.SimpleNamespace(results_dict={})
        class Inferencer:
            _tools_output_dir = str(self.base / 'tools_output/model')
            @HELPER.scoped_tool_outputs
            def sample(inner, incoming, index, output_handler):
                self.assertIs(incoming, chat)
                output_handler.results_dict[str(index)] = {'prediction': 'same'}
                return 'returned unchanged'
        result = Inferencer().sample(chat, 3, handler)
        self.assertEqual(result, 'returned unchanged')
        self.assertEqual(chat, original_chat)
        self.assertEqual(handler.results_dict['3']['prediction'], 'same')
        self.assertIn('task_real-id', handler.results_dict['3']['tools_output_dir'])

    def test_identifiers_are_encoded_without_collisions(self):
        self.assertNotEqual(HELPER.path_component('a/b'), HELPER.path_component('a%2Fb'))
        self.assertNotIn('/', HELPER.path_component('../id'))

    def test_binding_is_idempotent(self):
        action = make_remote()
        HELPER.bind_tool_output(action)
        bound = action.run
        HELPER.bind_tool_output(action)
        self.assertIs(action.run, bound)
        with HELPER.task_output_scope(self.base / 'tools_output/model', 13) as root:
            action.run(filename='one.html', output_dir='ignored')
        self.assertEqual(len(list((root / 'calls').iterdir())), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
