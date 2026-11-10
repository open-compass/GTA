"""Offline workflow tests with temporary WAV files and synthetic search responses."""
import importlib.util
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agentlego.tools.search.google import GoogleSearch

ROOT = Path(__file__).resolve().parents[2]


def load_source(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WebEvidenceTests(unittest.TestCase):
    def test_cache_is_opt_in_and_only_successes_replay(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'AGENTLEGO_WEB_CACHE_DIR': ''}):
            mock = Mock(side_effect=[
                Mock(status_code=503, json=lambda: {'error': 'unavailable'}),
                Mock(status_code=200, json=lambda: {'organic': [{'title': 'example'}]})])
            with patch('agentlego.tools.search.google.requests.post', mock):
                self.assertIsNone(GoogleSearch(api_key='example').cache_dir)
                search = GoogleSearch(api_key='example', cache_dir=tmp)
                self.assertEqual(search._search('test', k=2)[0], 503)
                self.assertEqual(list(Path(tmp).iterdir()), [])
                self.assertEqual(search._search('test', k=2), search._search('test', k=2))
            self.assertEqual(mock.call_count, 2)
            record, = Path(tmp).glob('*.json')
            saved = json.loads(record.read_text())
            self.assertFalse(saved['archived_after_evaluation'])
            self.assertNotIn('headers', saved)
            self.assertEqual(saved['params']['q'], 'test')

    def test_legacy_and_native_calls_match_only_their_own_replies(self):
        extractor = load_source('extract_web_trajectory', 'analysis/extract_web_trajectory.py')
        def call(name, ident):
            return dict(id=ident, function=dict(name=name, arguments={'query': 'test'}))
        turns = [[dict(role='assistant', tool_calls=[call('GoogleSearch', 'w'), call('Calculator', 'c')]),
                  dict(role='tool', tool_call_id='c', content='sum'),
                  dict(role='tool', tool_call_id='w', content='web page')],
                 [dict(role='assistant', tool_calls=[call('GoogleSearch', None)])],
                 [dict(role='assistant', tool_calls=[call('Calculator', None)]),
                  dict(role='tool', content='unrelated'),
                  dict(role='assistant', tool_calls=[call('WebSearch', None)]),
                  dict(role='tool', content='legacy page')]]
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'predictions/model/dataset.json'
            dest.parent.mkdir(parents=True)
            dest.write_text(json.dumps({'0': {'task_id': 72, 'prediction': turns}}))
            rows = list(extractor.extract(Path(tmp)))
        self.assertEqual([r['search_response'] for r in rows], ['web page', '', 'legacy page'])
        self.assertEqual([r['response_saved'] for r in rows], [True, False, True])
        self.assertEqual([r['task_id'] for r in rows], [72] * 3)

    def test_archives_keep_original_fetch_time_on_resume(self):
        archiver = load_source('archive_web_urls', 'analysis/archive_web_urls.py')
        url = 'https://example.org/report'
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                archiver, 'fetch', return_value=(url, 'text/plain', b'report')) as fetch:
            first = archiver.archive([url, url], Path(tmp), 1, 8)[url]
            second = archiver.archive([url], Path(tmp), 1, 8)[url]
            self.assertEqual(first, second)
            self.assertTrue(first['archived_after_evaluation'])
            self.assertEqual((Path(tmp) / first['path']).read_bytes(), b'report')
            fetch.assert_called_once()

    def test_archiver_rejects_nonpublic_addresses(self):
        archiver = load_source('archive_web_urls', 'analysis/archive_web_urls.py')
        with patch.object(socket, 'getaddrinfo', return_value=[(None, None, None, '', ('127.0.0.1', 80))]), \
             patch.object(socket, 'create_connection') as connect, self.assertRaises(ValueError):
            archiver.fetch('http://localhost/private', 1, 8)
        connect.assert_not_called()


class NoiseReductionTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('noisereduce'), 'requires GTA audio dependencies')
    def test_mono_and_stereo_layout(self):
        import noisereduce
        import numpy as np
        import soundfile as sf

        bench = load_source('gta_noise_benchmark', 'agentlego/benchmark.py')
        rate, frames = 16000, 8192
        t = np.arange(frames) / rate
        left = 0.4 * np.sin(2 * np.pi * 440 * t)
        right = 0.3 * np.sin(2 * np.pi * 800 * t)
        for signal in (left, np.column_stack([left, right])):
            with self.subTest(shape=signal.shape), tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / 'input.wav'
                sf.write(source, signal, rate, subtype='FLOAT')
                reduce = Mock(wraps=noisereduce.reduce_noise)
                with patch.object(bench, 'nr', SimpleNamespace(reduce_noise=reduce)):
                    bench.NoiseReduction().apply(str(source), output_dir=tmp)
                actual = reduce.call_args.kwargs['y']
                np.testing.assert_allclose(actual, signal.T, atol=1e-7)
                denoised, sample_rate = sf.read(Path(tmp) / 'denoised.wav')
                self.assertEqual(denoised.shape, signal.shape)
                self.assertEqual(sample_rate, rate)
                self.assertTrue(np.isfinite(denoised).all())


if __name__ == '__main__':
    unittest.main()
