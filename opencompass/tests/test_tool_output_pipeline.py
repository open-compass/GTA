"""Actual inference pipeline with scripted text and fake RemoteTool HTTP only.

Run after deploying the output-routing modules, from the OpenCompass root.
No model API, judge API, or live tool server is contacted.
"""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lagent.agents import ReAct
from lagent.agents.react import ReActProtocol
from opencompass.models.lagent import LagentAgent
from opencompass.models import tool_output
from opencompass.openicl.icl_inferencer.icl_agent_inferencer import (
    AgentInferencer, AgentInferencerOutputHandler)

from test_tool_output import FakeHTTP, make_remote


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, inputs, *args, **kwargs):
        self.calls.append(copy.deepcopy(inputs))
        if len(inputs) != 1 or not self.responses:
            raise AssertionError('Unexpected model request in scripted pipeline')
        return [self.responses.pop(0)]


class ToolOutputPipelineTests(unittest.TestCase):
    def test_real_inferencer_react_tool_then_final(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            output = base / 'run/tools_output/model-a'
            remote = make_remote().tool
            fake = FakeHTTP()
            protocol = ReActProtocol()
            first = (protocol.thought['begin'] + 'Create the requested report.\n'
                     + protocol.action['begin'] + 'WriteFile\n'
                     + protocol.action_input['begin']
                     + json.dumps({'filename': '../report.html', 'output_dir': 'ignored'}))
            final = protocol.thought['begin'] + 'The server returned the report.\n' + protocol.finish['begin'] + 'Completed.'
            with patch('agentlego.tools.remote.RemoteTool.from_server', return_value=[remote]), \
                 patch('agentlego.tools.remote.requests.request', side_effect=fake), \
                 patch.dict(os.environ, {'AGENTLEGO_TMPDIR': str(base / 'legacy')}):
                model = LagentAgent(agent_type=ReAct,
                    llm=dict(type=ScriptedLLM, responses=[first, final]),
                    tool_server='http://fake.invalid',
                    protocol=dict(type=ReActProtocol), max_turn=10)
                inferencer = AgentInferencer.__new__(AgentInferencer)
                inferencer.model = model
                inferencer.configure_tools_output(str(output))
                handler = AgentInferencerOutputHandler()
                chat = {'id': 'original-task-7',
                    'dialogs': [{'role': 'user', 'content': 'Create an HTML report.'}],
                    'resources': [{'type': 'tool', 'name': 'WriteFile'}]}
                before = copy.deepcopy(chat)
                protocol_before = copy.deepcopy(vars(model.agent._protocol))
                expected_messages = model.agent._protocol.format(
                    chat_history=[], inner_step=[{'role': 'user', 'content': chat['dialogs'][0]['content']}],
                    action_executor=model.agent._action_executor)
                inferencer.infer_every(chat, 0, handler)

            record = handler.results_dict['0']
            steps = record['prediction'][0]
            self.assertNotIn('AgentError', json.dumps(steps))
            self.assertEqual(chat, before)
            self.assertEqual(vars(model.agent._protocol), protocol_before)
            max_turn = getattr(model.agent, '_max_turn', getattr(model.agent, 'max_turn', None))
            self.assertEqual(max_turn, 10)
            self.assertEqual(len(model.agent._llm.calls), 2)
            self.assertEqual(len(fake.calls), 1)
            self.assertEqual(record['origin_prompt'], ['Create an HTML report.'])
            # model_adapter must only map roles/content; routing adds no prompt text.
            role_map = {'user': 'HUMAN', 'assistant': 'BOT', 'system': 'SYSTEM'}
            expected_prompt = [{'prompt': msg['content'], 'role': role_map[msg['role']]}
                               for msg in expected_messages]
            self.assertEqual(model.agent._llm.calls[0][0], expected_prompt)
            root = Path(record['tools_output_dir']).resolve()
            self.assertTrue(root.is_relative_to(output))
            self.assertEqual(root.parent.name, 'task_original-task-7')
            attachments = [Path(path).resolve() for path in record['attached_files']]
            self.assertEqual(len(attachments), 1)
            for path in attachments:
                self.assertTrue(path.is_file())
                self.assertTrue(path.is_relative_to(root))
                self.assertEqual(path.read_bytes(), (root / 'files/report.html').read_bytes())
                self.assertIn(str(path), json.dumps(model.agent._llm.calls[1]))
            calls = [step['tool_calls'][0]['function'] for step in steps if step.get('tool_calls')]
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]['name'], 'WriteFile')
            self.assertEqual(calls[0]['arguments'], fake.calls[0])
            self.assertEqual(calls[0]['arguments']['filename'], 'report.html')
            self.assertEqual(calls[0]['arguments']['output_dir'], str(root / 'files'))
            self.assertIn('Completed.', steps[-1]['content'])
            self.assertIsNone(tool_output._TASK_ROOT.get())
            self.assertIsNone(tool_output._CALL_ROOT.get())


if __name__ == '__main__':
    unittest.main(verbosity=2)
