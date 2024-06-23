from copy import deepcopy
import json
from typing import List, Tuple

try:
    import qwen_agent
except ImportError:
    qwen_agent = None

class DummyTool(qwen_agent.tools.BaseTool):
    def __init__(self, cfg):
        parameters = []
        type_mapping = {
            'int': 'int',
            'float': 'float',
            'bool': 'bool'
        }
        for p in cfg['inputs']:
            parameters.append(
                dict(
                    name=p['name'],
                    description=p['description'],
                    type=type_mapping.get(p['type'], 'string'),
                    required=not p.get('optional', False)
                ))
        self.name = cfg['type']
        self.description = cfg['description']
        self.parameters = parameters
        super().__init__(cfg=cfg)

    def call(self, *args, **kwargs):
        return ''


def dummy_tools(tools):
    function_map = {}
    for cfg in tools:
        tool = DummyTool(cfg)
        function_map[tool.name] = tool
    return function_map

def qwen_style_history(history, files=None) -> List[dict]:
    from qwen_agent.llm.schema import USER, ASSISTANT, FUNCTION
    inner_steps = []
    if files:
        prompt = '\nAll files you need are at ' + ', '.join(f'`{file["path"]}`' for file in files)
    else:
        prompt = None

    for step in history:
        if step['role'] == 'user':
            content = step['content']
            if prompt:
                content += prompt
                prompt = None
            inner_steps.append(dict(role=USER, content=content))
        elif step['role'] == 'assistant' and step.get('tool_calls'):
            name = step['tool_calls'][0]['function']['name']
            args = step['tool_calls'][0]['function']['arguments']
            function_call = dict(name=name, arguments=json.dumps(args))
            inner_steps.append(dict(role=ASSISTANT, content='', function_call=function_call))
        elif step['role'] == 'tool' and step.get('content'):
            inner_steps.append(dict(role=FUNCTION, name=step['name'], content=step['content']))
        elif step['role'] == 'assistant' and step.get('content'):
            inner_steps.append(dict(role=ASSISTANT, content=step['content']))
    return inner_steps

class QwenAgent:
    """Agent wrapper for Lagent.

    https://github.com/InternLM/lagent.
    """
    is_api = True

    def __init__(self, llm, function_list=None, **kwargs):
        self.llm_cfg = llm
        self.function_list = function_list
        self.agent_kwargs = kwargs

        # Not use
        self.template_parser = None

    def reset(self):
        pass

    def next_step(self, history, tools=None, files=None):
        from qwen_agent.agents import Assistant

        function_map = dummy_tools(tools or [])
        agent = Assistant(llm=self.llm_cfg, **self.agent_kwargs)
        agent.function_map = function_map

        history = qwen_style_history(history, files)

        for response in agent.run(messages=history):
            if response and response[-1]['role'] != 'assistant':
                response = response[:-1]
                break
        response = response[-1]

        if response['content']:
            return dict(role='assistant', content=response['content'])
        else:
            name = response['function_call']['name']
            try:
                tool = function_map[name]
                args = tool._verify_json_format_args(response['function_call']['arguments'])
            except Exception:
                args = response['function_call']['arguments']
            function = dict(name=name, arguments=args)
            msg = dict(role='assistant',
                       tool_calls=[dict(type='function', function=function)])
            return msg

    def chat(self, history, tools=None, files=None):
        __import__('ipdb').set_trace()
        action_executor = self.agent._action_executor
        self.agent._action_executor.actions = [
            self.tools[cfg['type']] for cfg in tools
        ]

        history = react_style_history(history, files, self.agent._protocol)
        agent_return = self.agent.chat(history)
        steps = []

        for action in agent_return.actions:
            if action.type == action_executor.finish_action.name:
                step = dict(role='assistant', content=action.format_result())
            else:
                function = dict(name=action.type,
                                arguments=action.args)
                step = dict(role='assistant',
                           tool_calls=[dict(type='function', function=function)])
                if action.errmsg is not None:
                    step['error'] = dict(type=action.state.name, msg=action.errmsg)
            steps.append(step)
        return steps
