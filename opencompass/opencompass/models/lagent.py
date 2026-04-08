from copy import deepcopy
import types
import json
from typing import List, Tuple

from mmengine.registry import Registry

REGISTRY = Registry('helper')

try:
    import lagent
    import agentlego
except ImportError:
    lagent = None
    agentlego = None


def _patch_lagent_action_executor_split():
    """Patch lagent ActionExecutor parsing to be resilient to tool names containing dots.

    Upstream lagent parses toolkit calls like "<toolkit>.<api>" via `name.split('.')`,
    which crashes when `<toolkit>` itself contains dots (e.g. "Solver.v2.solve").

    Strategy:
      1) If `name` matches an action name exactly, treat it as a normal action (api='run').
      2) Otherwise, if it contains dots, split from the right once: `rsplit('.', 1)`.

    This keeps backward compatibility while preventing `ValueError: too many values to unpack`.
    """
    if lagent is None:
        return

    try:
        from lagent.actions.action_executor import ActionExecutor as _ActionExecutor
        from lagent.schema import ActionValidCode
    except Exception:
        return

    if getattr(_ActionExecutor, '_opencompass_safe_split_patched', False):
        return

    def _safe_call(self, name: str, command: str):
        if name in getattr(self, 'actions', {}):
            action_name, api_name = name, 'run'
        elif '.' in name:
            action_name, api_name = name.rsplit('.', 1)
        else:
            action_name, api_name = name, 'run'

        if not self.is_valid(action_name):
            if name == self.no_action.name:
                action_return = self.no_action(command)
            elif name == self.finish_action.name:
                action_return = self.finish_action(command)
            else:
                action_return = self.invalid_action(command)
        else:
            action_return = self.actions[action_name](command, api_name)
            action_return.valid = ActionValidCode.OPEN
        return action_return

    _ActionExecutor.__call__ = _safe_call
    setattr(_ActionExecutor, '_opencompass_safe_split_patched', True)

class DummyTool(agentlego.tools.BaseTool):

    def __init__(self, toolmeta):
        self.toolmeta = agentlego.schema.ToolMeta.from_json_dict(toolmeta)
        self.set_parser(agentlego.parsers.DefaultParser)
        self._is_setup = False

    def apply(self, *args, **kwargs):
        return 'Dummy Result'

def dummy_action_executor(tools):
    return lagent.ActionExecutor(
        actions=[DummyTool(cfg).to_lagent() for cfg in tools])


def model_adapter(model):
    """Modify the generate method to accept and return single item."""
    if getattr(model, '_generate_is_wrapped', False):
        # Avoid wrap twice.
        return model

    from opencompass.utils import PromptList

    def chat(self, inputs, *args, **kwargs):
        prompt = PromptList()
        for item in inputs:
            msg = {'prompt': item['content']}
            if item['role'] == 'user':
                msg['role'] = 'HUMAN'
            elif item['role'] == 'assistant':
                msg['role'] = 'BOT'
            elif item['role'] == 'system':
                msg['role'] = 'SYSTEM'
            prompt.append(msg)
        return self.generate([prompt], *args, **kwargs)[0]

    model.chat = types.MethodType(chat, model)
    setattr(model, '_generate_is_wrapped', True)
    return model


def react_style_history(history, files, protocol) -> List[dict]:
    from lagent.schema import ActionReturn
    inner_steps = []
    if files:
        prompt = 'The related files are at ' + ', '.join(f'`{file["path"]}`'
                                                         for file in files)
        inner_steps.append(dict(role='system', content=prompt))
    for step in history:
        if step['role'] == 'user':
            inner_steps.append(dict(role='user', content=step['content']))
        elif step['role'] == 'assistant' and step.get('tool_calls'):
            name = step['tool_calls'][0]['function']['name']
            args = step['tool_calls'][0]['function']['arguments']
            response = "{action}{name}\n{action_input}{args}".format(
                action=protocol.action['begin'],
                name=name,
                action_input=protocol.action_input['begin'],
                args=json.dumps(args),
            )
            inner_steps.append(dict(role='assistant', content=response))
        elif step['role'] == 'tool' and step.get('content'):
            # action= ActionReturn(result=[dict(type='text', content=step['content'])])
            action= ActionReturn(result=[step['content']])
            inner_steps.append(dict(role='system', content=action.format_result()))
        elif step['role'] == 'assistant' and step.get('content'):
            inner_steps.append(dict(role='assistant', content=step['content']))
    return inner_steps


class LagentAgent:
    """Agent wrapper for Lagent.

    https://github.com/InternLM/lagent.
    """
    is_api = True

    def __init__(self,
                 agent_type,
                 llm,
                 actions=None,
                 tool_server=None,
                 tool_meta=None,
                 protocol=None,
                 **kwargs):
        _patch_lagent_action_executor_split()
        llm = model_adapter(REGISTRY.build(llm))
        agent_cfg = {'type': agent_type, 'llm': llm, **kwargs}

        tools = {}
        if actions is not None:
            for action in actions:
                action = REGISTRY.build(action)
                if isinstance(action, agentlego.tools.BaseTool):
                    action = action.to_lagent()
                tools[action.name] = action
        if tool_server is not None:
            from agentlego.tools.remote import RemoteTool
            for tool in RemoteTool.from_server(tool_server):
                tools[tool.name] = tool.to_lagent()
        if tool_meta is not None:
            metas = json.load(open(tool_meta, 'r'))
            for meta in metas.values():
                tool = DummyTool(meta).to_lagent()
                tools.setdefault(tool.name, tool)

        self.tools = tools

        if protocol is not None:
            protocol = REGISTRY.build(protocol)
            agent_cfg['protocol'] = protocol

        from lagent import BaseAgent, ActionExecutor
        agent_cfg['action_executor'] = ActionExecutor(tools.values())
        self.agent: BaseAgent = REGISTRY.build(agent_cfg)

    def reset(self):
        pass

    def gt_response(self, prompt):
        if 'CIReAct' in str(self.agent.__class__):
            gold = prompt
            prompt = f"""{self.agent._protocol.action['begin']} IPythonInterpreter
{self.agent._protocol.action_input['begin']} ```python\n{gold}\n```\n"""  # noqa
            action_input = dict(
                command=f"""```python\n{gold}\n```\n""",
                timeout=120,
            )
            response = self.agent._action_executor('IPythonInterpreter',
                                                   action_input)
            gt_response = dict(role='assistant', content=prompt)
            system_response = dict(
                role='system',
                content=self.agent._protocol.format_response(response))
            return [gt_response, system_response]
        else:
            gt_response = dict(role='assistant', content=prompt)
            return [gt_response]

    @property
    def template_parser(self):
        return self.agent._llm.template_parser

    @template_parser.setter
    def template_parser(self, value):
        self.agent._llm.template_parser = value

    def next_step(self, history, resources=None, stop=False):
        from lagent.schema import ActionReturn
        tools = []
        files = []
        if resources is not None:
            tools = [
                self.tools[item['name']] for item in resources
                if item['type'] == 'tool'
            ]
            files = [item for item in resources if item['type'] == 'file']

        action_executor = lagent.ActionExecutor(actions=tools)
        if stop:
            history = history + [{'role': 'user', 'content': 'Please summarize the chat history and give a final answer. Do not call any tools.'}]
        history = react_style_history(history, files, self.agent._protocol)
        prompt = self.agent._protocol.format(chat_history=[],
                                             inner_step=history,
                                             action_executor=action_executor)
        response = self.agent._llm.chat(prompt)
        thought, action_name, action_input = self.agent._protocol.parse(
            response, action_executor)
        action: ActionReturn = action_executor(action_name, action_input)

        if action.type == action_executor.finish_action.name:
            return dict(role='assistant', content=action.format_result())
        else:
            msg = {'role': 'assistant'}
            args = action.args
            if action.errmsg is not None:
                msg['error'] = dict(type=action.state.name, msg=action.errmsg)
                # Handle fallback args
                args = args.get('inputs', args)
            function = dict(name=action.type, arguments=args)
            msg['tool_calls'] = [dict(type='function', function=function)]
            return msg

    def chat(self, query, memory=None, resources=None):
        tools = []
        files = []
        if resources is not None:
            tools = {
                item['name']: self.tools[item['name']]
                for item in resources if item['type'] == 'tool'
            }
            files = [item for item in resources if item['type'] == 'file']

        action_executor = self.agent._action_executor
        action_executor.actions = tools
        if memory is None:
            memory = []
            if files:
                prompt = 'The related files are at ' + ', '.join(
                    f'`{file["path"]}`' for file in files)
                memory.append(dict(role='user', content=prompt))
        memory.append(dict(role='user', content=query))

        try:
            agent_return = self.agent.chat(memory)
        except Exception as e:
            # Keep evaluation running even if tool execution crashes.
            steps = [
                {
                    'role': 'assistant',
                    'content': f'[LagentAgentError] {type(e).__name__}: {e}',
                    'error': {
                        'type': type(e).__name__,
                        'msg': str(e),
                    },
                    'meta': {'type': 'error'},
                }
            ]
            return steps, memory

        steps = []
        for action in agent_return.actions:
            # Insert model thought in order before each action/tool call
            thought = getattr(action, 'thought', None)
            if isinstance(thought, str) and thought.strip():
                steps.append({
                    'role': 'assistant',
                    'content': thought,
                    'meta': {'type': 'thought'}
                })
            if action.type == action_executor.finish_action.name:
                step = dict(role='assistant', content=action.format_result())
                steps.append(step)
            else:
                step = {'role': 'assistant'}
                args = action.args
                if action.errmsg is not None:
                    step['error'] = dict(type=action.state.name,
                                         msg=action.errmsg)
                    # Handle fallback args
                    args = args.get('inputs', args)
                function = dict(name=action.type, arguments=args)
                step['tool_calls'] = [dict(type='function', function=function)]
                steps.append(step)
                steps.append(dict(role='tool', content=action.result))
        return steps, memory



FORCE_STOP_PROMPT_EN = (
    """You should directly give results based on history information."""  # noqa
)

FEWSHOT_INSTRUCTION = """\
You are a assistant who can utilize external tools.
{tool_description}
To use a tool, please response with the following format:
```
{thought} Think what you need to solve, do you need to use tools?
{action} The tool name, should be one of [{action_names}].
{action_input} The input to the tool that you want to use.
```
The tool will give you response after your response using the following format:
```
{response} the results after call the tool.
```
Therefore DO NOT generate tool response by yourself.

Also please follow the guidelines:
1. Always use code interpreter to solve the problem.
2. The generated codes should always in a markdown code block format.
3. The generated codes will be executed in an ipython manner and the results will be cached.
4. Your responded code should always be simple and only solves the problem in current step.

Begin!
"""  # noqa

PYTHON_INTERPRETER_DESCRIPTION = """\
It can run a Python code. The code must be a valid code that contains only python method, and the method' name must be 'solution' and returns a dict, which key is variable name. The libraries I recommend are sympy and scipy. the format is:
```python
# import packages
import xxx
def solution():
    # initialize some variables
    variable_names_with_real_meaning = xxx
    # middle steps
    mid_variable = func(mid_variable)
    # final answer
    final_answer = func(mid_variable)
    return final_answer
```"""  # noqa


class CodeAgent(LagentAgent):
    """Code Agent wrapper for Lagent."""

    def __init__(self, llm, **kwargs):
        from lagent import PythonInterpreter, ReAct
        from lagent.agents.react import ReActProtocol

        agent_type = kwargs.pop('agent_type', ReAct)
        max_turn = kwargs.pop('max_turn', 3)
        actions = kwargs.pop(
            'actions',
            [
                dict(type=PythonInterpreter,
                     description=PYTHON_INTERPRETER_DESCRIPTION),
            ],
        )
        protocol = kwargs.pop(
            'protocol',
            dict(
                type=ReActProtocol,
                call_protocol=FEWSHOT_INSTRUCTION,
                force_stop=FORCE_STOP_PROMPT_EN,
                finish=dict(role='FINISH', begin='Final Answer:', end='\n'),
            ),
        )
        super().__init__(agent_type=agent_type,
                         llm=llm,
                         actions=actions,
                         protocol=protocol,
                         max_turn=max_turn,
                         **kwargs)
