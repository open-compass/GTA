"""
使用LangChain/LangGraph的GTA评测配置文件
替代原来的AgentLego配置，提供相同的评测能力
"""

# 注意：这个配置文件展示了如何从AgentLego迁移到LangChain/LangGraph
# 主要变化：
# 1. 使用LangChainAgent替代LagentAgent
# 2. 工具加载方式从agentlego改为langchain tools
# 3. 保持评测数据集和流程不变

from mmengine.config import read_base
from lagent.agents import ReAct
from lagent.agents.react import ReActProtocol

# 导入模型 - 添加LangChain支持
from opencompass.models import OpenAI, Qwen, Gemini, DeepseekAPI, ManusAPI
from opencompass.models.lagent import LagentAgent  # 原来的AgentLego agent
# from opencompass.models.langchain_agent import LangChainAgent  # 新的LangChain agent

from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

# 导入数据集配置（保持不变）
with read_base():
    from .datasets.gta_bench_ai import gta_bench_datasets as datasets


# system_prompt = """You are a tool-using reasoning agent. Follow EXACTLY this protocol each turn.\n\nThought: concise reasoning (no code fences)\nTool: one of {tool_names} OR FINISH\nTool Input: arguments for the chosen tool\nFinalAnswer: ONLY when Tool=FINISH, the final user answer\n\nWhen you invoke the Python/code interpreter tool you MUST output ONLY a single JSON object: {{\"code\": \"<python code>\"}}\nRules for code JSON: no markdown fences, no commentary before/after, no extra keys, pure valid JSON.\nIf previous attempt failed with format errors, correct strictly now.\n\nAvailable tools:\n{tool_description}\n"""

# protocol = dict(
#     type=ReActProtocol,
#     action=dict(role='ACTION', begin='Tool:', end='\n'),
#     action_input=dict(role='ARGS', begin='Tool Input:', end='\n'),
#     finish=dict(role='FINISH', begin='FinalAnswer:', end='\n'),
#     call_protocol=system_prompt,
# )

models = [
    # dict(
    #         abbr='chatgpt-4o-latest',
    #         type=LagentAgent,
    #         agent_type=ReAct,
    #         max_turn=10,
    #         llm=dict(
    #             type=OpenAI,
    #             path='chatgpt-4o-latest',
    #             key='EMPTY',
    #             openai_api_base='https://ai.nengyongai.cn/v1/chat/completions',
    #             query_per_second=1,
    #             max_seq_len=131072,
    #             stop='<|im_end|>',
    #         ), 
    #         tool_server='http://127.0.1.1:16181',
    #         tool_meta='data/gta_dataset_v2/toolmeta.json',
    #         batch_size=8,
    #     ),
    # dict(   // qwen with OpenAI api  X
    #         abbr='qwen3-max',
    #         type=LagentAgent,
    #         agent_type=ReAct,
    #         max_turn=10,
    #         llm=dict(
    #             type=OpenAI,
    #             path='qwen3-max',
    #             key='EMPTY',
    #             openai_api_base='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
    #             query_per_second=1,
    #             max_seq_len=131072,
    #             stop='<|im_end|>',
    #         ), 
    #         tool_server='http://127.0.1.1:16181',
    #         tool_meta='data/gta_dataset_v2/toolmeta.json',
    #         batch_size=8,
    #     ),
    # dict(
    #     abbr='qwen1.5-7b-chat',
    #     type=LagentAgent,
    #     agent_type=ReAct,
    #     max_turn=10,
    #     llm=dict(
    #         type=Qwen,
    #         path='qwen1.5-7b-chat',
    #         key='EMPTY', # put your key here
    #         # Native Qwen API wrapper uses dashscope Generation.call directly;
    #         # remove OpenAI-compatible arguments (openai_api_base, template, stop, temperature top-level).
    #         query_per_second=1,
    #         max_seq_len=131072,
    #         retry=10,
    #         generation_kwargs={
    #             'temperature': 0.0,
    #             # Optional examples you can enable if needed:
    #             # 'top_p': 1.0,
    #             # 'max_tokens': 512,
    #             # 'stop': ['FinalAnswer:']  # Only if you want to early stop; verify dashscope supports list.
    #         }
    #     ),
    #     protocol=protocol,
    #     tool_server='http://127.0.1.1:16181',
    #     tool_meta='data/gta_dataset_v2/toolmeta.json',
    #     batch_size=8,
    # ),
    # dict(
    #     abbr='deepseek-v3.2',
    #     type=LagentAgent,
    #     agent_type=ReAct,
    #     max_turn=10,
    #     llm=dict(
    #         type=DeepseekAPI,
    #         path='deepseek-chat',
    #         key='EMPTY',
    #         url='https://api.deepseek.com/v1/chat/completions',
    #         query_per_second=1,
    #         max_seq_len=131072,
    #         retry=10,
    #         system_prompt=system_prompt,
    #     ),
    #     protocol=protocol,
    #     tool_server='http://127.0.1.1:16181',
    #     tool_meta='data/gta_dataset_v2/toolmeta.json',
    #     batch_size=8,
    # ),
    # dict(
    #         abbr='qwen3-235b-a22b',
    #         type=LagentAgent,
    #         agent_type=ReAct,
    #         max_turn=10,
    #         llm=dict(
    #             type=OpenAI,
    #             path='qwen/qwen3-235b-a22b-07-25',
    #             key='EMPTY',
    #             openai_api_base='https://ai.nengyongai.cn/v1/chat/completions',
    #             query_per_second=1,
    #             max_seq_len=131072,
    #             stop='<|im_end|>',
    #         ), 
    #         tool_server='http://127.0.1.1:16181',
    #         tool_meta='data/gta_dataset_v2/toolmeta.json',
    #         batch_size=8,
    #     ),
    # dict(
    #         abbr='qwen1.5-7b-chat',
    #         type=LagentAgent,
    #         agent_type=ReAct,
    #         max_turn=10,
    #         llm=dict(
    #             type=OpenAI,
    #             path='qwen1.5-7b-chat',
    #             key='EMPTY',
    #             openai_api_base='http://127.0.1.1:12580/v1/chat/completions',
    #             query_per_second=1,
    #             max_seq_len=131072,
    #             stop='<|im_end|>',
    #         ), 
    #         tool_server='http://127.0.1.1:16181',
    #         tool_meta='data/gta_dataset_v2/toolmeta.json',
    #         batch_size=8,
    #     ),
    dict(
            abbr='gpt-5',
            type=LagentAgent,
            agent_type=ReAct,
            max_turn=10,
            llm=dict(
                type=OpenAI,
                path='gpt-5-2025-08-07',
                key='EMPTY',
                openai_api_base='https://ai.nengyongai.cn/v1/chat/completions',
                query_per_second=1,
                max_seq_len=131072,
            ), 
            # protocol=protocol,
            tool_server='http://127.0.1.1:16181',
            tool_meta='data/gta_dataset_v2/toolmeta.json',
            batch_size=8,
        ),
    # dict(
    #         abbr='gemini-2.5-pro',
    #         type=LagentAgent,
    #         agent_type=ReAct,
    #         max_turn=10,
    #         llm=dict(
    #             type=OpenAI,
    #             path='google/gemini-2.5-pro',
    #             key='EMPTY',
    #             openai_api_base='https://ai.nengyongai.cn/v1/chat/completions',
    #             query_per_second=1,
    #             max_seq_len=131072,
    #             stop='<|im_end|>',
    #         ), 
    #         tool_server='http://127.0.1.1:16181',
    #         tool_meta='data/gta_dataset_v2/toolmeta.json',
    #         batch_size=8,
    #     ),
    # dict(
    #         abbr='claude-sonnet-4.5',
    #         type=LagentAgent,
    #         agent_type=ReAct,
    #         max_turn=10,
    #         llm=dict(
    #             type=OpenAI,
    #             path='claude-sonnet-4-5-20250929',
    #             key='EMPTY',
    #             openai_api_base='https://ai.nengyongai.cn/v1/chat/completions',
    #             query_per_second=1,
    #             max_seq_len=131072,
    #             stop='<|im_end|>',
    #         ), 
    #         tool_server='http://127.0.1.1:16181',
    #         tool_meta='data/gta_dataset_v2/toolmeta.json',
    #         batch_size=8,
    #     ),
    # dict(
    #         abbr='kimi-k2',
    #         type=LagentAgent,
    #         agent_type=ReAct,
    #         max_turn=10,
    #         llm=dict(
    #             type=OpenAI,
    #             path='kimi-k2-0905-preview',
    #             key='EMPTY',
    #             openai_api_base='https://ai.nengyongai.cn/v1/chat/completions',
    #             query_per_second=1,
    #             max_seq_len=131072,
    #             temperature=0.0,
    #         ), 
    #         protocol=protocol,
    #         tool_server='http://127.0.1.1:16181',
    #         tool_meta='data/gta_dataset_v2/toolmeta.json',
    #         batch_size=8,
    #     ),
    # dict(
    #         abbr='llama-4-scout',
    #         type=LagentAgent,
    #         agent_type=ReAct,
    #         max_turn=10,
    #         llm=dict(
    #             type=OpenAI,
    #             path='meta-llama/llama-4-scout',
    #             key='EMPTY',
    #             openai_api_base='https://ai.nengyongai.cn/v1/chat/completions',
    #             query_per_second=1,
    #             max_seq_len=131072,
    #             stop='<|im_end|>',
    #         ), 
    #         tool_server='http://127.0.1.1:16181',
    #         tool_meta='data/gta_dataset_v2/toolmeta.json',
    #         batch_size=8,
    #     ),
    # dict(
    #         abbr='grok-4',
    #         type=LagentAgent,
    #         agent_type=ReAct,
    #         max_turn=10,
    #         llm=dict(
    #             type=OpenAI,
    #             path='grok-4',
    #             key='EMPTY',
    #             openai_api_base='https://ai.nengyongai.cn/v1/chat/completions',
    #             query_per_second=1,
    #             max_seq_len=131072,
    #             stop='<|im_end|>',
    #         ), 
    #         tool_server='http://127.0.1.1:16181',
    #         tool_meta='data/gta_dataset_v2/toolmeta.json',
    #         batch_size=8,
    #     ),
    # dict(
    #         abbr='qwen3-8b',
    #         type=LagentAgent,
    #         agent_type=ReAct,
    #         max_turn=10,
    #         llm=dict(
    #             type=OpenAI,
    #             path='qwen3-8b',
    #             key='EMPTY',
    #             openai_api_base='http://127.0.1.1:12580/v1/chat/completions',
    #             query_per_second=1,
    #             max_seq_len=131072,
    #             stop='<|im_end|>',
    #         ), 
    #         tool_server='http://127.0.1.1:16181',
    #         tool_meta='data/gta_dataset_v2/toolmeta.json',
    #         batch_size=8,
    #     ),
    # dict(
    #         abbr='llama-3.1-8b-instruct',
    #         type=LagentAgent,
    #         agent_type=ReAct,
    #         max_turn=10,
    #         llm=dict(
    #             type=OpenAI,
    #             path='llama-3.1-8b-instruct',
    #             key='EMPTY',
    #             openai_api_base='http://127.0.1.1:12580/v1/chat/completions',
    #             query_per_second=1,
    #             max_seq_len=131072,
    #             stop='<|im_end|>',
    #         ), 
    #         protocol=protocol,
    #         tool_server='http://127.0.1.1:16181',
    #         tool_meta='data/gta_dataset_v2/toolmeta.json',
    #         batch_size=8,
    #     ),
    # dict(
    #         abbr='llama-3.2-3b-instruct',
    #         type=LagentAgent,
    #         agent_type=ReAct,
    #         max_turn=10,
    #         llm=dict(
    #             type=OpenAI,
    #             path='llama-3.2-3b-instruct',
    #             key='EMPTY',
    #             openai_api_base='http://127.0.1.1:12580/v1/chat/completions',
    #             query_per_second=1,
    #             max_seq_len=131072,
    #             stop='<|im_end|>',
    #         ), 
    #         protocol=protocol,
    #         tool_server='http://127.0.1.1:16181',
    #         tool_meta='data/gta_dataset_v2/toolmeta.json',
    #         batch_size=8,
    #     ),
    # dict(
    #         abbr='llama-3.1-70b-instruct',
    #         type=LagentAgent,
    #         agent_type=ReAct,
    #         max_turn=10,
    #         llm=dict(
    #             type=OpenAI,
    #             path='llama-3.1-70b-instruct',
    #             key='EMPTY',
    #             openai_api_base='http://127.0.1.1:12580/v1/chat/completions',
    #             query_per_second=1,
    #             max_seq_len=131072,
    #             stop='<|im_end|>',
    #         ), 
    #         protocol=protocol,
    #         tool_server='http://127.0.1.1:16181',
    #         tool_meta='data/gta_dataset_v2/toolmeta.json',
    #         batch_size=8,
    #     ),
    # dict(
    #         abbr='qwen3-30b-a3b',
    #         type=LagentAgent,
    #         agent_type=ReAct,
    #         max_turn=10,
    #         llm=dict(
    #             type=OpenAI,
    #             path='qwen3-30b-a3b',
    #             key='EMPTY',
    #             openai_api_base='http://127.0.1.1:12580/v1/chat/completions',
    #             query_per_second=1,
    #             max_seq_len=131072,
    #             stop='<|im_end|>',
    #         ), 
    #         protocol=protocol,
    #         tool_server='http://127.0.1.1:16181',
    #         tool_meta='data/gta_dataset_v2/toolmeta.json',
    #         batch_size=8,
    #     ),
]

# 推理配置（保持不变）
infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=200, gen_task_coef=1),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLInferTask)),
)

# 评测模式缺少显式 eval 配置会导致 --mode eval 分区结果为 0 个任务，评测器未被调用。
# 增加 eval 块：与推理相同的分区策略，runner 使用 OpenICLEvalTask 来读取预测并调用 GPTOSSEvaluator。
eval = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=200, gen_task_coef=1),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLEvalTask)),
)
