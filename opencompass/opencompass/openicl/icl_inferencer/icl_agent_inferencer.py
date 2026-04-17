import json
import os
import os.path as osp
import types
from typing import List
from itertools import takewhile

from opencompass.models.lagent import LagentAgent
from opencompass.registry import ICL_INFERENCERS

from ..utils.logging import get_logger
from .icl_base_inferencer import dump_results_dict
from .icl_chat_inferencer import ChatInferencer

logger = get_logger(__name__)

class AgentInferencerOutputHandler:
    """
    Output handler for AgentInferencer.

    现在支持：
      - 保存多轮结果（save_multiround_results）并记录 attached_files 字段（列表）
      - write_to_json 会把整体 results_dict 写盘（使用 dump_results_dict）

    并发/多进程注意：
      - 如果 inferencer 在多进程/多线程环境下并发调用 save_* 方法，
        results_dict 的内存写入对多线程通常安全（GIL），但对多进程并不共享，
        且最后 dump 到磁盘时需要保证只有主进程执行写入或采用每-sample 文件合并策略。
      - 更稳妥的做法是在每个 worker 中写 per-sample JSON 文件到同一 outputs 目录，
        最后由主进程合并；或在写共享文件时使用 file lock（例如 filelock 库）。
    """

    def __init__(self) -> None:
        self.results_dict = {}

    def write_to_json(self, save_dir: str, filename: str):
        """Dump the result to a json file."""
        dump_results_dict(self.results_dict, osp.join(save_dir, filename))

    def save_results(self,
                     origin_prompt: list,
                     prediction: str,
                     idx: int,
                     gold: str = None):
        result_dict = {}
        if gold:
            result_dict['gold'] = gold
        result_dict.update({
            'prediction': prediction,
            'origin_prompt': origin_prompt,
        })
        self.results_dict[str(idx)] = result_dict

    def save_multiround_results(self,
                                origin_prompt: list,
                                prediction: str,
                                idx: int,
                                gold: str = None,
                                full_tree: dict = None,
                                                                attached_files: list = None,
                                                                assistant_outputs: list = None):
        """
        保存多回合结果，并可附加 agent 写出的文件路径列表 attached_files。

        参数:
          - origin_prompt: 原始 prompt / 上下文
          - prediction: agent 输出（字符串或更复杂结构）
          - idx: 样本序号（或 id）
          - gold: 人类参考答案（若有）
          - full_tree: 若存在的任务树结构
          - attached_files: agent 写入的文件路径列表（绝对或相对路径）
          - assistant_outputs: agent 在该回合输出的自然语言文本（列表）
        """
        result_dict = self.results_dict.get(str(idx), {
            'gold': [],
            'prediction': [],
            'origin_prompt': [],
            'steps': [],
            'full_tree': None,
            'attached_files': [],
            'assistant_outputs': []
        })
        if gold is not None:
            result_dict['gold'].append(gold)
        result_dict['prediction'].append(prediction)
        result_dict['origin_prompt'].append(origin_prompt)
        if full_tree is not None:
            result_dict['full_tree'] = full_tree
        if attached_files:
            # 追加（可能多回合都会产出文件）
            result_dict.setdefault('attached_files', [])
            # 过滤重复并保持原有顺序
            for p in attached_files:
                if p not in result_dict['attached_files']:
                    result_dict['attached_files'].append(p)
        if assistant_outputs:
            result_dict.setdefault('assistant_outputs', [])
            result_dict['assistant_outputs'].append(assistant_outputs)
        self.results_dict[str(idx)] = result_dict


@ICL_INFERENCERS.register_module()
class AgentInferencer(ChatInferencer):
    HandlerType = AgentInferencerOutputHandler

    def __init__(self, model, **kwargs) -> None:
        super().__init__(model, **kwargs)

    def get_chat_list(self,
                      ice_idx_list: List[List[int]],
                      retriever,
                      prompt_template=None) -> List[dict]:
        prompt_list = []
        input_columns = retriever.dataset_reader.input_columns

        # 加载 toolmeta 内容（若存在）并可选注入到 prompt。
        # 兼容旧版本：默认不把 toolmeta 拼接到 prompt 前；只有显式提供环境变量时才注入。
        #
        # 用法：
        #   export OPENCOMPASS_TOOLMETA_PATH=data/gta_dataset_v2/toolmeta.json
        #   # 可用多个路径（按 os.pathsep 分隔，如 ':'），会按顺序尝试加载
        toolmeta_str = ''
        toolmeta_env = os.getenv('OPENCOMPASS_TOOLMETA_PATH')
        toolmeta_paths = []
        if toolmeta_env:
            for p in toolmeta_env.split(os.pathsep):
                p = p.strip()
                if p:
                    toolmeta_paths.append(p)

        for toolmeta_path in toolmeta_paths:
            if not osp.exists(toolmeta_path):
                continue
            try:
                with open(toolmeta_path, 'r') as f:
                    toolmeta = json.load(f)
                # 拼接为可读字符串
                toolmeta_str = '\n'.join([
                    f"Tool names: {tool.get('name', '')}\nDescription: {tool.get('description', '')}\nArguments: {json.dumps(tool.get('parameters', {}), ensure_ascii=False)}\n" for tool in toolmeta
                ])
                toolmeta_str = '[Available Tool List]\n' + toolmeta_str
                break
            except Exception:
                toolmeta_str = ''

        for idx, ice_idx in enumerate(ice_idx_list):
            entry = {
                k: json.loads(item)
                for k, item in retriever.test_ds[idx].items()
                if k in input_columns
            }
            # 兼容 full_tree 字段
            if 'full_tree' in retriever.test_ds[idx]:
                try:
                    entry['full_tree'] = json.loads(retriever.test_ds[idx]['full_tree'])
                except Exception:
                    entry['full_tree'] = retriever.test_ds[idx]['full_tree']


            # 将 toolmeta 内容加入 prompt（在第一个 user prompt 前拼接）
            # 兼容旧版本：当 OPENCOMPASS_TOOLMETA_PATH 未设置时，toolmeta_str 为空，不会注入。
            if toolmeta_str and 'dialogs' in entry and isinstance(entry['dialogs'], list) and len(entry['dialogs']) > 0:
                for dialog in entry['dialogs']:
                    if dialog.get('role') == 'user':
                        dialog['content'] = toolmeta_str + '\n' + dialog['content']
                        break
            prompt_list.append(entry)
        return prompt_list


    def inference(self,
                  retriever,
                  ice_template=None,
                  prompt_template=None,
                  output_json_filepath=None,
                  output_json_filename=None):
        return super().inference(
            retriever,
            ice_template=ice_template,
            prompt_template=prompt_template,
            output_json_filepath=output_json_filepath,
            output_json_filename=output_json_filename
        )

    def infer_last(self, chat: dict, index: int, output_handler):
        raise NotImplementedError

    def infer_every(self, chat: dict, index: int, output_handler):
        """
        对于每个 user 提交，调用 model.chat(...) 得到 steps（模型内部可含 tool 调用）。
        额外生成一个“压缩工具调用”的版本 assistant_outputs_compressed，
        保留原 steps 为 prediction，压缩连续失败的工具调用。
        """
        dialogs = chat['dialogs']
        user_indices = [i for i, item in enumerate(dialogs) if item['role'] == 'user']

        memory = None
        full_tree = chat.get('full_tree', None)
        for i in user_indices:
            # 调用模型
            try:
                steps, memory = self.model.chat(
                    query=dialogs[i]['content'],
                    memory=memory,
                    resources=chat.get('resources'),
                )
            except Exception as e:
                # 单条样本失败不应影响整体评测流程：记录错误并结束该样本的多轮推理。
                steps = [
                    {
                        'role': 'assistant',
                        'content': f'[AgentInferencerError] {type(e).__name__}: {e}',
                        'meta': {'type': 'error'},
                    }
                ]
                # 保留已有 memory（若为 None 则初始化），避免后续代码崩溃。
                if memory is None:
                    memory = []

            # --- 🔹 从 steps 中解析出 attached_files ---
            attached_files = []
            allowed_suffixes = (
                '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp',
                '.pdf', '.docx', '.pptx', '.ppt', '.xlsx', '.xls',
                '.csv', '.txt', '.md', '.json', '.html', '.zip', 
                '.mp4', '.mp3'
            )

            def _maybe_add_path(raw: str):
                if not isinstance(raw, str):
                    return
                candidate = raw.strip()
                if not candidate:
                    return
                candidates = []
                if candidate.startswith('sandbox:/'):
                    sanitized = candidate[len('sandbox:/'):]
                    if sanitized:
                        candidates.append(sanitized)
                    candidates.append(candidate)
                elif candidate.startswith('sandbox:'):
                    sanitized = candidate[len('sandbox:'):]
                    if sanitized:
                        candidates.append(sanitized)
                    candidates.append(candidate)
                else:
                    candidates.append(candidate)

                for path in candidates:
                    lower = path.lower()
                    if any(lower.endswith(ext) for ext in allowed_suffixes):
                        if '/' not in path and '\\' not in path:
                            continue
                        if path not in attached_files:
                            attached_files.append(path)
                        break

            def _collect_from_obj(obj):
                if isinstance(obj, str):
                    _maybe_add_path(obj)
                elif isinstance(obj, dict):
                    for value in obj.values():
                        _collect_from_obj(value)
                elif isinstance(obj, (list, tuple, set)):
                    for item in obj:
                        _collect_from_obj(item)

            assistant_outputs = []
            try:
                for step in steps:
                    if isinstance(step, dict):
                        if step.get('role') == 'assistant' and step.get('content'):
                            assistant_outputs.append(step['content'])
                        if 'observation' in step:
                            _collect_from_obj(step['observation'])
                        if step.get('role') == 'tool' and 'content' in step:
                            _collect_from_obj(step['content'])
                        continue

                    obs = getattr(step, 'observation', None)
                    if obs:
                        _collect_from_obj(obs)
                    role = getattr(step, 'role', None)
                    content = getattr(step, 'content', None)
                    if role == 'assistant' and content:
                        assistant_outputs.append(content)
                    if role == 'tool' and content is not None:
                        _collect_from_obj(content)
            except Exception:
                attached_files = []
                assistant_outputs = []

            # --- 🔹 生成压缩版 assistant output ---
            def _compress_tool_calls_with_retry(steps: list):
                import re
                import json

                merged = []
                n = len(steps)
                idx = 0

                def _is_tool_output_failed(content):
                    # 与原实现一致的失败检测
                    if content is None:
                        return True
                    if isinstance(content, (str, list, dict)) and len(content) == 0:
                        return True
                    texts = []
                    def _collect_texts(obj):
                        if obj is None:
                            return
                        if isinstance(obj, str):
                            texts.append(obj)
                        elif isinstance(obj, dict):
                            if isinstance(obj.get("content"), str):
                                texts.append(obj["content"])
                            for v in obj.values():
                                _collect_texts(v)
                        elif isinstance(obj, (list, tuple, set)):
                            for it in obj:
                                _collect_texts(it)
                    _collect_texts(content)
                    for t in texts:
                        s = t.strip().lower()
                        if re.search(r"\b(error|exception|traceback|failed|failure|invalid|bad request|typeerror|not found|not available|forbidden|unauthorized|403|404|500)\b", s):
                            return True
                    return False

                def _parse_arguments(arg_field):
                    if not arg_field:
                        return {}
                    if isinstance(arg_field, dict):
                        return arg_field
                    if isinstance(arg_field, str):
                        try:
                            return json.loads(arg_field)
                        except Exception:
                            return {"_raw": arg_field}
                    return {"_raw": str(arg_field)}

                # Helper: collect consecutive assistant-thoughts starting at idx
                def _collect_thoughts(start_idx):
                    thoughts = []
                    i = start_idx
                    while i < n:
                        s = steps[i]
                        if not isinstance(s, dict):
                            break
                        if s.get("role") == "assistant" and "tool_calls" not in s:
                            c = s.get("content")
                            if isinstance(c, str) and c.strip():
                                thoughts.append(c)
                            i += 1
                        else:
                            break
                    return thoughts, i

                while idx < n:
                    step = steps[idx]

                    # 非 dict 的 step：直接透传（视为边界）
                    if not isinstance(step, dict):
                        merged.append(step)
                        idx += 1
                        continue

                    role = step.get("role")

                    # 情形 A：连续的思考模块（assistant 且无 tool_calls）
                    if role == "assistant" and "tool_calls" not in step:
                        thoughts, j = _collect_thoughts(idx)
                        # 输出这个思考模块（可以为空，但若为空则不输出）
                        if thoughts:
                            merged.append({"role": "assistant", "content": "\n".join(thoughts).strip()})
                        idx = j
                        continue

                    # 情形 B：工具模块开始（assistant 带 tool_calls）
                    if role == "assistant" and "tool_calls" in step:
                        # 在开始工具模块之前，保证上一个思考模块已被处理（上文逻辑会先处理思考模块）
                        # 现在开始收集该工具模块内的 attempts 和 tool 输出，直到结算（成功或最终失败）
                        # 解析当前 attempt 的工具名与参数（但不要把本 step 的 content 作为 thought）
                        def _get_tool_name_and_args(s):
                            tcall = s["tool_calls"][0]
                            func = tcall.get("function", {}) if isinstance(tcall, dict) else {}
                            name = func.get("name") or tcall.get("name") or "UnknownTool"
                            args = _parse_arguments(func.get("arguments") or tcall.get("arguments"))
                            return name, args

                        current_tool, current_args = _get_tool_name_and_args(step)
                        attempts = 1
                        last_args = current_args or {}
                        had_success = False
                        saw_any_tool_output = False
                        idx += 1  # consume this assistant.tool_calls step (attempt)

                        # 进入收集工具模块的循环
                        while idx < n:
                            s = steps[idx]
                            # 非 dict 或 非 assistant/tool 的其他 step -> 视为边界（若未成功则作为最终失败结算）
                            if not isinstance(s, dict):
                                break

                            r = s.get("role")

                            # 如果遇到另一个 assistant.tool_calls
                            if r == "assistant" and "tool_calls" in s:
                                # 判断是否同工具
                                name2, args2 = _get_tool_name_and_args(s)
                                if name2 == current_tool:
                                    # 同工具的重试 attempt
                                    attempts += 1
                                    last_args = args2 or last_args
                                    idx += 1  # consume this attempt
                                    continue
                                else:
                                    # 不同工具的调用出现，按你的规则：将当前工具模块作为最终失败结算（如果还未成功）
                                    break

                            # 工具输出（tool）——可能是成功或失败
                            if r == "tool":
                                saw_any_tool_output = True
                                failed = _is_tool_output_failed(s.get("content"))
                                idx += 1  # consume this tool output
                                if failed:
                                    # 失败：继续循环，可能会有同工具的后续 attempt（handled above）
                                    # 标记未成功，继续等待后续 attempt or boundary
                                    continue
                                else:
                                    # 成功：立即结算成功（先不输出任何thought，因为这些被视为工具模块内部）
                                    merged.append({
                                        "role": "assistant",
                                        "tool_calls_summary": [{
                                            "name": current_tool,
                                            "last_arguments": last_args,
                                            "success": True,
                                            "attempts": attempts
                                        }]
                                    })
                                    had_success = True
                                    break  # 工具模块已结算，退出内部循环
                            # 其他角色（user/system/…），作为边界，停止工具模块处理（会在外面被处理）
                            else:
                                break

                        # 退出内部循环：如果未成功结算（had_success == False），则按最终失败结算
                        if not had_success:
                            # 按规则：工具模块只输出 summary（不包含模块内的 thought）
                            merged.append({
                                "role": "assistant",
                                "tool_calls_summary": [{
                                    "name": current_tool,
                                    "last_arguments": last_args,
                                    "success": False,
                                    "attempts": attempts
                                }]
                            })
                        # do NOT consume or flush thoughts here — next outer loop iteration will handle subsequent thoughts or next tool_modules
                        continue

                    # 其他角色（user/system 等）直接透传
                    merged.append(step)
                    idx += 1

                # 结尾：如果最后是思考模块该逻辑已在循环内输出；若末尾为其他需保持原样（已处理）
                return merged


            assistant_outputs_compressed = _compress_tool_calls_with_retry(steps)

            # --- ✅ 保存原始与整合版 ---
            output_handler.save_multiround_results(
                origin_prompt=dialogs[i]['content'],
                prediction=steps,
                gold=list(takewhile(lambda j: j['role'] != 'user', dialogs[i + 1:])),
                idx=index,
                full_tree=full_tree,
                attached_files=attached_files,
                assistant_outputs=assistant_outputs_compressed,
            )

            # 如果本轮已经发生错误，则停止该样本的后续轮次，避免无意义的级联错误。
            if steps and isinstance(steps, list) and isinstance(steps[0], dict):
                meta = steps[0].get('meta')
                if isinstance(meta, dict) and meta.get('type') == 'error':
                    break

        self.model.reset()



    def infer_every_with_gt(self, chat: dict, index: int, output_handler):
        dialogs = chat['dialogs']
        assistant_indices = [
            i for i, item in enumerate(dialogs) if item['role'] == 'assistant'
        ]

        for idx in range(len(assistant_indices)):
            i = assistant_indices[idx]
            stop = True if idx == len(assistant_indices) - 1 else False
            step = self.model.next_step(
                history=dialogs[:i],
                resources=chat.get('resources'),
                stop=stop
            )
            output_handler.save_multiround_results(
                origin_prompt=dialogs[:i],
                prediction=step,
                gold=dialogs[i],
                idx=index,
            )
            self.model.reset()
