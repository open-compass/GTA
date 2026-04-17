import json
import os
from pathlib import Path
from typing import List, Optional
import copy
import re

import numpy as np
from datasets.arrow_dataset import Dataset
from sentence_transformers import SentenceTransformer, util

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET



from .base import BaseDataset  # 你的项目 base dataset 路径

# ---------- 辅助函数 ----------
def get_all_file_paths(directory: str) -> list:
    """
    返回 directory 下所有文件的绝对路径列表（递归）。
    """
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths


def organize_dialogs(sample: dict, path: str) -> List[dict]:
    """
    把原始 sample['dialogs'] 中的文件相对路径替换为绝对路径，
    并保持 list-of-dict 结构，供 parsed 字段使用。
    """
    dialogs = []
    file_paths = get_all_file_paths(path)
    for item in sample.get('dialogs', []):
        # 把 role 'tool' 保留 name/content 字段
        if item.get('role') == 'tool':
            dialog = dict(
                role='tool',
                name=item.get('name'),
                content=item.get('content'),
            )
            dialogs.append(dialog)
        # assistant 带 tool_calls 的情形：把 arguments 中的相对路径替换为绝对路径（若存在）
        elif item.get('role') == 'assistant' and 'tool_calls' in item.keys():
            dialog = copy.deepcopy(item)
            try:
                calls = dialog.get('tool_calls', [])
                if calls and isinstance(calls, list):
                    # 仅处理第一个 tool_call（常见格式）
                    func = calls[0].get('function', {})
                    args = func.get('arguments', {})
                    for name, value in list(args.items()):
                        if isinstance(value, str):
                            candidate = os.path.join(path, value)
                            if candidate in file_paths:
                                dialog['tool_calls'][0]['function']['arguments'][name] = candidate
            except Exception:
                # 忽略解析错误，保留原始
                pass
            dialogs.append(dialog)
        else:
            # user / assistant 普通对话
            dialogs.append(item)
    return dialogs


# ---------- Dataset 类 ----------
@LOAD_DATASET.register_module()
class GTABenchDataset(BaseDataset):
    """GTA Benchmark dataset loader (兼容旧格式并提供 parsed 字段供 agent 使用)。"""

    def __init__(self, reader_cfg: dict = {}, **kwargs):
        """
        kwargs 包含:
          path: dataset 根目录（包含 step.json / end.json）
          mode: 'step' 或 'end'
          prefer_parsed: (可选) bool，若 True 则后续使用者可默认选 parsed 字段；本 loader 只是提供字段，不会主动删除旧字段
        """
        self.dataset = self.load(**kwargs)
        self._init_reader(**reader_cfg)

    @staticmethod
    def load(path: str, mode: str = 'step'):
        """
        加载 step.json 或 end.json。
        返回一个 huggingface Dataset（Dataset.from_list）。
        每条 record 同时包含向后兼容的字符串字段和 parsed 字段：
          - 'dialogs' (str JSON) / 'dialogs_parsed' (list)
          - 'resources' (str JSON) / 'resources_parsed' (list)
          - 'full_tree' (str JSON) / 'full_tree_parsed' (list)
        """
        data_root = Path(path)
        step_file = data_root / 'step.json'
        end_file = data_root / 'end.json'

        # helper: diagnostic + coercion before Dataset.from_list
        def _diagnose_and_coerce(data_list):
            """
            打印每个 key 的 Python 类型集合，如果发现某 key 在不同样本间类型不一致（非 None），
            则把该 key 中的 dict/list 值统一序列化为 JSON 字符串（对所有样本）。
            返回被修改后的 data_list（in-place 修改原列表）。
            """
            from collections import defaultdict
            types_per_key = defaultdict(set)
            examples_per_key = defaultdict(list)

            for idx, row in enumerate(data_list):
                if not isinstance(row, dict):
                    print(f"[GTABenchDataset Debug] WARNING: row {idx} is not a dict: {type(row)}")
                    continue
                for k, v in row.items():
                    types_per_key[k].add(type(v).__name__)
                    if len(examples_per_key[k]) < 6:
                        examples_per_key[k].append((idx, v))

            mixed_keys = []
            for k, tset in types_per_key.items():
                # 忽略只有一种类型或只有 NoneType 的字段
                non_none_types = {t for t in tset if t != 'NoneType'}
                if len(non_none_types) > 1:
                    mixed_keys.append((k, tset))

            if mixed_keys:
                print("[GTABenchDataset Debug] Found columns with multiple Python types (this may break pyarrow):")
                for k, tset in mixed_keys:
                    print(f"  - Key: {k}  types: {sorted(list(tset))}")
                    for idx, sample in examples_per_key[k]:
                        s = repr(sample)
                        if len(s) > 300:
                            s = s[:300] + "...(truncated)"
                        print(f"      idx={idx} sample={s}")
                print("[GTABenchDataset Debug] Strategy: coerce dict/list values to JSON strings for mixed keys to ensure homogeneous column types.")
                # 对混合类型的 key，统一把 dict/list -> JSON 字符串（并将其他类型转为 str 以保证一致性）
                import json
                for row in data_list:
                    if not isinstance(row, dict):
                        continue
                    for k, _ in mixed_keys:
                        if k in row:
                            v = row[k]
                            if isinstance(v, (dict, list)):
                                row[k] = json.dumps(v, ensure_ascii=False)
                            elif v is None:
                                row[k] = None
                            else:
                                # 强制转为字符串（可根据需要改为保持原类型）
                                row[k] = str(v)
            else:
                # 没有发现混合类型，仍然为所有 dict/list 字段做一次轻量序列化保护（可选）
                # 目的是防止后续某些样本携带嵌套结构导致 Arrow 推断为 struct 列
                import json
                for row in data_list:
                    if not isinstance(row, dict):
                        continue
                    for k, v in list(row.items()):
                        if isinstance(v, (dict, list)):
                            # 这里只序列化那些明显是 nested 的字段；如果你想保留 parsed 字段为结构体，可注释掉下面一行
                            # row[k] = json.dumps(v, ensure_ascii=False)
                            # 默认行为：不强制序列化所有 nested 字段，这里注释掉以保留 parsed 字段
                            pass
            return data_list

        # ----------------- step 模式 -----------------
        if mode == 'step' and os.path.exists(step_file):
            data_root = Path(path)
            data_file = data_root / 'step.json'
            assert os.path.exists(data_file), f'Path {path} does not exist.'

            data = json.load(open(data_file))
            data_list = []
            for idx, item in data.items():
                idx = int(idx)
                tools = [
                    dict(type='tool', name=tool['name']) for tool in item['tools']
                ]
                files = [
                    dict(type='file',
                        filetype=file['type'],
                        path=str((data_root / file['path']).absolute()))
                    for file in item['files']
                ]
                gt_answer = item.get('gt_answer', None)
                sample = {
                    'dialogs': json.dumps(organize_dialogs(item, str(data_root.absolute()))),
                    'resources': json.dumps(tools + files),
                    'gt_answer': json.dumps(gt_answer)
                }
                data_list.append(sample)
            dataset = Dataset.from_list(data_list)

            return dataset

        # ----------------- end 模式 -----------------
        elif mode == 'end' and os.path.exists(end_file):
            with open(end_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 加载全量 toolmeta（如果存在）
            # 通过环境变量覆写：OPENCOMPASS_TOOLMETA_PATH（支持 os.pathsep 分隔的多个候选路径）
            default_toolmeta_path = os.path.join('data', 'gta_dataset_v2', 'toolmeta.json')
            toolmeta_env = os.getenv('OPENCOMPASS_TOOLMETA_PATH')
            toolmeta_candidates = []
            if toolmeta_env:
                toolmeta_candidates.extend([p.strip() for p in toolmeta_env.split(os.pathsep) if p.strip()])
            toolmeta_candidates.append(default_toolmeta_path)

            all_toolmeta = None
            loaded_toolmeta_path = None
            for toolmeta_path in toolmeta_candidates:
                if not os.path.exists(toolmeta_path):
                    continue
                try:
                    with open(toolmeta_path, 'r', encoding='utf-8') as f:
                        all_toolmeta = json.load(f)
                    loaded_toolmeta_path = toolmeta_path
                    break
                except Exception:
                    all_toolmeta = None

            if all_toolmeta is not None:
                all_tools = []
                # 向后兼容：toolmeta 可能是 [{"name":...}, ...] 或 ["ToolName", ...]
                for tool in all_toolmeta:
                    name = None
                    if isinstance(tool, dict):
                        name = tool.get('name') or tool.get('tool_name') or tool.get('id')
                    elif isinstance(tool, str):
                        name = tool
                    if name:
                        all_tools.append({'type': 'tool', 'name': name})
                print(f"[GTABenchDataset Debug] Loaded {len(all_tools)} tools from {loaded_toolmeta_path}")
            else:
                all_tools = []

            data_list = []
            for idx, item in data.items():
                # parsed 形式
                dialogs_parsed = organize_dialogs(item, str(data_root.absolute()))
                # 合并全量工具 meta（向后兼容）
                tools = all_tools
                files = []
                for file in item.get('files', []):
                    if isinstance(file, dict) and 'path' in file:
                        files.append(dict(type='file', filetype=file.get('type'), path=str((data_root / file['path']).absolute())))
                resources_parsed = tools + files

                # 向后兼容的字符串字段
                dialogs_str = json.dumps(dialogs_parsed, ensure_ascii=False)
                resources_str = json.dumps(resources_parsed, ensure_ascii=False)
                full_tree_parsed = item.get('sub_tasks', [])

                sample = {
                    'id': idx,
                    'dialogs': dialogs_str,
                    'resources': resources_str,
                    'dialogs_parsed': dialogs_parsed,
                    'resources_parsed': resources_parsed,
                    # full_tree 同时保留字符串与 parsed
                    'full_tree': json.dumps(full_tree_parsed, ensure_ascii=False),
                    'full_tree_parsed': full_tree_parsed
                }
                data_list.append(sample)

            # 诊断并做必要的类型统一化处理（避免 pyarrow 报错）
            data_list = _diagnose_and_coerce(data_list)

            dataset = Dataset.from_list(data_list)
            return dataset

        else:
            raise FileNotFoundError(f'No {mode}.json found in {path}')

    def _init_reader(self, **kwargs):
        try:
            from opencompass.openicl.icl_dataset_reader import DatasetReader
            self.reader = DatasetReader(self.dataset, **kwargs)
        except Exception:
            self.reader = self.dataset

    def __getitem__(self, idx):
        """
        返回结构与之前兼容：
          inputs: [ sample ]  # 完整样本 dict（含 parsed 与 string 字段）
          output: gt_answer（若存在）
        这样上层 pipeline 可通过 sample = inputs[0] 访问完整信息。
        """
        sample = self.dataset[idx]
        inputs = [sample]
        output = sample.get('gt_answer', None)
        return {'inputs': inputs, 'output': output}

    def __len__(self):
        return len(self.dataset)

    @property
    def test(self):
        return self.dataset


from openai import OpenAI

try:
    import httpx  # OpenAI python SDK uses httpx; import for explicit proxy support
except Exception:  # pragma: no cover
    httpx = None


MODEL = os.getenv("EVAL_OPENAI_MODEL", "gpt-5.2")   # 可替换 gpt-4.1 / gpt-4o 等
DEBUG_ENV = "EVAL_DEBUG"


def _resolve_proxy_url(explicit_proxy: Optional[str] = None) -> Optional[str]:
    if explicit_proxy:
        return explicit_proxy
    return (
        os.getenv("EVAL_PROXY")
        or os.getenv("OPENAI_PROXY")
        or os.getenv("ALL_PROXY")
        or os.getenv("HTTPS_PROXY")
        or os.getenv("HTTP_PROXY")
    )


def _build_openai_client(proxy: Optional[str] = None) -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("EVAL_OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("EVAL_OPENAI_BASE_URL")
    proxy_url = _resolve_proxy_url(proxy)

    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    if proxy_url:
        if httpx is None:
            print(f"[GPTEvaluator] proxy set but httpx unavailable: {proxy_url}")
        else:
            try:
                # httpx API changed across versions: proxy vs proxies
                try:
                    http_client = httpx.Client(proxy=proxy_url)
                except TypeError:
                    http_client = httpx.Client(proxies=proxy_url)
                kwargs["http_client"] = http_client
                print(f"[GPTEvaluator] Using proxy: {proxy_url}")
            except Exception as e:
                print(f"[GPTEvaluator] Failed to init proxy client ({proxy_url}): {e}")

    return OpenAI(**kwargs)


import mimetypes
import subprocess
import tempfile
from pathlib import Path

SUPPORTED_FILE_EXT = {
    ".pdf",
    ".png", ".jpg", ".jpeg", ".webp",
    ".mp3", ".wav", ".m4a",
    ".mp4", ".mov", ".webm"
}


@ICL_EVALUATORS.register_module()
class GPTEvaluator(BaseEvaluator):

    """
    Official OpenAI Responses API Multimodal Evaluator

    - True file / audio / video / image reading
    - Checkpoint weighted aggregation
    - Leaf checkpoint scored by GPT
    """

    def __init__(self, mode="every", debug=None, proxy: Optional[str] = None):

        assert mode == "every"

        self.mode = mode
        self.debug = debug if debug is not None else \
            str(os.getenv(DEBUG_ENV, "true")).lower() in ("1", "true", "yes")

        self.proxy = proxy
        self.client = _build_openai_client(proxy=self.proxy)

        print(f"[GPTEvaluator] Init | model={MODEL} | debug={self.debug}")


    def _prepare_upload_file(self, path):
        """
        Convert unsupported formats to PDF if needed.
        Return final path for upload.
        """

        suffix = Path(path).suffix.lower()

        if suffix in SUPPORTED_FILE_EXT:
            return path

        if suffix in [".html", ".txt", ".md", ".json"]:

            pdf_path = self._convert_to_pdf(path)

            if pdf_path and os.path.exists(pdf_path):
                if self.debug:
                    print(f"[Convert] {path} -> {pdf_path}")
                return pdf_path

            print(f"[Convert Failed] {path}")
            return None

        print(f"[Skip Unsupported] {path}")
        return None

    def _convert_to_pdf(self, path):

        try:
            out_dir = tempfile.mkdtemp()
            out_pdf = os.path.join(
                out_dir,
                Path(path).stem + ".pdf"
            )

            cmd = [
                "wkhtmltopdf",
                path,
                out_pdf
            ]

            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15
            )

            if os.path.exists(out_pdf):
                return out_pdf

        except Exception as e:
            print(f"[PDF Convert Error] {e}")

        return None
    
    def _extract_video_frames(self, path, max_frames=4):

        out_dir = tempfile.mkdtemp()
        out_pattern = os.path.join(out_dir, "frame_%03d.jpg")

        cmd = [
            "ffmpeg", "-i", path,
            "-vf", f"fps={max_frames}/10",
            out_pattern
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return sorted(Path(out_dir).glob("*.jpg"))


    def _infer_file_type(self, path):

        ext = Path(path).suffix.lower()

        if ext in [".png", ".jpg", ".jpeg", ".webp"]:
            return "image"

        if ext in [".mp3", ".wav", ".m4a"]:
            return "audio"

        if ext in [".mp4", ".mov", ".webm"]:
            return "video"

        return "file"


    def _upload_file(self, path: str):

        try:

            purpose = "user_data"

            if self.debug:
                print(f"[Upload] file={path} purpose={purpose}")

            with open(path, "rb") as f:

                res = self.client.files.create(
                    file=f,
                    purpose=purpose
                )

            return res.id

        except Exception as e:

            print(f"[Upload Warning] Failed to upload {path}: {e}")
            return None



    # ==================================================
    # Entry
    # ==================================================

    def score(
        self,
        predictions=None,
        references=None,
        gold=None,
        checkpoints=None,
        assistant_outputs=None,
        attached_files=None,
        origin_prompt=None,
        indices=None,
        **kwargs
    ):

        return self._score_with_checkpoint(
            assistant_outputs,
            checkpoints,
            origin_prompt,
            attached_files,
            indices
        )

    # ==================================================
    # Main Loop
    # ==================================================

    def _score_with_checkpoint(
        self,
        assistant_outputs,
        checkpoints,
        origin_prompts,
        attached_files,
        indices
    ):

        if origin_prompts is None:
            origin_prompts = [None] * len(assistant_outputs)

        scores = []
        details = []

        for i, (trace, ckpt) in enumerate(zip(assistant_outputs, checkpoints)):

            sid = indices[i] if indices else i

            print(f"\n========== SAMPLE {sid} ==========")

            tree = self._normalize_tree(ckpt)

            node_logs = []

            def dfs(node, depth=0):

                indent = "  " * depth
                nid = node.get("id", "unknown")

                print(f"{indent}[Checkpoint] {nid}")

                children = node.get("sub_tasks", [])

                # ---------------- Leaf ----------------

                if not children:

                    prompt = self._build_leaf_prompt(
                        node,
                        trace,
                        origin_prompts[i]
                    )

                    raw = self._call_openai(
                        prompt,
                        attached_files[i] if isinstance(attached_files, list) and i < len(attached_files) else attached_files
                    )

                    parsed = self._parse_score(raw)

                    print(f"{indent}=> score={parsed['score']}")

                    node_logs.append({
                        "id": nid,
                        "score": parsed["score"],
                        "analysis": parsed["analysis"]
                    })

                    return parsed["score"]

                # ---------------- Internal ----------------

                total_s, total_w = 0, 0

                for c in children:
                    w = c.get("weight", 1)
                    s = dfs(c, depth + 1)

                    total_s += s * w
                    total_w += w

                agg = total_s / total_w if total_w else 0

                print(f"{indent}=> aggregated={agg:.2f}")

                return agg

            root_score = dfs(tree)

            scores.append(root_score)

            details.append({
                "sample_index": sid,
                "score": root_score,
                "nodes": node_logs
            })

        final = float(np.mean(scores)) if scores else 0

        return {
            "gpt_score": final,
            "sample_scores": scores,
            "details": details
        }

    # ==================================================
    # Tree Utils
    # ==================================================

    def _normalize_tree(self, tree):

        if isinstance(tree, list):
            return {"id": "root", "sub_tasks": tree}

        if isinstance(tree, dict):
            if "sub_tasks" not in tree:
                return {"id": "root", "sub_tasks": [tree]}
            return tree

        raise ValueError("Invalid checkpoint tree")

    # ==================================================
    # Prompt
    # ==================================================

    def _build_leaf_prompt(self, node, trace, origin_prompt):

        req = node.get("requirements", "(none)")

        task = ""
        if origin_prompt:
            task = "\n".join(origin_prompt) if isinstance(origin_prompt, list) else str(origin_prompt)

        def _flatten_steps(obj):
            if obj is None:
                return []
            if isinstance(obj, list):
                out = []
                for item in obj:
                    out.extend(_flatten_steps(item))
                return out
            if isinstance(obj, tuple):
                out = []
                for item in obj:
                    out.extend(_flatten_steps(item))
                return out
            if isinstance(obj, dict):
                return [obj]
            return [obj]

        def _extract_final_answer(steps):
            flat = _flatten_steps(steps)
            # Prefer last assistant content
            for item in reversed(flat):
                if isinstance(item, dict):
                    if item.get("role") == "assistant":
                        content = item.get("content")
                        if isinstance(content, str) and content.strip():
                            return content.strip()
                elif isinstance(item, str) and item.strip():
                    return item.strip()
            return ""

        final_answer = _extract_final_answer(trace) or "(empty)"

        return f"""
You are a professional LLM Agent evaluator. All of the generated files(converted to pdf) and images from the LLM Agent are uploaded, videos are converted to several image frames, and audio files are checked for existence.

Original task:
{task}

Final answer:
{final_answer}

Checkpoint requirement:
{req}

Please score this checkpoint from 0 to 10. If required files are missing, give a low score.

Only output valid JSON:
{{"score": number, "analysis": "short explanation (<80 words)"}}
""".strip()

    # ==================================================
    # OpenAI Responses Call
    # ==================================================

    def _call_openai(self, prompt: str, attached_files):

        content_blocks = [
            {"type": "input_text", "text": prompt}
        ]

        # -------- Attach multimodal files --------

        def _flatten_files(obj):
            if obj is None:
                return []
            if isinstance(obj, list):
                out = []
                for item in obj:
                    out.extend(_flatten_files(item))
                return out
            if isinstance(obj, tuple):
                out = []
                for item in obj:
                    out.extend(_flatten_files(item))
                return out
            return [obj]

        def _normalize_file(item):
            if isinstance(item, dict):
                path = item.get("path")
                ftype = self._infer_file_type(path)
                if path:
                    return {"path": path, "type": ftype}
                return None
            if isinstance(item, str) and item.strip():
                return {"path": item.strip(), "type": None}
            return None

        flat_files = []
        for it in _flatten_files(attached_files):
            norm = _normalize_file(it)
            if norm is not None:
                flat_files.append(norm)

        if flat_files:

            for f in flat_files:

                path = f.get("path")
                ftype = self._infer_file_type(path)

                if not path:
                    continue

                final_path = self._prepare_upload_file(path)

                if not final_path:
                    if self.debug:
                        print(f"[Upload Skip] unsupported: {path}")
                    continue

                file_id = self._upload_file(final_path)


                if not file_id:
                    if self.debug:
                        print(f"[Upload] skipped (no file_id) for {path}")
                    continue

                if self.debug:
                    print(f"[Upload] {ftype} -> {file_id}")

                if ftype == "image":
                    content_blocks.append({"type": "input_image", "file_id": file_id})

                elif ftype == "audio":

                    content_blocks.append({
                        "type": "input_text",
                        "text": f"[Attached audio file exists: {os.path.basename(path)}]"
                    })

                elif ftype == "video":

                    frames = self._extract_video_frames(path)

                    for img in frames:
                        img_id = self._upload_file(str(img))

                        if img_id:
                            content_blocks.append({
                                "type": "input_image",
                                "file_id": img_id
                            })

                else:
                    content_blocks.append({"type": "input_file", "file_id": file_id})

        if self.debug:
            print("\n[Responses Input Preview]")
            print(json.dumps(content_blocks, ensure_ascii=False)[:1200])

        # -------- OpenAI Responses API --------

        response = self.client.responses.create(
            model=MODEL,
            input=[
                {
                    "role": "user",
                    "content": content_blocks
                }
            ],
            temperature=0.2
        )

        if self.debug:
            print("\n[Responses Raw Output]")
            print(response)

        # Unified text extraction
        return response.output_text

    # ==================================================
    # Parser
    # ==================================================

    @staticmethod
    def _parse_score(text):

        try:
            data = json.loads(text)
            s = float(data.get("score", 0))
            s = max(0, min(10, s))
            return {"score": s, "analysis": data.get("analysis", "")}
        except Exception:
            pass

        # fallback
        m = re.search(r"\d+(\.\d+)?", text)
        if m:
            return {"score": min(10, float(m.group())), "analysis": "regex fallback"}

        return {"score": 0, "analysis": "parse failed"}
