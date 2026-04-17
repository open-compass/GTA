"""Run GTA v2 Agent-Application tasks with external agents (Manus/OpenClaw).

Design goals
- No OpenCompass/Lagent runtime dependency for inference.
- Emit an eval-pack style JSON compatible with the existing GPT-5.2 checkpoint evaluator.
- Record per-task elapsed time and usage (if returned by the provider).

This script is intentionally minimal and provider-agnostic. Only Manus is implemented
end-to-end; OpenClaw is scaffolded (dry-run supported) because its API/CLI is not
specified yet.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import requests

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None

try:
    # Optional: only used for best-effort download of OpenAI-style `file_id` outputs.
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


SUPPORTED_INPUT_EXT = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".csv",
    ".md",
    ".xlsx",
    ".mp3",
    ".wav",
    ".m4a",
    ".mp4",
    ".mov",
    ".webm",
}


def _guess_mime_type(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or "application/octet-stream"


def _extract_file_id(data: Any) -> Optional[str]:
    """Best-effort extraction of an OpenAI-style uploaded file id."""

    if not isinstance(data, dict):
        return None

    for key in ("id", "file_id", "fileId"):
        v = data.get(key)
        if isinstance(v, str) and v:
            return v

    if isinstance(data.get("data"), dict):
        for key in ("id", "file_id", "fileId"):
            v = data["data"].get(key)
            if isinstance(v, str) and v:
                return v

    if isinstance(data.get("file"), dict):
        for key in ("id", "file_id", "fileId", "name"):
            v = data["file"].get(key)
            if isinstance(v, str) and v:
                return v

    return None


class ManusHTTPError(RuntimeError):
    def __init__(self, message: str, status_code: Optional[int] = None, data: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.data = data


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _utc_iso_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _relpath_from_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_repo_root()))
    except Exception:
        return str(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_task_ids(path: Path) -> List[int]:
    out: List[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        out.append(int(s))
    return out


def _infer_resource_type_from_path(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image"
    if ext in {".mp3", ".wav", ".m4a"}:
        return "audio"
    if ext in {".mp4", ".mov", ".webm"}:
        return "video"
    return "file"


def _normalize_manus_api_base_url(base_url: str) -> str:
    """Normalize Manus API base URL to an api_base that ends with `/v1`.

    The repository's OpenCompass wrapper (`opencompass/models/manus_api.py`) uses the
    Manus Tasks API:
      - POST   <api_base>/tasks
      - GET    <api_base>/tasks/{task_id}

    Users sometimes pass:
      - https://api.manus.ai
      - https://api.manus.ai/v1
      - https://api.manus.ai/v1/tasks
    """

    u = (base_url or "").strip().rstrip("/")
    if not u:
        return "https://api.manus.ai/v1"

    if u.endswith("/tasks"):
        u = u[: -len("/tasks")]

    if u.endswith("/v1"):
        return u

    # If user provided a longer base containing /v1/..., trim to .../v1.
    if "/v1/" in u:
        return u.split("/v1/", 1)[0] + "/v1"

    return u + "/v1"


def _extract_manus_task_id(data: Any) -> Optional[str]:
    if not isinstance(data, dict):
        return None

    for key in ("task_id", "taskId", "id"):
        v = data.get(key)
        if isinstance(v, str) and v:
            return v

    if isinstance(data.get("data"), dict):
        for key in ("task_id", "taskId", "id"):
            v = data["data"].get(key)
            if isinstance(v, str) and v:
                return v

    return None


def _extract_manus_status(data: Any) -> Optional[str]:
    if not isinstance(data, dict):
        return None

    for key in ("status", "state"):
        v = data.get(key)
        if isinstance(v, str) and v:
            return v

    if isinstance(data.get("data"), dict):
        for key in ("status", "state"):
            v = data["data"].get(key)
            if isinstance(v, str) and v:
                return v

    return None


def _extract_manus_output_text(data: Any) -> str:
    """Best-effort output extraction for Manus Tasks API responses.

    Manus' payload schema may vary across versions. This function is defensive
    and attempts several common patterns to locate assistant-visible final text.
    """

    def _extract_text_deep(obj: Any, depth: int = 0) -> str:
        if depth > 8 or obj is None:
            return ""

        if isinstance(obj, str):
            s = obj.strip()
            return s

        if isinstance(obj, (int, float, bool)):
            return ""

        if isinstance(obj, dict):
            role = obj.get("role") or obj.get("sender")
            if isinstance(role, str) and role.lower() in {"assistant", "bot"}:
                for k in ("content", "text", "answer", "output", "result"):
                    s = _extract_text_deep(obj.get(k), depth + 1)
                    if s:
                        return s

            # Common explicit keys.
            for k in (
                "final_answer",
                "finalAnswer",
                "final",
                "answer",
                "output_text",
                "outputText",
                "output",
                "result",
                "text",
                "completion",
                "content",
            ):
                if k in obj:
                    s = _extract_text_deep(obj.get(k), depth + 1)
                    if s:
                        return s

            # Common nesting.
            for k in ("data", "result", "message", "output"):
                if k in obj:
                    s = _extract_text_deep(obj.get(k), depth + 1)
                    if s:
                        return s

            # Common message lists.
            for k in ("messages", "history", "dialog", "outputs", "steps", "choices"):
                if k in obj:
                    s = _extract_text_deep(obj.get(k), depth + 1)
                    if s:
                        return s

            return ""

        if isinstance(obj, list):
            # Prefer the last assistant message if present.
            for item in reversed(obj):
                if isinstance(item, dict):
                    role = item.get("role") or item.get("sender")
                    if isinstance(role, str) and role.lower() in {"assistant", "bot"}:
                        s = _extract_text_deep(item, depth + 1)
                        if s:
                            return s

            # Common content-block pattern: [{"type":"text","text":"..."}, ...]
            parts: List[str] = []
            for item in obj:
                if not isinstance(item, dict):
                    continue
                t = item.get("type")
                if t in {"text", "output_text"} and isinstance(item.get("text"), str):
                    parts.append(item["text"].strip())
            joined = "\n".join([p for p in parts if p])
            if joined.strip():
                return joined.strip()

            # Fallback: first non-empty extraction.
            for item in obj:
                s = _extract_text_deep(item, depth + 1)
                if s:
                    return s

            return ""

        return ""

    if not isinstance(data, dict):
        return ""

    # Fast-path for official Manus Tasks schema:
    # GET /v1/tasks/{task_id} returns `output`: [{role, content:[{type:"output_text", text:"...", ...}]}]
    out = data.get("output")
    if isinstance(out, list) and out:
        def _content_blocks_to_text(blocks: Any) -> str:
            if not isinstance(blocks, list):
                return ""
            parts: List[str] = []
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                if b.get("type") in {"output_text", "text"} and isinstance(b.get("text"), str):
                    s = b["text"].strip()
                    if s:
                        parts.append(s)
            return "\n".join(parts).strip()

        # Prefer the last assistant message.
        for msg in reversed(out):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role") or msg.get("sender")
            if isinstance(role, str) and role.lower() in {"assistant", "bot"}:
                s = _content_blocks_to_text(msg.get("content"))
                if s:
                    return s

        # Fallback: if role is missing/unreliable, use last message's output_text.
        last_msg = out[-1]
        if isinstance(last_msg, dict):
            s = _content_blocks_to_text(last_msg.get("content"))
            if s:
                return s

    # 1) Prefer any assistant-role message found in the tree.
    last_assistant = ""
    for node in _walk(data):
        if not isinstance(node, dict):
            continue
        role = node.get("role") or node.get("sender")
        if isinstance(role, str) and role.lower() in {"assistant", "bot"}:
            for k in ("content", "text", "answer", "output", "result"):
                s = _extract_text_deep(node.get(k))
                if s:
                    last_assistant = s
    if last_assistant:
        return last_assistant

    # 2) Try common output fields near the top.
    for key in ("output", "result", "text", "completion", "answer", "content", "data", "message"):
        if key in data:
            s = _extract_text_deep(data.get(key))
            if s:
                return s

    return ""


def _normalize_prompt_for_attachments(original_prompt: str, resources: List[Dict[str, Any]]) -> str:
    if not resources:
        return original_prompt

    lines = ["\n\nInput files attached to this task:"]
    for r in resources:
        rel = r.get("rel_path") or r.get("path")
        rtype = r.get("resource_type")

        fname = r.get("file_name") or r.get("filename")
        if isinstance(rel, str) and rel and isinstance(fname, str) and fname and rel != fname:
            lines.append(f"- {rel} ({rtype}) attached as: {fname}")
        elif isinstance(rel, str) and rel:
            lines.append(f"- {rel} ({rtype})")
        elif isinstance(fname, str) and fname:
            lines.append(f"- {fname} ({rtype})")
        else:
            lines.append(f"- (unknown) ({rtype})")

    return original_prompt.rstrip() + "\n" + "\n".join(lines)


def _safe_model_dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    return obj


def _extract_usage(resp_dump: Any) -> Dict[str, Any]:
    if isinstance(resp_dump, dict) and isinstance(resp_dump.get("usage"), dict):
        return resp_dump["usage"]
    return {}


def _walk(obj: Any) -> Iterable[Any]:
    stack = [obj]
    while stack:
        cur = stack.pop()
        yield cur
        if isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)


def _extract_output_files(resp_dump: Any) -> List[Dict[str, Any]]:
    """Best-effort extraction of provider output file descriptors."""

    found: List[Dict[str, Any]] = []

    for node in _walk(resp_dump):
        if not isinstance(node, dict):
            continue

        # Manus-style
        if "fileUrl" in node and ("fileName" in node or "filename" in node):
            found.append(
                {
                    "file_url": node.get("fileUrl"),
                    "file_name": node.get("fileName") or node.get("filename"),
                    "mime_type": node.get("mimeType") or node.get("mime_type"),
                }
            )
            continue

        # OpenAI-style content block: {"type":"output_file", "file_id":...}
        if node.get("type") in {"output_file", "file"}:
            file_id = node.get("file_id") or node.get("fileId") or node.get("id")
            file_name = node.get("filename") or node.get("file_name")
            if file_id:
                found.append({"file_id": str(file_id), "file_name": file_name})

    # Deduplicate (file_url or file_id)
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for f in found:
        key = f.get("file_url") or f.get("file_id")
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(f)

    return uniq


def _sanitize_filename(name: str) -> str:
    name = name.strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name or "artifact"


def _download_url(
    url: str,
    out_path: Path,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 120.0,
    proxies: Optional[Dict[str, str]] = None,
) -> None:
    _ensure_parent_dir(out_path)
    with requests.get(url, headers=headers or {}, stream=True, timeout=timeout, proxies=proxies) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)


def _write_openai_file_content(content_obj: Any, out_path: Path) -> None:
    _ensure_parent_dir(out_path)

    # openai>=1: HttpxBinaryResponseContent has helper methods; keep defensive.
    if hasattr(content_obj, "write_to_file"):
        content_obj.write_to_file(str(out_path))
        return

    if hasattr(content_obj, "read"):
        data = content_obj.read()
        out_path.write_bytes(data)
        return

    if hasattr(content_obj, "iter_bytes"):
        with open(out_path, "wb") as f:
            for chunk in content_obj.iter_bytes():
                if chunk:
                    f.write(chunk)
        return

    if isinstance(content_obj, (bytes, bytearray)):
        out_path.write_bytes(bytes(content_obj))
        return

    raise TypeError(f"Unsupported file content object: {type(content_obj)}")


@dataclass
class TaskInput:
    task_id: int
    prompt: str
    full_tree: Any
    resources: List[Dict[str, Any]]


def load_tasks(dataset_end_json: Path, task_ids: List[int]) -> Tuple[Path, List[TaskInput]]:
    dataset_end_json = dataset_end_json.resolve()
    dataset_root = dataset_end_json.parent
    data = _read_json(dataset_end_json)

    tasks: List[TaskInput] = []
    for tid in task_ids:
        item = data.get(str(tid))
        if item is None:
            raise KeyError(f"task_id {tid} not found in {dataset_end_json}")

        dialogs = item.get("dialogs") or []
        if not dialogs or not isinstance(dialogs, list):
            raise ValueError(f"task_id {tid} has invalid dialogs")

        prompt = str(dialogs[0].get("content") or "")
        resources_raw = dialogs[0].get("resources") or []
        full_tree = item.get("sub_tasks")

        resources: List[Dict[str, Any]] = []
        for r in resources_raw:
            if not isinstance(r, dict):
                continue
            rel_path = r.get("path")
            if not rel_path:
                continue
            abs_path = (dataset_root / rel_path).resolve()
            resources.append(
                {
                    "declared_type": r.get("type"),
                    "rel_path": rel_path,
                    "path": str(abs_path),
                    "url": r.get("url"),
                    "resource_type": _infer_resource_type_from_path(abs_path),
                    "exists": abs_path.exists(),
                }
            )

        tasks.append(TaskInput(task_id=tid, prompt=prompt, full_tree=full_tree, resources=resources))

    return dataset_root, tasks


class ManusBackend:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        agent_profile: str,
        task_mode: str,
        proxy: Optional[str] = None,
        model: Optional[str] = None,
        poll_interval: float = 1.0,
        timeout_sec: float = 1800.0,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.agent_profile = agent_profile
        self.task_mode = task_mode
        self.model = model
        self.poll_interval = poll_interval
        self.timeout_sec = timeout_sec

        # Manus Tasks API (async): POST <api_base>/tasks then poll GET <api_base>/tasks/{task_id}
        self.api_base = _normalize_manus_api_base_url(base_url)
        self.create_url = f"{self.api_base}/tasks"
        self.get_url_tmpl = f"{self.api_base}/tasks/{{task_id}}"

        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "API_KEY": str(api_key),
        }

        # Manus Files API: create file record (POST /v1/files) then PUT to a presigned upload_url.
        self.files_url = f"{self.api_base}/files"

        # Cache file uploads by a stable fingerprint to avoid re-uploading.
        self._upload_cache: Dict[str, Dict[str, Any]] = {}

        proxy_url = _resolve_proxy_url(proxy)
        if proxy_url:
            self.proxies = {"http": proxy_url, "https": proxy_url}
        else:
            self.proxies = None

        # Optional OpenAI client for best-effort file_id downloads (if Manus ever returns them).
        self.openai_client = None
        if OpenAI is not None:
            kwargs: Dict[str, Any] = {
                "api_key": "EMPTY",
                "base_url": self.api_base,
                "default_headers": {"API_KEY": api_key},
            }
            if proxy_url and httpx is not None:
                try:
                    try:
                        kwargs["http_client"] = httpx.Client(proxy=proxy_url)
                    except TypeError:
                        kwargs["http_client"] = httpx.Client(proxies=proxy_url)
                except Exception:
                    pass
            try:
                self.openai_client = OpenAI(**kwargs)
            except Exception:
                self.openai_client = None

    def _poll_task(self, task_id: str, debug: bool) -> Tuple[Any, str]:
        url = self.get_url_tmpl.format(task_id=task_id)
        deadline = time.monotonic() + float(self.timeout_sec)
        last: Any = None
        last_text: str = ""

        while time.monotonic() < deadline:
            try:
                resp = requests.get(url, headers=self.headers, timeout=60, proxies=self.proxies)
            except requests.exceptions.RequestException as e:
                last = f"{type(e).__name__}: {e}"
                if debug:
                    print(f"[manus] poll transient exception: {last}")
                time.sleep(self.poll_interval)
                continue

            try:
                data = resp.json()
            except Exception:
                data = {"_non_json": True, "text": resp.text}

            # Manus may return 404 briefly right after create (eventual consistency).
            if resp.status_code == 404:
                msg = ""
                if isinstance(data, dict) and isinstance(data.get("message"), str):
                    msg = data.get("message")
                if "task not found" in (msg or "").lower():
                    last = data
                    if debug:
                        print("[manus] poll got 404 task not found; retrying")
                    time.sleep(self.poll_interval)
                    continue

            # Retry on transient server errors / rate limits.
            if resp.status_code in {429, 500, 502, 503, 504}:
                last = data
                if debug:
                    print(f"[manus] poll got {resp.status_code}; retrying")
                time.sleep(self.poll_interval)
                continue

            if resp.status_code >= 400:
                raise RuntimeError(f"Manus poll failed: {resp.status_code} {data}")

            last = data
            out = _extract_manus_output_text(data)
            if out:
                last_text = out

            status = (_extract_manus_status(data) or "").lower()
            if status in {"completed", "succeeded", "success", "finished", "done"}:
                return data, last_text
            if status in {"failed", "error", "cancelled", "canceled"}:
                raise RuntimeError(f"Manus task ended with status={status}: {data}")

            time.sleep(self.poll_interval)

        raise TimeoutError(f"timeout waiting Manus task_id={task_id}; last={last}")

    def _create_task_with_retry(self, payload: Dict[str, Any], debug: bool) -> Any:
        """Create a Manus task with backoff on transient errors (429/5xx/network)."""

        max_retries = int(os.getenv("MANUS_CREATE_MAX_RETRIES", "8"))
        base_sleep = float(os.getenv("MANUS_CREATE_BACKOFF_SEC", "6"))
        max_sleep = float(os.getenv("MANUS_CREATE_BACKOFF_MAX_SEC", "60"))

        last_err: Optional[str] = None

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(
                    self.create_url,
                    json=payload,
                    headers=self.headers,
                    timeout=60,
                    proxies=self.proxies,
                )
            except requests.exceptions.RequestException as e:
                last_err = f"{type(e).__name__}: {e}"
                if debug:
                    print(f"[manus] create_task exception attempt={attempt}/{max_retries}: {last_err}")
                sleep_s = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
                time.sleep(sleep_s)
                continue

            try:
                data = resp.json()
            except Exception:
                data = {"_non_json": True, "text": resp.text}

            if resp.status_code in {429, 500, 502, 503, 504}:
                retry_after = resp.headers.get("Retry-After")
                sleep_s: Optional[float] = None
                if retry_after:
                    try:
                        sleep_s = float(retry_after)
                    except Exception:
                        sleep_s = None
                if sleep_s is None:
                    sleep_s = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
                last_err = f"HTTP {resp.status_code}: {data}"
                if debug:
                    print(
                        f"[manus] create_task transient {resp.status_code} attempt={attempt}/{max_retries}; "
                        f"sleep {sleep_s:.1f}s"
                    )
                time.sleep(float(sleep_s))
                continue

            if resp.status_code >= 400:
                raise ManusHTTPError(
                    f"Manus create_task failed: {resp.status_code} {data} "
                    f"(check MANUS_BASE_URL; expected like https://api.manus.ai/v1)",
                    status_code=resp.status_code,
                    data=data,
                )

            return data

        raise RuntimeError(f"Manus create_task failed after retries: {last_err}")

    def _file_fingerprint(self, path: str) -> str:
        try:
            st = os.stat(path)
            return f"{path}|{int(st.st_size)}|{int(st.st_mtime)}"
        except Exception:
            return f"{path}|unknown"

    def _upload_input_file(self, path: str, debug: bool) -> Optional[Dict[str, Any]]:
        """Upload a local file to Manus (best-effort) and return descriptor.

        Official flow (per Manus docs):
        1) POST /v1/files {"filename": "..."} -> {id, upload_url, ...}
        2) PUT <upload_url> with raw file bytes
        3) (optional) poll GET /v1/files/{id} until status == uploaded
        """

        fp = self._file_fingerprint(path)
        cached = self._upload_cache.get(fp)
        if cached and (cached.get("file_id") or cached.get("file_url")):
            return cached

        suffix = Path(path).suffix.lower()
        if suffix and suffix not in SUPPORTED_INPUT_EXT:
            return None

        filename = Path(path).name
        create_payload = {"filename": filename}

        try:
            if debug:
                print(f"[manus] create_file: url={self.files_url} filename={filename}")
            resp = requests.post(
                self.files_url,
                json=create_payload,
                headers=self.headers,
                timeout=60,
                proxies=self.proxies,
            )
            try:
                data = resp.json()
            except Exception:
                data = {"_non_json": True, "text": resp.text}

            if resp.status_code >= 400:
                if debug:
                    print(f"[manus] create_file failed: {resp.status_code} {data}")
                return None

            if not isinstance(data, dict):
                if debug:
                    print(f"[manus] create_file unexpected response: {type(data)}")
                return None

            file_id = _extract_file_id(data) or (data.get("id") if isinstance(data.get("id"), str) else None)
            upload_url = data.get("upload_url") or data.get("uploadUrl")
            if not isinstance(upload_url, str) or not upload_url.strip():
                if debug:
                    print(f"[manus] create_file missing upload_url: {data}")
                return None
            if not isinstance(file_id, str) or not file_id.strip():
                if debug:
                    print(f"[manus] create_file missing id: {data}")
                return None

            if debug:
                print(f"[manus] upload PUT -> presigned_url (id={file_id})")
            with open(path, "rb") as f:
                put = requests.put(
                    upload_url,
                    data=f,
                    timeout=600,
                    proxies=self.proxies,
                )

            if put.status_code >= 400:
                if debug:
                    print(
                        f"[manus] upload PUT failed: {put.status_code} "
                        f"{(put.text[:2000] if isinstance(put.text, str) else put.text)}"
                    )
                return None

            # Best-effort: wait until file status becomes 'uploaded'.
            wait_s = float(os.getenv("MANUS_FILE_UPLOAD_WAIT_SEC", "20"))
            poll_s = float(os.getenv("MANUS_FILE_UPLOAD_POLL_SEC", "1.0"))
            if wait_s > 0:
                deadline = time.monotonic() + wait_s
                while time.monotonic() < deadline:
                    meta_url = f"{self.files_url}/{file_id}"
                    try:
                        mresp = requests.get(meta_url, headers=self.headers, timeout=30, proxies=self.proxies)
                        mdata = mresp.json() if mresp.headers.get("content-type", "").startswith("application/json") else None
                        if isinstance(mdata, dict):
                            st = (mdata.get("status") or "").lower()
                            if st == "uploaded":
                                break
                            if st == "deleted":
                                if debug:
                                    print(f"[manus] uploaded file unexpectedly deleted: {mdata}")
                                break
                    except Exception:
                        pass
                    time.sleep(poll_s)

            desc = {"file_id": str(file_id), "file_name": filename, "path": path}
            self._upload_cache[fp] = desc
            return desc

        except Exception as e:
            if debug:
                print(f"[manus] upload exception: {type(e).__name__}: {e}")
            return None

    def _download_file_id(self, file_id: str, out_path: Path, debug: bool) -> None:
        """Download an OpenAI-style file_id to disk without requiring the OpenAI SDK."""

        url = f"{self.api_base}/files/{file_id}/content"
        headers = {k: v for k, v in self.headers.items() if k.lower() != "content-type"}
        try:
            _download_url(url, out_path, headers=headers, timeout=600, proxies=self.proxies)
        except Exception as e:
            if debug:
                print(f"[manus] download file_id via requests failed: {type(e).__name__}: {e}")
            raise

    def run_one(
        self,
        task: TaskInput,
        artifacts_dir: Path,
        dry_run: bool,
        debug: bool,
    ) -> Dict[str, Any]:
        """Return a single task result in eval-pack task schema."""

        task_started_utc = _utc_iso_ts()
        start = time.monotonic()

        prompt = _normalize_prompt_for_attachments(
            task.prompt,
            [
                {
                    "rel_path": r.get("rel_path"),
                    "resource_type": r.get("resource_type"),
                    "file_name": (Path(str(r.get("rel_path") or r.get("path") or "")).name or None),
                }
                for r in task.resources
            ],
        )

        upload_errors: List[str] = []
        uploaded_inputs: List[Dict[str, Any]] = []

        output_text = "(dry-run)" if dry_run else ""
        resp_dump: Any = None
        usage: Dict[str, Any] = {}
        out_files: List[str] = []
        status: Optional[str] = None
        err: Optional[str] = None
        manus_task_id: Optional[str] = None
        call_started_utc: Optional[str] = None
        call_finished_utc: Optional[str] = None
        call_elapsed_sec: Optional[float] = None

        if not dry_run:
            call_started_utc = _utc_iso_ts()
            call_start = time.monotonic()
            try:
                # If there are input resources, best-effort upload then attach
                # via Manus CreateTask `attachments` (file_id/url/base64).
                attachments: List[Dict[str, Any]] = []

                if task.resources:
                    for r in task.resources:
                        r_path = r.get("path")
                        r_rel = r.get("rel_path")
                        r_url = r.get("url")
                        rtype = r.get("resource_type")

                        attach_name = None
                        if isinstance(r_rel, str) and r_rel.strip():
                            attach_name = Path(r_rel).name
                        if not attach_name and isinstance(r_path, str) and r_path:
                            attach_name = Path(r_path).name
                        if not attach_name and isinstance(r_url, str) and r_url.strip():
                            attach_name = Path(urlparse(r_url.strip()).path).name or "attachment"
                        attach_name = attach_name or "attachment"

                        if isinstance(r_url, str) and r_url.strip():
                            # If dataset provides a URL already, try to reference it.
                            url = r_url.strip()
                            uploaded_inputs.append({"rel_path": r_rel, "resource_type": rtype, "file_url": url})
                            att: Dict[str, Any] = {"filename": attach_name, "url": url}
                            mt = _guess_mime_type(attach_name)
                            if mt and mt != "application/octet-stream":
                                att["mimeType"] = mt
                            attachments.append(att)
                            continue

                        if not isinstance(r_path, str) or not r_path:
                            continue

                        if not Path(r_path).exists():
                            upload_errors.append(f"missing_input_file: {r_rel or r_path}")
                            continue

                        if Path(r_path).suffix.lower() not in SUPPORTED_INPUT_EXT:
                            upload_errors.append(f"unsupported_input_ext: {r_rel or r_path}")
                            continue

                        desc = self._upload_input_file(r_path, debug=debug)
                        if not desc:
                            upload_errors.append(f"upload_failed: {r_rel or r_path}")
                            continue

                        uploaded_inputs.append({"rel_path": r_rel, "resource_type": rtype, **{k: v for k, v in desc.items() if k in {"file_id", "file_url", "file_name"}}})

                        if desc.get("file_id"):
                            attachments.append({"filename": attach_name, "file_id": str(desc["file_id"])})
                        elif desc.get("file_url"):
                            attachments.append({"filename": attach_name, "url": str(desc["file_url"])})

                # Build payload for Manus CreateTask.
                payload: Dict[str, Any] = {
                    "prompt": prompt,
                    "agentProfile": self.agent_profile,
                }
                if attachments:
                    payload["attachments"] = attachments
                if self.task_mode:
                    payload["taskMode"] = self.task_mode
                # Note: `model` is not documented for CreateTask; keep it out to avoid schema rejection.

                if debug:
                    print(f"[manus] create_task | api_base={self.api_base} url={self.create_url} task_id={task.task_id}")

                create_json = self._create_task_with_retry(payload, debug=debug)

                immediate = _extract_manus_output_text(create_json)
                if immediate:
                    resp_dump = create_json
                    output_text = immediate
                else:
                    manus_task_id = _extract_manus_task_id(create_json)
                    if not manus_task_id:
                        raise RuntimeError(f"Manus create_task response missing task id/output: {create_json}")
                    resp_dump, last_text = self._poll_task(manus_task_id, debug=debug)
                    output_text = _extract_manus_output_text(resp_dump) or (last_text or "")

                usage = _extract_usage(resp_dump)
                status = _extract_manus_status(resp_dump)

                # Download output files if any.
                file_descs = _extract_output_files(resp_dump)
                if debug:
                    print(f"[manus] output_files_detected={len(file_descs)}")

                for j, fdesc in enumerate(file_descs):
                    task_dir = artifacts_dir / f"task_{task.task_id}"
                    task_dir.mkdir(parents=True, exist_ok=True)

                    if fdesc.get("file_url"):
                        fname = _sanitize_filename(str(fdesc.get("file_name") or f"artifact_{j}"))
                        out_path = task_dir / fname
                        try:
                            # Some downloads are already pre-signed; try with header then fallback.
                            try:
                                _download_url(
                                    str(fdesc["file_url"]),
                                    out_path,
                                    headers={"API_KEY": self.api_key},
                                    proxies=self.proxies,
                                )
                            except Exception:
                                _download_url(str(fdesc["file_url"]), out_path, headers=None, proxies=self.proxies)
                            out_files.append(_relpath_from_repo(out_path))
                        except Exception as e:
                            if debug:
                                print(f"[manus] download url failed: {e}")

                    elif fdesc.get("file_id") and self.openai_client is not None:
                        file_id = str(fdesc["file_id"])
                        fname = _sanitize_filename(str(fdesc.get("file_name") or f"{file_id}"))
                        out_path = task_dir / fname
                        try:
                            content = self.openai_client.files.content(file_id)
                            _write_openai_file_content(content, out_path)
                            out_files.append(_relpath_from_repo(out_path))
                        except Exception as e:
                            if debug:
                                print(f"[manus] download file_id failed: {e}")

                    elif fdesc.get("file_id"):
                        # Fallback when `openai` SDK is not installed.
                        file_id = str(fdesc["file_id"])
                        fname = _sanitize_filename(str(fdesc.get("file_name") or f"{file_id}"))
                        out_path = task_dir / fname
                        try:
                            self._download_file_id(file_id, out_path, debug=debug)
                            out_files.append(_relpath_from_repo(out_path))
                        except Exception:
                            pass

            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            finally:
                call_elapsed_sec = time.monotonic() - call_start
                call_finished_utc = _utc_iso_ts()

        elapsed = time.monotonic() - start
        task_finished_utc = _utc_iso_ts()

        final_text = (output_text or "").strip()
        if not final_text:
            if err:
                final_text = f"(error) {err}"
            elif out_files:
                final_text = "Produced files:\n" + "\n".join([f"- {p}" for p in out_files])

        trace = [{"role": "assistant", "content": final_text}]

        return {
            "task_id": int(task.task_id),
            "sample_idx": None,
            "origin_prompt": [task.prompt],
            "assistant_outputs": [trace],
            "attached_files": out_files,
            "full_tree": task.full_tree,
            "meta": {
                "backend": "manus",
                "dry_run": dry_run,
                "elapsed_sec": elapsed,
                "call_elapsed_sec": call_elapsed_sec,
                "call_started_utc": call_started_utc,
                "call_finished_utc": call_finished_utc,
                "task_started_utc": task_started_utc,
                "task_finished_utc": task_finished_utc,
                "status": status,
                "usage": usage,
                "upload_errors": upload_errors,
                "uploaded_inputs": uploaded_inputs,
                "error": err,
                "manus_task_id": manus_task_id,
                "manus_api_base": self.api_base,
            },
        }


class OpenClawBackend:
    """Scaffold only.

    OpenClaw might be a local agent framework (not an HTTP API). Without a
    stable invocation contract we cannot implement it safely.

    For now:
    - dry-run works (placeholder output)
    - real runs raise a clear error
    """

    def run_one(self, task: TaskInput, dry_run: bool) -> Dict[str, Any]:
        task_started_utc = _utc_iso_ts()
        start = time.monotonic()
        if not dry_run:
            raise NotImplementedError(
                "OpenClaw backend is not implemented yet. Please provide either "
                "(1) an OpenAI-compatible endpoint for OpenClaw, or (2) a CLI/python entrypoint "
                "contract so we can call it per-task and capture outputs/files."
            )

        trace = [
            {
                "role": "assistant",
                "content": "(dry-run) OpenClaw backend placeholder; no request sent.",
            }
        ]

        elapsed = time.monotonic() - start
        task_finished_utc = _utc_iso_ts()

        return {
            "task_id": int(task.task_id),
            "sample_idx": None,
            "origin_prompt": [task.prompt],
            "assistant_outputs": [trace],
            "attached_files": [],
            "full_tree": task.full_tree,
            "meta": {
                "backend": "openclaw",
                "dry_run": True,
                "elapsed_sec": elapsed,
                "task_started_utc": task_started_utc,
                "task_finished_utc": task_finished_utc,
            },
        }


def build_eval_pack(tasks: List[Dict[str, Any]], generated_from: Dict[str, Any]) -> Dict[str, Any]:
    # Assign sample_idx deterministically
    for i, t in enumerate(tasks, start=1):
        t["sample_idx"] = i
    return {"schema_version": 1, "generated_from": generated_from, "tasks": tasks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-end-json",
        default="opencompass/data/gta_dataset_v2/end.json",
        help="Path to GTA-Workflow (GTA-2) end.json",
    )
    ap.add_argument(
        "--task-ids",
        default="analysis/human_eval_gpt5_stratified_mini_20260312_pack/selected_task_ids.txt",
        help="Path to selected_task_ids.txt",
    )
    ap.add_argument(
        "--backend",
        choices=["manus", "openclaw"],
        required=True,
        help="Which backend to run",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output eval-pack JSON path",
    )
    ap.add_argument(
        "--artifacts-dir",
        default=None,
        help="Directory to save downloaded output files",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--debug", action="store_true")

    # Manus config
    ap.add_argument("--manus-base-url", default=os.getenv("MANUS_BASE_URL", "https://api.manus.ai/v1"))
    ap.add_argument("--manus-agent-profile", default=os.getenv("MANUS_AGENT_PROFILE", "manus-1.6"))
    ap.add_argument("--manus-task-mode", default=os.getenv("MANUS_TASK_MODE", "agent"))
    ap.add_argument("--manus-model", default=os.getenv("MANUS_MODEL") or None)
    ap.add_argument("--manus-poll-interval", type=float, default=float(os.getenv("MANUS_POLL_INTERVAL", "1.0")))
    ap.add_argument("--manus-timeout-sec", type=float, default=float(os.getenv("MANUS_TIMEOUT_SEC", "1800")))
    ap.add_argument("--proxy", default=None, help="Optional proxy override; otherwise uses EVAL_PROXY/HTTP(S)_PROXY")

    args = ap.parse_args()

    dataset_end_json = Path(args.dataset_end_json)
    task_ids_file = Path(args.task_ids)

    task_ids = _read_task_ids(task_ids_file)
    dataset_root, task_inputs = load_tasks(dataset_end_json, task_ids)

    out_path = Path(args.out) if args.out else Path("agent_app_eval/runs") / f"{args.backend}_{_now_ts()}_eval_pack.json"
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else Path("agent_app_eval/runs") / "artifacts" / args.backend

    tasks_out: List[Dict[str, Any]] = []

    run_started_utc = _utc_iso_ts()
    run_start = time.monotonic()

    if args.backend == "manus":
        manus_key = os.getenv("MANUS_API_KEY")
        if not manus_key and not args.dry_run:
            raise RuntimeError("MANUS_API_KEY is required for non-dry-run Manus calls")

        backend = ManusBackend(
            api_key=manus_key or "",
            base_url=args.manus_base_url,
            agent_profile=args.manus_agent_profile,
            task_mode=args.manus_task_mode,
            proxy=args.proxy,
            model=args.manus_model,
            poll_interval=args.manus_poll_interval,
            timeout_sec=args.manus_timeout_sec,
        )

        for t in task_inputs:
            if args.debug:
                print(f"\n=== manus task {t.task_id} ===")
                for r in t.resources:
                    print(f"  - resource {r.get('rel_path')} exists={r.get('exists')}")
            tasks_out.append(backend.run_one(t, artifacts_dir=artifacts_dir, dry_run=args.dry_run, debug=args.debug))

    elif args.backend == "openclaw":
        backend = OpenClawBackend()
        for t in task_inputs:
            if args.debug:
                print(f"\n=== openclaw task {t.task_id} ===")
            tasks_out.append(backend.run_one(t, dry_run=args.dry_run))

    run_elapsed_sec = time.monotonic() - run_start
    run_finished_utc = _utc_iso_ts()

    sum_task_elapsed_sec = 0.0
    sum_call_elapsed_sec = 0.0
    for t in tasks_out:
        m = t.get("meta") or {}
        try:
            if m.get("elapsed_sec") is not None:
                sum_task_elapsed_sec += float(m.get("elapsed_sec"))
        except Exception:
            pass
        try:
            if m.get("call_elapsed_sec") is not None:
                sum_call_elapsed_sec += float(m.get("call_elapsed_sec"))
        except Exception:
            pass

    generated_from = {
        "script": _relpath_from_repo(Path(__file__)),
        "backend": args.backend,
        "dataset_end_json": _relpath_from_repo(dataset_end_json.resolve()),
        "task_ids": _relpath_from_repo(task_ids_file.resolve()),
        "dataset_root": _relpath_from_repo(dataset_root),
        "dry_run": bool(args.dry_run),
        "run_started_utc": run_started_utc,
        "run_finished_utc": run_finished_utc,
        "total_elapsed_sec": run_elapsed_sec,
        "num_tasks": len(task_inputs),
        "sum_task_elapsed_sec": sum_task_elapsed_sec,
        "sum_call_elapsed_sec": sum_call_elapsed_sec,
    }

    pack = build_eval_pack(tasks_out, generated_from=generated_from)

    _ensure_parent_dir(out_path)
    out_path.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[OK] wrote eval pack: {out_path}")


if __name__ == "__main__":
    main()
