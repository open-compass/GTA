"""Build a GPT-5.2-scoreable eval-pack from existing OpenClaw results.

This script consolidates outputs from:
- agent_app_eval/openclaw/openclaw_result/<task_id>/ (deliverables)
- Optional: agent_app_eval/openclaw/openclaw_execution_log.csv (timing/status)

It produces:
- An eval-pack JSON compatible with agent_app_eval/score_with_gpt52.py
- Optional converted/proxy artifacts for evaluator-friendly formats:
  - .docx/.pptx/.xlsx -> PDF via LibreOffice
  - .csv/.zip/code-like files -> .txt proxy files

Rationale:
The GTA v2 GPTEvaluator can only directly upload a limited set of file types.
These conversions help ensure OpenClaw deliverables are visible to the judge.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# File types that the GPT evaluator can directly upload.
_DIRECT_OK_EXT = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".mp3",
    ".wav",
    ".m4a",
    ".mp4",
    ".mov",
    ".webm",
}

# File types that the evaluator will convert to PDF using wkhtmltopdf.
_CONVERTIBLE_OK_EXT = {".html", ".md", ".txt", ".json"}

# Office formats we convert to PDF (LibreOffice).
_OFFICE_EXT = {".docx", ".pptx", ".xlsx"}

# Common text-like outputs we can proxy to .txt so the evaluator can upload.
_TEXT_PROXY_EXT = {
    ".csv",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".cs",
    ".unity",
    ".asset",
    ".css",
    ".mmd",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".toml",
    ".sh",
    ".bat",
    ".ps1",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".tex",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _relpath_from_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_repo_root()))
    except Exception:
        return str(path)


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _parse_iso_utc(ts: Optional[str]) -> Optional[datetime]:
    if not ts or not isinstance(ts, str):
        return None
    try:
        # Expect format: 2026-03-16T09:20:53Z
        if ts.endswith("Z"):
            return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        # Fallback: allow offset-less ISO
        return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
    except Exception:
        return None


@dataclass
class DatasetTask:
    task_id: int
    prompt: str
    full_tree: Any


def load_dataset_tasks(dataset_end_json: Path, task_ids: List[int]) -> Dict[int, DatasetTask]:
    data = _read_json(dataset_end_json)
    out: Dict[int, DatasetTask] = {}

    for tid in task_ids:
        item = data.get(str(tid))
        if item is None:
            raise KeyError(f"task_id {tid} not found in {dataset_end_json}")

        dialogs = item.get("dialogs") or []
        if not dialogs or not isinstance(dialogs, list):
            raise ValueError(f"task_id {tid} has invalid dialogs")

        prompt = str(dialogs[0].get("content") or "")
        full_tree = item.get("sub_tasks")

        out[tid] = DatasetTask(task_id=tid, prompt=prompt, full_tree=full_tree)

    return out


_SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".idea",
    ".vscode",
    "node_modules",
}


def _list_files_recursive(root: Path) -> List[Path]:
    """List all files under root (recursively) with a stable order."""

    if not root.exists():
        return []
    if root.is_file():
        return [root]

    files: List[Path] = []

    for p in sorted(root.rglob("*")):
        try:
            if p.is_dir() and p.name in _SKIP_DIR_NAMES:
                # rglob will still walk into it; we filter on files below.
                continue
        except Exception:
            continue

        try:
            if p.is_file():
                # Skip files inside ignored dirs.
                parts = set(p.parts)
                if parts & _SKIP_DIR_NAMES:
                    continue
                files.append(p)
        except Exception:
            continue

    return files


def _libreoffice_convert_to_pdf(src: Path, out_dir: Path) -> Optional[Path]:
    """Convert an Office file to PDF using LibreOffice.

    Returns output PDF path if created, else None.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / (src.stem + ".pdf")

    try:
        if out_pdf.exists() and out_pdf.stat().st_mtime >= src.stat().st_mtime:
            return out_pdf
    except Exception:
        pass

    cmd = [
        "libreoffice",
        "--headless",
        "--nologo",
        "--nofirststartwizard",
        "--convert-to",
        "pdf",
        "--outdir",
        str(out_dir),
        str(src),
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
    except Exception:
        return None

    return out_pdf if out_pdf.exists() else None


def _copy_bytes_to_txt(src: Path, out_dir: Path, out_name: str) -> Optional[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / out_name

    try:
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            return dst
    except Exception:
        pass

    try:
        dst.write_bytes(src.read_bytes())
        return dst
    except Exception:
        return None


def _zip_contents_txt(zip_path: Path, out_dir: Path) -> Optional[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / (zip_path.name + ".contents.txt")

    try:
        if dst.exists() and dst.stat().st_mtime >= zip_path.stat().st_mtime:
            return dst
    except Exception:
        pass

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            infos = z.infolist()
        lines = [f"zip: {zip_path.name}", f"num_files: {len(infos)}", "", "files:"]
        for info in infos:
            lines.append(f"- {info.filename} ({info.file_size} bytes)")
        dst.write_text("\n".join(lines), encoding="utf-8")
        return dst
    except Exception:
        return None


def _load_openclaw_execution_log(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}

    out: Dict[int, Dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                try:
                    tid = int(row.get("task_id") or "")
                except Exception:
                    continue

                rec: Dict[str, Any] = {
                    "task_id": tid,
                    "start_time": row.get("start_time"),
                    "end_time": row.get("end_time"),
                    "status": row.get("status"),
                }

                dur = row.get("duration_seconds")
                try:
                    rec["duration_seconds"] = float(dur) if dur is not None and str(dur).strip() else None
                except Exception:
                    rec["duration_seconds"] = None

                out[tid] = rec
    except Exception:
        return {}

    return out


def _build_task_attachments(
    task_id: int,
    openclaw_result_dir: Path,
    out_artifacts_dir: Path,
) -> Tuple[List[str], Dict[str, Any]]:
    """Return (attached_files_relpaths, debug_info)."""

    debug: Dict[str, Any] = {
        "sources": [],
        "converted": [],
        "proxied": [],
        "skipped": [],
    }

    sources: List[Path] = []
    task_dir = openclaw_result_dir / str(task_id)
    if task_dir.exists():
        sources.append(task_dir)

    debug["sources"] = [_relpath_from_repo(p) for p in sources]

    all_files: List[Path] = []
    for d in sources:
        all_files.extend(_list_files_recursive(d))

    attached: List[Path] = []

    # Pre-index existing PDFs in sources to avoid unnecessary conversions.
    pdf_by_stem = {p.stem: p for p in all_files if p.suffix.lower() == ".pdf"}

    task_out_dir = out_artifacts_dir / f"task_{task_id}"

    for p in all_files:
        name = p.name
        if name in {".DS_Store"}:
            continue
        if name.startswith("._"):
            continue

        ext = p.suffix.lower()

        if ext in _DIRECT_OK_EXT or ext in _CONVERTIBLE_OK_EXT:
            attached.append(p)
            continue

        if ext in _OFFICE_EXT:
            # If a PDF with same stem exists already, prefer it.
            if p.stem in pdf_by_stem:
                attached.append(pdf_by_stem[p.stem])
                continue
            out_pdf = _libreoffice_convert_to_pdf(p, task_out_dir)
            if out_pdf is not None:
                debug["converted"].append({"src": _relpath_from_repo(p), "out": _relpath_from_repo(out_pdf)})
                attached.append(out_pdf)
            else:
                debug["skipped"].append({"file": _relpath_from_repo(p), "reason": "office_convert_failed"})
            continue

        if ext == ".zip":
            out_txt = _zip_contents_txt(p, task_out_dir)
            if out_txt is not None:
                debug["proxied"].append({"src": _relpath_from_repo(p), "out": _relpath_from_repo(out_txt), "kind": "zip_contents"})
                attached.append(out_txt)
            else:
                debug["skipped"].append({"file": _relpath_from_repo(p), "reason": "zip_read_failed"})
            continue

        if ext in _TEXT_PROXY_EXT:
            out_name = p.name + ".txt" if ext != ".txt" else p.name
            out_txt = _copy_bytes_to_txt(p, task_out_dir, out_name=out_name)
            if out_txt is not None:
                debug["proxied"].append({"src": _relpath_from_repo(p), "out": _relpath_from_repo(out_txt), "kind": "text_copy"})
                attached.append(out_txt)
            else:
                debug["skipped"].append({"file": _relpath_from_repo(p), "reason": "text_proxy_failed"})
            continue

        debug["skipped"].append({"file": _relpath_from_repo(p), "reason": f"unsupported_ext:{ext}"})

    # Dedup + stable sort
    uniq: Dict[str, Path] = {}
    for p in attached:
        uniq[str(p.resolve())] = p

    attached_rel = sorted({_relpath_from_repo(p) for p in uniq.values()})

    return attached_rel, debug


def build_eval_pack(tasks: List[Dict[str, Any]], generated_from: Dict[str, Any]) -> Dict[str, Any]:
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
        "--openclaw-result-dir",
        default="agent_app_eval/openclaw/openclaw_result",
        help="Directory containing <task_id>/ deliverables",
    )
    ap.add_argument(
        "--openclaw-exec-log",
        default="agent_app_eval/openclaw/openclaw_execution_log.csv",
        help="Optional OpenClaw execution log CSV",
    )
    ap.add_argument(
        "--out-pack",
        default=None,
        help="Output eval-pack JSON path",
    )
    ap.add_argument(
        "--out-artifacts-dir",
        default=None,
        help="Directory to store converted/proxy artifacts",
    )

    args = ap.parse_args()

    dataset_end_json = Path(args.dataset_end_json)
    task_ids_file = Path(args.task_ids)
    openclaw_result_dir = Path(args.openclaw_result_dir)
    exec_log_path = Path(args.openclaw_exec_log) if args.openclaw_exec_log else None

    task_ids = _read_task_ids(task_ids_file)
    dataset_tasks = load_dataset_tasks(dataset_end_json, task_ids)

    exec_log = _load_openclaw_execution_log(exec_log_path) if exec_log_path else {}

    out_root = Path("agent_app_eval/gpt52_ready_openclaw")
    out_pack = Path(args.out_pack) if args.out_pack else out_root / f"compiled_openclaw_{_now_ts()}_eval_pack.json"
    out_artifacts_dir = Path(args.out_artifacts_dir) if args.out_artifacts_dir else out_root / "artifacts"

    tasks_out: List[Dict[str, Any]] = []

    for tid in task_ids:
        ds = dataset_tasks.get(tid)
        if ds is None:
            raise KeyError(f"missing dataset task {tid}")

        attached_files, attach_debug = _build_task_attachments(
            task_id=tid,
            openclaw_result_dir=openclaw_result_dir,
            out_artifacts_dir=out_artifacts_dir,
        )

        if attached_files:
            lines = ["Produced files (for evaluation):"]
            for p in attached_files:
                lines.append(f"- {p}")
            final_answer = "\n".join(lines)
        else:
            final_answer = "(no output files found)"

        task_entry: Dict[str, Any] = {
            "task_id": int(tid),
            "sample_idx": None,
            "origin_prompt": [ds.prompt],
            "assistant_outputs": [[{"role": "assistant", "content": final_answer}]],
            "attached_files": attached_files,
            "full_tree": ds.full_tree,
            "meta": {
                "compiled_from": {
                    "openclaw_result_dir": _relpath_from_repo((openclaw_result_dir / str(tid)).resolve())
                    if (openclaw_result_dir / str(tid)).exists()
                    else None,
                    "openclaw_execution_log": _relpath_from_repo(exec_log_path.resolve())
                    if exec_log_path and exec_log_path.exists()
                    else None,
                },
                "execution": exec_log.get(int(tid)),
                "attachments": attach_debug,
            },
        }

        tasks_out.append(task_entry)

    generated_from = {
        "script": _relpath_from_repo(Path(__file__)),
        "created_utc": _utc_now_iso(),
        "dataset_end_json": _relpath_from_repo(dataset_end_json.resolve()),
        "task_ids": _relpath_from_repo(task_ids_file.resolve()),
        "openclaw_result_dir": _relpath_from_repo(openclaw_result_dir.resolve()),
        "openclaw_execution_log": _relpath_from_repo(exec_log_path.resolve()) if exec_log_path and exec_log_path.exists() else None,
        "out_artifacts_dir": _relpath_from_repo(out_artifacts_dir.resolve()),
        "num_tasks": len(tasks_out),
    }

    pack = build_eval_pack(tasks_out, generated_from=generated_from)

    _ensure_parent_dir(out_pack)
    out_pack.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[OK] wrote eval pack: {out_pack}")


if __name__ == "__main__":
    main()
