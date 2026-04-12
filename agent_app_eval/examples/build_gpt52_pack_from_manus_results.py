"""Build a GPT-5.2-scoreable eval-pack from existing Manus results.

This script consolidates outputs from:
- agent_app_eval/runs/*_eval_pack.json (timing/meta, sometimes artifacts)
- agent_app_eval/manus_result/<task_id>/ (final deliverables)

It produces:
- A new eval-pack JSON compatible with agent_app_eval/score_with_gpt52.py
- One timing file per task ("complete time")
- Optional converted artifacts for evaluator-friendly formats:
  - .docx/.pptx/.xlsx -> PDF via LibreOffice
  - .csv/.zip/code-like files -> .txt proxy files

The GPT evaluator only uploads a limited set of file types; these conversions help
ensure deliverables are visible to the judge.
"""

from __future__ import annotations

import argparse
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
}

_SUCCESS_STATUSES = {"completed", "succeeded", "success", "finished", "done"}


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


def _iter_run_packs(runs_dir: Path) -> Iterable[Path]:
    if not runs_dir.exists():
        return []
    for p in sorted(runs_dir.glob("*_eval_pack.json")):
        if p.is_file():
            yield p


def _collect_run_records(runs_dir: Path) -> Dict[int, List[Dict[str, Any]]]:
    """Map task_id -> list of task dicts from all packs (with pack_path)."""

    out: Dict[int, List[Dict[str, Any]]] = {}

    for pack_path in _iter_run_packs(runs_dir):
        try:
            pack = _read_json(pack_path)
        except Exception:
            continue

        tasks = pack.get("tasks")
        if not isinstance(tasks, list):
            continue

        for t in tasks:
            if not isinstance(t, dict):
                continue
            try:
                tid = int(t.get("task_id"))
            except Exception:
                continue
            rec = dict(t)
            rec["_pack_path"] = _relpath_from_repo(pack_path)
            out.setdefault(tid, []).append(rec)

    return out


def _score_record_for_timing(rec: Dict[str, Any]) -> Tuple[int, float]:
    """Higher is better. Returns (rank, duration_guess)."""

    meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
    status = str(meta.get("status") or "").lower()
    err = meta.get("error")

    call_elapsed = meta.get("call_elapsed_sec")
    elapsed = meta.get("elapsed_sec")

    dur = 0.0
    for v in (call_elapsed, elapsed):
        if isinstance(v, (int, float)) and v and v > dur:
            dur = float(v)

    rank = 0
    if status in _SUCCESS_STATUSES and not err:
        rank = 3
    elif status in _SUCCESS_STATUSES:
        rank = 2
    elif isinstance(call_elapsed, (int, float)) and call_elapsed:
        rank = 1

    return rank, dur


def select_best_timing_record(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not records:
        return None

    best: Optional[Dict[str, Any]] = None
    best_key: Tuple[int, float, str] = (-1, -1.0, "")

    for rec in records:
        rank, dur = _score_record_for_timing(rec)
        meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
        finished = meta.get("call_finished_utc") or meta.get("task_finished_utc") or ""
        # prefer later completion when equal
        key = (rank, dur, str(finished))
        if key > best_key:
            best_key = key
            best = rec

    return best


def _list_top_files(dir_path: Path) -> List[Path]:
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    files: List[Path] = []
    for p in sorted(dir_path.iterdir()):
        if p.is_file():
            files.append(p)
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


def _write_task_time_file(
    out_path: Path,
    task_id: int,
    timing_rec: Optional[Dict[str, Any]],
) -> None:
    _ensure_parent_dir(out_path)

    if not timing_rec:
        out_path.write_text(
            f"task_id: {task_id}\nstatus: unknown\nelapsed_sec: unknown\n",
            encoding="utf-8",
        )
        return

    meta = timing_rec.get("meta") if isinstance(timing_rec.get("meta"), dict) else {}
    pack_path = timing_rec.get("_pack_path")

    call_started = meta.get("call_started_utc")
    call_finished = meta.get("call_finished_utc")
    elapsed = meta.get("elapsed_sec")
    call_elapsed = meta.get("call_elapsed_sec")
    status = meta.get("status")

    # Best-effort recompute from timestamps if missing.
    recomputed: Optional[float] = None
    dt0 = _parse_iso_utc(call_started)
    dt1 = _parse_iso_utc(call_finished)
    if dt0 and dt1 and dt1 >= dt0:
        recomputed = (dt1 - dt0).total_seconds()

    lines = [
        f"task_id: {task_id}",
        f"best_pack: {pack_path}",
        f"status: {status}",
        f"elapsed_sec: {elapsed}",
        f"call_elapsed_sec: {call_elapsed}",
        f"call_started_utc: {call_started}",
        f"call_finished_utc: {call_finished}",
    ]
    if recomputed is not None:
        lines.append(f"call_elapsed_sec_recomputed: {recomputed}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_task_attachments(
    task_id: int,
    manus_result_dir: Path,
    runs_artifacts_dir: Path,
    out_artifacts_dir: Path,
) -> Tuple[List[str], Dict[str, Any]]:
    """Return (attached_files_relpaths, debug_info)."""

    debug: Dict[str, Any] = {
        "sources": [],
        "converted": [],
        "proxied": [],
        "skipped": [],
    }

    # Prefer manus_result if present; also include runs artifacts when manus_result missing.
    sources: List[Path] = []
    manus_task_dir = manus_result_dir / str(task_id)
    runs_task_dir = runs_artifacts_dir / f"task_{task_id}"

    if manus_task_dir.exists():
        sources.append(manus_task_dir)
    if runs_task_dir.exists() and not manus_task_dir.exists():
        sources.append(runs_task_dir)
    # For task 82, manus_result is absent, we rely on runs artifacts.

    debug["sources"] = [_relpath_from_repo(p) for p in sources]

    all_files: List[Path] = []
    for d in sources:
        all_files.extend(_list_top_files(d))

    # Also include the runs artifacts directory if both exist but manus_result is missing key types.
    # (Keep minimal; we only add runs if manus_result absent, per above.)

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
        "--runs-dir",
        default="agent_app_eval/runs",
        help="Directory containing *_eval_pack.json and artifacts/",
    )
    ap.add_argument(
        "--manus-result-dir",
        default="agent_app_eval/manus_result",
        help="Directory containing <task_id>/ deliverables",
    )
    ap.add_argument(
        "--out-pack",
        default=None,
        help="Output eval-pack JSON path",
    )
    ap.add_argument(
        "--out-times-dir",
        default=None,
        help="Directory to write one timing file per task",
    )
    ap.add_argument(
        "--out-artifacts-dir",
        default=None,
        help="Directory to store converted/proxy artifacts",
    )

    args = ap.parse_args()

    dataset_end_json = Path(args.dataset_end_json)
    task_ids_file = Path(args.task_ids)
    runs_dir = Path(args.runs_dir)
    manus_result_dir = Path(args.manus_result_dir)

    task_ids = _read_task_ids(task_ids_file)
    dataset_tasks = load_dataset_tasks(dataset_end_json, task_ids)

    run_records = _collect_run_records(runs_dir)

    out_root = Path("agent_app_eval/gpt52_ready")
    out_pack = Path(args.out_pack) if args.out_pack else out_root / f"compiled_{_now_ts()}_eval_pack.json"
    out_times_dir = Path(args.out_times_dir) if args.out_times_dir else out_root / "task_times"
    out_artifacts_dir = Path(args.out_artifacts_dir) if args.out_artifacts_dir else out_root / "artifacts"

    tasks_out: List[Dict[str, Any]] = []

    for tid in task_ids:
        ds = dataset_tasks.get(tid)
        if ds is None:
            raise KeyError(f"missing dataset task {tid}")

        timing_rec = select_best_timing_record(run_records.get(tid, []))

        attached_files, attach_debug = _build_task_attachments(
            task_id=tid,
            manus_result_dir=manus_result_dir,
            runs_artifacts_dir=runs_dir / "artifacts" / "manus",
            out_artifacts_dir=out_artifacts_dir,
        )

        # Build a compact assistant final answer listing key deliverables.
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
                    "runs_records": len(run_records.get(tid, [])),
                    "best_timing_pack": (timing_rec.get("_pack_path") if isinstance(timing_rec, dict) else None),
                    "manus_result_dir": _relpath_from_repo((manus_result_dir / str(tid)).resolve())
                    if (manus_result_dir / str(tid)).exists()
                    else None,
                    "runs_artifacts_dir": _relpath_from_repo((runs_dir / "artifacts" / "manus" / f"task_{tid}").resolve())
                    if (runs_dir / "artifacts" / "manus" / f"task_{tid}").exists()
                    else None,
                },
                "timing": (timing_rec.get("meta") if isinstance(timing_rec, dict) else None),
                "attachments": attach_debug,
            },
        }

        tasks_out.append(task_entry)

        # Write per-task timing file.
        time_file = out_times_dir / f"task_{tid}_time.txt"
        _write_task_time_file(time_file, task_id=tid, timing_rec=timing_rec)

    generated_from = {
        "script": _relpath_from_repo(Path(__file__)),
        "created_utc": _utc_now_iso(),
        "dataset_end_json": _relpath_from_repo(dataset_end_json.resolve()),
        "task_ids": _relpath_from_repo(task_ids_file.resolve()),
        "runs_dir": _relpath_from_repo(runs_dir.resolve()),
        "manus_result_dir": _relpath_from_repo(manus_result_dir.resolve()),
        "out_times_dir": _relpath_from_repo(out_times_dir.resolve()),
        "out_artifacts_dir": _relpath_from_repo(out_artifacts_dir.resolve()),
        "num_tasks": len(tasks_out),
    }

    pack = build_eval_pack(tasks_out, generated_from=generated_from)

    _ensure_parent_dir(out_pack)
    out_pack.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[OK] wrote eval pack: {out_pack}")
    print(f"[OK] wrote per-task timing files: {out_times_dir}")


if __name__ == "__main__":
    main()
