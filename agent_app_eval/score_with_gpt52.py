"""Score an eval-pack with the GTA v2 GPT-5.2 checkpoint evaluator.

- Independent entrypoint (no OpenCompass runner needed).
- Reuses the repo's checkpoint evaluator implementation.
- Supports proxy/base_url via env:
  - EVAL_PROXY (preferred)
  - EVAL_OPENAI_API_KEY / OPENAI_API_KEY
  - EVAL_OPENAI_BASE_URL / OPENAI_BASE_URL

Note: The evaluator uses OpenAI Responses API. If you previously configured a
Chat Completions URL like ".../v1/chat/completions", this script will normalize
it to ".../v1" to avoid calling a wrong endpoint.

This script is *resumable* by default:
- Scores tasks one-by-one.
- After each task, writes the updated result JSON to disk.
- If interrupted, rerun with the same `--out` and it will skip tasks already
    scored successfully.

It also writes a separate summary JSON (see `--summary-out`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically to avoid partial files on crash/interrupt."""

    _ensure_parent_dir(path)

    payload = json.dumps(data, ensure_ascii=False, indent=2)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _normalize_openai_base_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None

    u = url.strip().rstrip("/")

    # Common misconfiguration in older configs.
    suffix = "/v1/chat/completions"
    if u.endswith(suffix):
        u = u[: -len(suffix)]

    # If user passed a full endpoint, attempt to keep just the base.
    if u.endswith("/chat/completions"):
        u = u[: -len("/chat/completions")]

    return u


def _maybe_fix_env_base_url() -> None:
    # The evaluator reads OPENAI_BASE_URL or EVAL_OPENAI_BASE_URL.
    for key in ("EVAL_OPENAI_BASE_URL", "OPENAI_BASE_URL"):
        if os.getenv(key):
            fixed = _normalize_openai_base_url(os.getenv(key))
            if fixed and fixed != os.getenv(key):
                os.environ[key] = fixed
                print(f"[score] Normalized {key} -> {fixed}")


def _load_evaluator_class() -> Any:
    # Ensure we import the evaluator from the *repo-local* OpenCompass checkout
    # (not the site-packages `opencompass`).
    repo = _repo_root()
    sys_path_entry = repo / "opencompass"

    import sys

    if str(sys_path_entry) not in sys.path:
        sys.path.insert(0, str(sys_path_entry))

    # Use GTA-Workflow (GTA-2) evaluator.
    from opencompass.datasets.gta_bench_v2 import GPTEvaluator  # type: ignore

    return GPTEvaluator


def _select_trace(assistant_outputs_field: Any) -> Any:
    # Expected: list of traces, choose the first.
    if isinstance(assistant_outputs_field, list) and assistant_outputs_field:
        # Could be [[...steps...]] or already [...steps...]
        first = assistant_outputs_field[0]
        if isinstance(first, list):
            return first
    return assistant_outputs_field


def _abs_paths(attached_files: Any) -> Any:
    # Evaluator expects local filesystem paths.
    if attached_files is None:
        return []
    if isinstance(attached_files, list):
        out = []
        for p in attached_files:
            if isinstance(p, str) and p.strip():
                out.append(str((_repo_root() / p).resolve()) if not os.path.isabs(p) else p)
        return out
    if isinstance(attached_files, str):
        p = attached_files
        return str((_repo_root() / p).resolve()) if not os.path.isabs(p) else p
    return attached_files


def _infer_default_out_dir(pack: Dict[str, Any], pack_path: Path) -> Path:
    """Choose a default output directory.

    Requirement from user:
    - OpenClaw results should live under the OpenClaw folder.
    - Manus keeps the historical behavior (next to the pack) unless overridden.
    """

    gen = pack.get("generated_from") if isinstance(pack.get("generated_from"), dict) else {}
    openclaw_dir = gen.get("openclaw_result_dir")
    if isinstance(openclaw_dir, str) and openclaw_dir.strip():
        # e.g. agent_app_eval/openclaw/openclaw_result -> parent is .../openclaw
        p = Path(openclaw_dir)
        try:
            return (_repo_root() / p).resolve().parent
        except Exception:
            return pack_path.resolve().parent
    return pack_path.resolve().parent


def _ensure_openai_api_key() -> None:
    if os.getenv("OPENAI_API_KEY") or os.getenv("EVAL_OPENAI_API_KEY"):
        return
    raise SystemExit(
        "Missing OpenAI API key. Set EVAL_OPENAI_API_KEY (preferred) or OPENAI_API_KEY before scoring."
    )


def _init_or_resume_state(
    *,
    out_path: Path,
    pack_path: Path,
    pack_fingerprint: Dict[str, Any],
    proxy: Optional[str],
    dry_run: bool,
    overwrite: bool,
    force_resume: bool,
) -> Dict[str, Any]:
    if overwrite or (not out_path.exists()):
        return {
            "schema_version": 2,
            "generated_from": {
                "script": _relpath_from_repo(Path(__file__)),
                "pack": _relpath_from_repo(pack_path.resolve()),
                "model": os.getenv("EVAL_OPENAI_MODEL", "gpt-5.2"),
                "proxy": proxy or os.getenv("EVAL_PROXY") or os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY"),
                "base_url": os.getenv("EVAL_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
                "dry_run": bool(dry_run),
                "created_utc": _utc_now_iso(),
            },
            "pack_fingerprint": pack_fingerprint,
            "task_results": {},
            "updated_utc": _utc_now_iso(),
        }

    try:
        state = _read_json(out_path)
    except Exception:
        if not force_resume:
            raise SystemExit(f"Failed to read existing out file: {out_path} (use --overwrite to start fresh)")
        state = {}

    # Basic validation + compatibility guard.
    if isinstance(state, dict):
        old_fp = state.get("pack_fingerprint")
        if isinstance(old_fp, dict):
            if old_fp.get("sha1") and pack_fingerprint.get("sha1") and old_fp.get("sha1") != pack_fingerprint.get("sha1"):
                if not force_resume:
                    raise SystemExit(
                        "Out file belongs to a different pack (sha1 mismatch). "
                        "Use --overwrite to start fresh, or --force-resume to ignore."
                    )
    else:
        if not force_resume:
            raise SystemExit(f"Invalid out JSON (not a dict): {out_path} (use --overwrite)")
        state = {}

    # Normalize minimal fields.
    if not isinstance(state, dict):
        state = {}
    if not isinstance(state.get("task_results"), dict):
        state["task_results"] = {}
    state["schema_version"] = state.get("schema_version") or 2
    state["pack_fingerprint"] = state.get("pack_fingerprint") or pack_fingerprint
    state["updated_utc"] = _utc_now_iso()
    if not isinstance(state.get("generated_from"), dict):
        state["generated_from"] = {
            "script": _relpath_from_repo(Path(__file__)),
            "pack": _relpath_from_repo(pack_path.resolve()),
            "model": os.getenv("EVAL_OPENAI_MODEL", "gpt-5.2"),
            "proxy": proxy or os.getenv("EVAL_PROXY") or os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY"),
            "base_url": os.getenv("EVAL_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
            "dry_run": bool(dry_run),
            "created_utc": _utc_now_iso(),
        }

    return state


def _compute_summary(task_ids: List[int], state: Dict[str, Any]) -> Dict[str, Any]:
    task_results = state.get("task_results") if isinstance(state.get("task_results"), dict) else {}

    per_task = []
    ok_scores: List[float] = []
    n_error = 0
    n_ok = 0

    for tid in task_ids:
        rec = task_results.get(str(tid))
        if not isinstance(rec, dict):
            per_task.append({"task_id": tid, "status": "pending", "score": None})
            continue

        status = rec.get("status")
        score = rec.get("score")
        if status == "ok":
            n_ok += 1
            try:
                ok_scores.append(float(score))
            except Exception:
                pass
        elif status == "error":
            n_error += 1

        per_task.append({"task_id": tid, "status": status or "unknown", "score": score})

    mean_ok = (sum(ok_scores) / len(ok_scores)) if ok_scores else None
    n_total = len(task_ids)
    n_done = n_ok + n_error

    return {
        "schema_version": 1,
        "generated_from": {
            "details_file": _relpath_from_repo(Path(state.get("_out_path", "")).resolve())
            if state.get("_out_path")
            else None,
            "pack": state.get("generated_from", {}).get("pack") if isinstance(state.get("generated_from"), dict) else None,
            "model": state.get("generated_from", {}).get("model") if isinstance(state.get("generated_from"), dict) else None,
            "created_utc": state.get("generated_from", {}).get("created_utc") if isinstance(state.get("generated_from"), dict) else None,
        },
        "updated_utc": _utc_now_iso(),
        "summary": {
            "n_total": n_total,
            "n_done": n_done,
            "n_ok": n_ok,
            "n_error": n_error,
            "mean_ok_score": mean_ok,
        },
        "per_task": per_task,
    }


def score_pack_incremental(
    *,
    pack_path: Path,
    out_path: Path,
    summary_out: Path,
    proxy: Optional[str],
    dry_run: bool,
    overwrite: bool,
    force_resume: bool,
    max_tasks: Optional[int],
) -> Dict[str, Any]:
    pack = _read_json(pack_path)
    tasks = pack.get("tasks") or []

    if not isinstance(tasks, list):
        raise ValueError("Invalid pack: tasks must be a list")

    task_ids: List[int] = []
    task_by_id: Dict[int, Dict[str, Any]] = {}
    for t in tasks:
        if not isinstance(t, dict):
            continue
        tid = int(t.get("task_id"))
        task_ids.append(tid)
        task_by_id[tid] = t

    # Fingerprint pack for safe resume.
    try:
        st = pack_path.stat()
        fp = {"sha1": _sha1_file(pack_path), "size": int(st.st_size), "mtime": float(st.st_mtime)}
    except Exception:
        fp = {"sha1": None, "size": None, "mtime": None}

    state = _init_or_resume_state(
        out_path=out_path,
        pack_path=pack_path,
        pack_fingerprint=fp,
        proxy=proxy,
        dry_run=dry_run,
        overwrite=overwrite,
        force_resume=force_resume,
    )
    state["_out_path"] = str(out_path)

    # Dry-run: validate file existence and emit a compact report.
    if dry_run:
        missing_files: List[Tuple[int, str]] = []
        for tid in task_ids:
            t = task_by_id.get(tid) or {}
            files = _abs_paths(t.get("attached_files"))
            if isinstance(files, list):
                for p in files:
                    if isinstance(p, str) and p and not Path(p).exists():
                        missing_files.append((tid, p))
            elif isinstance(files, str) and files and not Path(files).exists():
                missing_files.append((tid, files))

        state["dry_run"] = {
            "missing_attached_files": missing_files,
            "n_tasks": len(task_ids),
        }
        state["updated_utc"] = _utc_now_iso()
        _atomic_write_json(out_path, state)

        summary = _compute_summary(task_ids, state)
        _atomic_write_json(summary_out, summary)
        return state

    _ensure_openai_api_key()

    _maybe_fix_env_base_url()
    if proxy:
        os.environ["EVAL_PROXY"] = proxy

    GPTEvaluator = _load_evaluator_class()
    evaluator = GPTEvaluator(mode="every", proxy=proxy)

    task_results = state.get("task_results") if isinstance(state.get("task_results"), dict) else {}

    remaining = []
    for tid in task_ids:
        rec = task_results.get(str(tid))
        if isinstance(rec, dict) and rec.get("status") == "ok":
            continue
        remaining.append(tid)

    if max_tasks is not None:
        remaining = remaining[: max(0, int(max_tasks))]

    for tid in remaining:
        t = task_by_id.get(tid) or {}

        prev = task_results.get(str(tid))
        attempt = 1
        if isinstance(prev, dict):
            try:
                attempt = int(prev.get("attempt") or 0) + 1
            except Exception:
                attempt = 1

        started = time.time()
        started_utc = _utc_now_iso()
        try:
            trace = _select_trace(t.get("assistant_outputs"))
            ckpt = t.get("full_tree")
            origin_prompt = t.get("origin_prompt")
            files = _abs_paths(t.get("attached_files"))

            scores = evaluator.score(
                assistant_outputs=[trace],
                checkpoints=[ckpt],
                origin_prompt=[origin_prompt],
                attached_files=[files],
                indices=[tid],
            )

            root_score = None
            try:
                ss = scores.get("sample_scores")
                if isinstance(ss, list) and ss:
                    root_score = float(ss[0])
                elif scores.get("gpt_score") is not None:
                    root_score = float(scores.get("gpt_score"))
            except Exception:
                root_score = None

            detail = None
            try:
                det = scores.get("details")
                if isinstance(det, list) and det:
                    detail = det[0]
            except Exception:
                detail = None

            task_results[str(tid)] = {
                "task_id": tid,
                "status": "ok",
                "score": root_score,
                "detail": detail,
                "attempt": attempt,
                "started_utc": started_utc,
                "finished_utc": _utc_now_iso(),
                "duration_sec": round(time.time() - started, 3),
            }
            print(f"[score] task_id={tid} OK score={root_score}")

        except KeyboardInterrupt:
            raise
        except Exception as e:
            task_results[str(tid)] = {
                "task_id": tid,
                "status": "error",
                "score": None,
                "detail": None,
                "attempt": attempt,
                "started_utc": started_utc,
                "finished_utc": _utc_now_iso(),
                "duration_sec": round(time.time() - started, 3),
                "error": f"{type(e).__name__}: {e}",
            }
            print(f"[score] task_id={tid} ERROR: {type(e).__name__}: {e}")

        # Persist after each task.
        state["task_results"] = task_results
        state["updated_utc"] = _utc_now_iso()
        _atomic_write_json(out_path, state)

        summary = _compute_summary(task_ids, state)
        _atomic_write_json(summary_out, summary)

    # Final write (already done), return state for callers.
    return state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-pack",
        required=True,
        help="Path to eval-pack JSON produced by run_agents.py",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output (details) JSON path; supports resume by default",
    )
    ap.add_argument(
        "--summary-out",
        default=None,
        help="Output summary JSON path (separate from details)",
    )
    ap.add_argument(
        "--proxy",
        default=None,
        help="Proxy override (also sets EVAL_PROXY); default uses EVAL_PROXY",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing --out instead of resuming",
    )
    ap.add_argument(
        "--force-resume",
        action="store_true",
        help="Resume even if pack fingerprint mismatches (not recommended)",
    )
    ap.add_argument(
        "--max-tasks",
        default=None,
        type=int,
        help="Only score up to N remaining tasks (debug)",
    )

    args = ap.parse_args()

    in_pack = Path(args.in_pack)

    pack = _read_json(in_pack)
    default_dir = _infer_default_out_dir(pack, in_pack)

    out_path = Path(args.out).resolve() if args.out else (default_dir / (in_pack.stem + "_gpt52_scores.json"))
    summary_out = (
        Path(args.summary_out).resolve()
        if args.summary_out
        else out_path.with_name(out_path.stem + "_summary.json")
    )

    res = score_pack_incremental(
        pack_path=in_pack,
        out_path=out_path,
        summary_out=summary_out,
        proxy=args.proxy,
        dry_run=bool(args.dry_run),
        overwrite=bool(args.overwrite),
        force_resume=bool(args.force_resume),
        max_tasks=args.max_tasks,
    )

    # One more summary print for quick inspection.
    try:
        task_ids = [int(t.get("task_id")) for t in (pack.get("tasks") or []) if isinstance(t, dict) and t.get("task_id") is not None]
        summary = _compute_summary(task_ids, {**res, "_out_path": str(out_path)})
        s = summary.get("summary") if isinstance(summary, dict) else {}
        print(f"[OK] wrote details: {out_path}")
        print(f"[OK] wrote summary:  {summary_out}")
        print(f"[DONE] ok={s.get('n_ok')}/{s.get('n_total')} mean_ok_score={s.get('mean_ok_score')}")
    except Exception:
        print(f"[OK] wrote details: {out_path}")
        print(f"[OK] wrote summary:  {summary_out}")


if __name__ == "__main__":
    main()
