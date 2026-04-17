#!/usr/bin/env python3
"""Build a GTA-Workflow eval-pack from *any* agent app's outputs.

This script is intentionally minimal and serves as an example adapter.

Expected result directory layout (customize as needed):

  <result_dir>/
    <task_id>/
      final.txt                 # required, the agent app's final answer
      files/                    # optional, any deliverables/attachments
        ...

It converts these outputs into the eval-pack JSON schema consumed by
`score_with_gpt52.py`.

Notes:
- Paths inside `attached_files` should be *relative to the repo root*.
- If you store outputs under the repo (recommended), this script will emit
  relative paths automatically.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _repo_root() -> Path:
    # examples/ -> agent_app_eval/ -> GTA/
    return Path(__file__).resolve().parents[2]


def _load_task_ids(path: Path) -> List[int]:
    ids: List[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        ids.append(int(s))
    return ids


def _index_end_json(end_json: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Index tasks by task_id from a GTA end.json.

    We try to be permissive about the JSON shape.
    """

    # Common shapes:
    # - {"tasks": [...]} 
    # - [...]
    candidates: Any = end_json.get("tasks") if isinstance(end_json, dict) else end_json
    if not isinstance(candidates, list):
        raise ValueError("Unsupported end.json format: expected a list or dict with 'tasks'")

    out: Dict[int, Dict[str, Any]] = {}
    for item in candidates:
        if not isinstance(item, dict):
            continue
        tid = item.get("task_id")
        if tid is None:
            continue
        out[int(tid)] = item
    return out


def _rel_from_repo(p: Path) -> str:
    repo = _repo_root()
    try:
        return str(p.resolve().relative_to(repo.resolve()))
    except Exception:
        # Fallback to absolute path (works, but less portable).
        return str(p.resolve())


def _read_text_if_exists(p: Path) -> Optional[str]:
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8", errors="replace")


def _collect_files(files_dir: Path) -> List[str]:
    if not files_dir.exists() or not files_dir.is_dir():
        return []

    paths: List[str] = []
    for fp in sorted(files_dir.rglob("*")):
        if fp.is_file():
            paths.append(_rel_from_repo(fp))
    return paths


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
        "--result-dir",
        required=True,
        help="Directory containing <task_id>/final.txt and optional <task_id>/files/",
    )
    ap.add_argument(
        "--out-pack",
        required=True,
        help="Output eval-pack JSON path",
    )

    args = ap.parse_args()

    repo = _repo_root()
    end_path = (repo / args.dataset_end_json).resolve() if not os.path.isabs(args.dataset_end_json) else Path(args.dataset_end_json)
    task_ids_path = (repo / args.task_ids).resolve() if not os.path.isabs(args.task_ids) else Path(args.task_ids)

    result_dir = (repo / args.result_dir).resolve() if not os.path.isabs(args.result_dir) else Path(args.result_dir).resolve()
    out_pack = (repo / args.out_pack).resolve() if not os.path.isabs(args.out_pack) else Path(args.out_pack).resolve()

    end_json = json.loads(end_path.read_text(encoding="utf-8"))
    indexed = _index_end_json(end_json)

    task_ids = _load_task_ids(task_ids_path)

    tasks: List[Dict[str, Any]] = []
    for i, tid in enumerate(task_ids, start=1):
        if tid not in indexed:
            raise KeyError(f"task_id={tid} not found in end.json")

        item = indexed[tid]
        prompt = item.get("prompt") or item.get("query") or ""
        full_tree = item.get("full_tree")

        task_dir = result_dir / str(tid)
        final_txt = task_dir / "final.txt"
        content = _read_text_if_exists(final_txt)
        if content is None:
            raise FileNotFoundError(f"Missing {final_txt} for task_id={tid}")

        attached = _collect_files(task_dir / "files")

        tasks.append(
            {
                "task_id": int(tid),
                "sample_idx": i,
                "origin_prompt": [prompt],
                "assistant_outputs": [[{"role": "assistant", "content": content}]],
                "attached_files": attached,
                "full_tree": full_tree,
                "meta": {
                    "backend": "agent_app",
                    "result_dir": _rel_from_repo(result_dir),
                },
            }
        )

    pack = {
        "schema_version": 1,
        "generated_from": {
            "backend": "agent_app",
            "dataset_end_json": _rel_from_repo(end_path),
            "task_ids": _rel_from_repo(task_ids_path),
            "result_dir": _rel_from_repo(result_dir),
        },
        "tasks": tasks,
    }

    out_pack.parent.mkdir(parents=True, exist_ok=True)
    out_pack.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote eval-pack: {out_pack}")


if __name__ == "__main__":
    main()
