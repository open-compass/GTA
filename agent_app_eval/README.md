# Agent App Eval (Example): End-to-End Agent Testing + GPT-5.2 Scoring

This folder is a minimal **end-to-end testing example** for an “agent app”:
you run your agent externally, convert its outputs into a GTA-Workflow eval-pack,
then score it with the repo-local GTA-Workflow (GTA-2) GPT-5.2 evaluator.

Key files:
- `examples/build_eval_pack_from_agent_app_result.py`: minimal adapter (edit it to fit your app’s output format).
- `score_with_gpt52.py`: scoring script (resumable; writes per-task results).
- `run_agents.py`: optional demo runner (Manus/OpenClaw examples).

## Prerequisites

Scoring uses OpenAI Responses API via the GTA-Workflow evaluator.

Set (any equivalent env var names are accepted by the evaluator):

```bash
export EVAL_OPENAI_API_KEY="YOUR_KEY"          # or OPENAI_API_KEY
export EVAL_OPENAI_BASE_URL="https://.../v1"   # must end with /v1

# optional
export EVAL_OPENAI_MODEL="gpt-5.2"
export EVAL_PROXY="http://127.0.0.1:7897"
```

Note: if you accidentally set a full endpoint like `.../v1/chat/completions`,
`score_with_gpt52.py` will try to normalize it to `.../v1`.

## End-to-End Example (recommended)

### 1) Run your agent app and save outputs

Create one folder per task id:

```
agent_app_eval/agent_app_result/
  <task_id>/
    final.txt
    files/        # optional deliverables
      ...
```

### 2) Convert outputs to an eval-pack

```bash
python agent_app_eval/examples/build_eval_pack_from_agent_app_result.py \
  --result-dir agent_app_eval/agent_app_result \
  --out-pack agent_app_eval/runs/agent_app_eval_pack.json
```

### 3) Score with the GTA-Workflow evaluator

Dry-run (no API calls):

```bash
python agent_app_eval/score_with_gpt52.py \
  --in-pack agent_app_eval/runs/agent_app_eval_pack.json \
  --dry-run
```

Run scoring:

```bash
python agent_app_eval/score_with_gpt52.py \
  --in-pack agent_app_eval/runs/agent_app_eval_pack.json
```

Re-run the same command to resume; use `--overwrite` to restart from scratch.

## Optional: Demo runners (Manus/OpenClaw)

```bash
python agent_app_eval/run_agents.py --backend manus --dry-run
python agent_app_eval/run_agents.py --backend openclaw --dry-run
```
