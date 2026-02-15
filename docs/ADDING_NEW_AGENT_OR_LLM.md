# Adding a New Agent or Integrating a Different LLM (GTA + OpenCompass + Lagent)

This guide shows how to:

1. Add a new model (LLM) to GTA evaluation.
2. Integrate an OpenAI-compatible endpoint (LMDeploy / vLLM / custom gateway).
3. Register a *custom agent wrapper* compatible with the OpenCompass `AgentInferencer` (Lagent-based).
4. Understand which config files to change and which dependencies/compat tweaks typically matter.

> Scope: This repo’s GTA evaluation uses OpenCompass configs under `GTA/opencompass/configs/` and the Lagent agent wrapper in `GTA/opencompass/opencompass/models/lagent.py`.

---

## 0) Mental model: what gets built where

### Key configs you will edit

- **Model/agent definition**: `GTA/opencompass/configs/eval_gta_bench.py`
  - The `models = [ ... ]` list defines **(agent wrapper + underlying LLM + tool wiring)**.

- **Dataset + infer/eval mode**: `GTA/opencompass/configs/datasets/gta_bench.py`
  - Choose step-by-step (every_with_gt) or end-to-end (every) mode.

### Key runtime components

- **Inferencer**: `opencompass.openicl.icl_inferencer.AgentInferencer`
  - GTA uses “agent-style” inferencer which called `model.chat` for every (end-to-end) mode and `model.next_step` for every_with_gt (step-by-step) mode.

- **Agent wrapper**: `opencompass.models.lagent.LagentAgent`
  - Bridges OpenCompass ↔ Lagent.
  - Handles `tool_server` (real remote tools) vs `tool_meta` (dummy tool list for tool-selection scoring / step-by-step).

- **Model wrapper** (LLM): typically one of `opencompass.models.*` (e.g. `OpenAI`, `Gemini`, `Qwen`)
  - For OpenAI-compatible endpoints, you usually use `opencompass.models.openai_api.OpenAI`.
  - You can also implement another model wrapper and register it.

---

## 1) Choose evaluation mode first (affects tool wiring)

In GTA, the same benchmark can be run in two major ways:

### A) Step-by-step (`every_with_gt`) — no real tool execution

- Use `tool_meta=...` in your model entry.
- Set infer/eval mode to `every_with_gt`.

Edit `GTA/opencompass/configs/datasets/gta_bench.py`:

```python
# step-by-step
inferencer=dict(type=AgentInferencer, infer_mode='every_with_gt')
...
gta_bench_eval_cfg = dict(evaluator=dict(type=GTABenchEvaluator, mode='every_with_gt'))
```

### B) End-to-end (`every`) — actually calls the deployed tools

- Use `tool_server='http://<host>:<port>'` in your model entry.
- Set infer/eval mode to `every`.

Edit `GTA/opencompass/configs/datasets/gta_bench.py`:

```python
# end-to-end
inferencer=dict(type=AgentInferencer, infer_mode='every')
...
gta_bench_eval_cfg = dict(evaluator=dict(type=GTABenchEvaluator, mode='every'))
```

---

## 2) Add a new LLM via an OpenAI-compatible endpoint (recommended)

This is the fastest path if your model is served behind an OpenAI-style `POST /v1/chat/completions`.

### Step 1 — Start your model server

Examples:

- **LMDeploy** (as in the README): serve your model on some port.
- **vLLM**: `vllm serve ...` exposing OpenAI-compatible endpoints.

Make sure you know:
- API base URL (local or remote), e.g. `http://127.0.0.1:12580/v1/chat/completions`, `https://api.example.com/v1/chat/completions`
- Model name used by the server (the `path` you pass below)

### Step 2 — Add a model entry in `eval_gta_bench.py`

Edit `GTA/opencompass/configs/eval_gta_bench.py` and add a new dict inside `models = [...]`.

**Minimal template (OpenAI-compatible):**

```python
from lagent.agents import ReAct
from opencompass.models.openai_api import OpenAI
from opencompass.models.lagent import LagentAgent

models = [
  dict(
    abbr='my-llm',
    type=LagentAgent,
    agent_type=ReAct,
    max_turn=10,
    llm=dict(
      type=OpenAI,
      path='your-model-name-on-server',
      key='EMPTY', # or your api key if needed
      openai_api_base='http://127.0.0.1:12580/v1/chat/completions',
      query_per_second=1,
      max_seq_len=4096,
      # Optional (useful for some chat templates):
      # stop='<|im_end|>',
      # retry=10,
    ),

    # Choose exactly ONE of these depending on mode:
    # tool_server='http://127.0.0.1:16181',
    tool_meta='data/gta_dataset/toolmeta.json',

    batch_size=8,
  )
]
```

### Step 3 — Sanity-check the dataset path

The GTA dataset config points to:

- `path="data/gta_dataset"` in `GTA/opencompass/configs/datasets/gta_bench.py`

So `tool_meta` should generally be:

- `data/gta_dataset/toolmeta.json`

Or you can change accordingly, but make sure they are consistence.

---

## 3) Integrate a different LLM **without** OpenAI compatibility

If your model is not OpenAI-compatible, you typically add a new **OpenCompass model wrapper**.

### Where to implement

- Add a new file under: `GTA/opencompass/opencompass/models/`
  - Example: `my_api_model.py`

### How registration works (OpenCompass registry)

OpenCompass uses MMEngine registries in `GTA/opencompass/opencompass/registry.py`. For models:

- `MODELS = Registry('model', locations=['opencompass.models'])`

So you register a model wrapper using:

```python
from opencompass.registry import MODELS

@MODELS.register_module()
class MyModel(...):
    ...
```

### Practical tip: ensure it’s importable

To avoid “type not found” issues during config loading, you can also add an import line to:

- `GTA/opencompass/opencompass/models/__init__.py`

For example:

```python
from .my_api_model import MyModel  # noqa: F401
```

### Minimal skeleton (API model)

Use existing wrappers as references (e.g. `openai_api.py`, `gemini_api.py`). At minimum, your model wrapper should provide a `generate()` method compatible with OpenCompass’ `BaseModel` / `BaseAPIModel` patterns.

---

## 4) Add / register a new agent in the OpenCompass + Lagent pipeline

There are two common cases:

### Case A — You only need different prompting / tools / protocol

You can usually do this **without writing new Python code**.

In `GTA/opencompass/configs/eval_gta_bench.py`:

- Change `agent_type` (e.g. `ReAct`)
- Add `actions=[ ... ]` (local tools/actions)
- Provide a custom `protocol=...`
- Tune `max_turn` / sampling params

Example patterns exist in:

- `GTA/opencompass/configs/eval_chat_agent.py`

### Case B — You are implementing a new agent wrapper

If you want a new agent wrapper class (e.g., to change how tool calls are parsed/executed, or to support a different agent framework), you need compatibility with the inferencer.

#### The compatibility contract (important)

`AgentInferencer` calls:

```python
steps, memory = model.chat(query=..., memory=..., resources=...)
```

So your agent wrapper should implement:

- `chat(query: str, memory=None, resources=None) -> (steps, memory)`

And return `steps` in a tool-agent-like structure. The built-in `LagentAgent` already matches this.

You can inspect how it behaves in:

- `GTA/opencompass/opencompass/models/lagent.py`
- `GTA/opencompass/opencompass/openicl/icl_inferencer/icl_agent_inferencer.py`

#### Where to put your new agent wrapper

Recommended location:

- `GTA/opencompass/opencompass/models/your_agent_wrapper.py`

If you want OpenCompass to build it by name (string type), register it:

```python
from opencompass.registry import MODELS

@MODELS.register_module()
class MyAgentWrapper:
    def __init__(self, ..., **kwargs):
        ...

    def chat(self, query, memory=None, resources=None):
        ...
        return steps, memory
```

Then in `eval_gta_bench.py` you can reference it either as:

- `type=MyAgentWrapper` (import the class in the config file), or
- `type='MyAgentWrapper'` (if you rely on registry auto-import; importing in `models/__init__.py` is safest)

---

## 5) Dependencies & compatibility adjustments (common gotchas)

### Python / environment split

The repo README suggests different conda envs:

- `opencompass` env (often Python 3.10) running the evaluation
- `agentlego` env (often Python 3.11) running tool server separately

### Required packages

- For Lagent agent wrapper: `lagent`
- For tool server / tool wrappers: `agentlego`
- For OpenAI wrapper: `requests`, `tiktoken` (already used in `OpenAI` wrapper)
- Other packages depend on which tools you deploy (OCR, vision models, etc.).

If you run into complex dependency conflicts (common when mixing CV/ML stacks), keep the tool server isolated (separate conda env, Docker container, or a dedicated machine) may be helpful. You *can* run multiple tool servers on different ports, but OpenCompass/Lagent is typically configured with a single `tool_server` URL, so using multiple servers requires extra wiring (e.g., a small proxy/aggregator or a custom wrapper that merges tool lists).
