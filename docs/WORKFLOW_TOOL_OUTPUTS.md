# GTA workflow tool outputs

OpenCompass stores AgentInferencer artifacts alongside predictions under
`tools_output/<model>/<dataset>/task_<id>/attempt_<unique>/`. Each attempt has
a `files/` directory for tools with an output directory argument. Files
materialized on the inference client are retained under `calls/`. The
prediction records the attempt path as `tools_output_dir`. When the tool
server and client run on different hosts, use a shared path for server-written
files. File responses sent to the client are materialized on the client.

## Web evidence

Set `AGENTLEGO_WEB_CACHE_DIR` in the tool server environment to record successful
Serper responses. Identical search types and parameters reuse that response.
Use a fresh directory for a new live collection. With this variable unset and
no explicit cache directory the search tool queries Serper as usual.

From the repository root, extract the search replies actually saved in a run:

```sh
python analysis/extract_web_trajectory.py --run-dir "$RUN" --output "$EVIDENCE/search.csv"
python analysis/archive_web_urls.py --input "$EVIDENCE/search.csv" --output-dir "$EVIDENCE/pages"
```

The first command is offline. It records the prediction file's modification time
and leaves unavailable tool replies empty. The second command fetches public
HTTP(S) pages at archive time and labels them `archived_after_evaluation`.
The manifest records the capture time, final URL, content type and content hash
for each successfully downloaded page. A subsequent invocation reuses verified
page bodies and retries failures.

## Native tool calls

Use `opencompass.models.native_workflow.NativeLagentAgent` as the model type to
opt in to the OpenRouter Chat Completions adapter. It exposes the available
tools declared for the task, executes at most one call per turn, and links tool
observations to the provider's call IDs. Its built-in system prompt can be
overridden by passing a string as `native_system_prompt`. The prompt for this
adapter is independent of the Lagent text protocol. The adapter writes per-turn
events inside the task attempt during scoped inference.

Run the focused tests from the repository root in an environment with the GTA
workflow and audio tool dependencies installed:

```sh
export PYTHONPATH="$PWD/opencompass:$PWD/agentlego${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s opencompass/tests -p 'test_tool_output*.py'
python -m unittest discover -s opencompass/tests -p 'test_native*.py'
python -m unittest discover -s opencompass/tests -p 'test_gta_web_audio.py'
```
