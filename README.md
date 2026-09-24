# batchlane

Submit asynchronous LLM batch jobs through one interface.

Eight adapters cover Anthropic, Gemini AI Studio, OpenAI, Groq, Mistral,
Fireworks, Together, and DeepInfra. Batch pricing, model availability, and
turnaround depend on the provider. Only Anthropic has been verified end to end
against a live API; the other adapters have mocked contract tests.

```python
import batchlane as bl

model = "groq/llama-3.3-70b-versatile"
prompts = ["The product was great.", "It broke in a week."]
answers = bl.map(model, prompts, system="Classify the sentiment.")
```

One prompt over many inputs, answers back in input order, `None` where a row
failed. Underneath it splits the job to fit the provider's caps, submits
however many batches that takes, waits, and rejoins the results.

From the shell, for the file-to-file case:

```bash
batchlane run rows.jsonl --model groq/llama-3.3-70b-versatile -o answers.jsonl
```

Every field on an input row is copied to its output row beside a new `answer`,
so your own columns stay attached to their results. `--dry-run` reports the
chunking without submitting anything; `batchlane providers` lists the lanes.

When you need per-row control, `run()` gives you each input back beside its
result:

```python
rows = [
    bl.BatchLine(
        "row-1",
        "groq/llama-3.3-70b-versatile",
        [{"role": "user", "content": "Classify: the product was great"}],
    ),
    bl.BatchLine(
        "row-2",
        "groq/llama-3.3-70b-versatile",
        [{"role": "user", "content": "Classify: it broke in a week"}],
    ),
]

for line, result in bl.run(rows, checkpoint="job.jsonl"):
    print(line.custom_id, bl.answer_text(result))
```

Results are yielded one row at a time. Requests and downloaded provider output
are currently buffered in memory; size jobs to fit your machine.

What that buys you:

| | |
|---|---|
| **Provider batch pricing** | Discounts and model exclusions vary; see the provider table below |
| **One code path** | chunking, polling, and joining results back to rows are handled |
| **Resumable** | recorded handles reattach to submitted jobs; recovery limits are described below |
| **Honest about limits** | refuses where no lane exists rather than emulating one, and distinguishes "no lane" from "not built yet" |

## Resume submitted jobs

With `checkpoint=` set, batchlane records an intent before each submission and
saves the returned handle immediately. Repeating the identical call reattaches
to recorded jobs. Changed requests, model parameters, ordering, or job settings
are rejected; use a new checkpoint for new work. Use one writer per checkpoint.

If a submission may have succeeded but no handle was saved, recovery depends
on the provider's listing and matching support. It is not an exactly-once
guarantee. Providers retain results for a limited time, so save collected answers
locally if you need them beyond that window. Checkpoints store handles, not answers.

`run()` submits every chunk before waiting. To submit now and collect later:

```python
handles = bl.submit_all(rows, checkpoint="job.jsonl")
for handle in handles:
    if bl.status(handle).state == "succeeded":
        for result in bl.results(handle):
            print(result.custom_id, bl.answer_text(result))
```

`plan(rows).chunks` describes which requests each handle covers.

## OpenAI Responses batches

Use native Responses `input` with `endpoint="responses"`. Leave `messages`
empty and put other Responses parameters in `params`:

```python
import batchlane as bl

lines = [
    bl.BatchLine(
        "page-1",
        "openai/gpt-4o-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Summarize this page."},
                ],
            }
        ],
        params={"max_output_tokens": 200},
    )
]
handles = bl.submit_all(lines, endpoint="responses", checkpoint="responses.jsonl")
for line, result in bl.run(lines, endpoint="responses", checkpoint="responses.jsonl"):
    print(line.custom_id, bl.answer_text(result))
```

Inputs can include native image and file blocks. Result bodies retain native
output items, refusals, tool calls, reasoning details, and usage; `answer_text`
extracts only output text. This uses the [OpenAI Batch API](https://developers.openai.com/api/reference/resources/batches/methods/create).
Streaming and background mode are not batch request modes. Other provider lanes
continue to accept chat-completion requests. Responses input tokens and costs
are not estimated offline; `actual_cost` can price collected usage.

## What will bite you, before it does

```python
for note in bl.plan(rows).caveats:
    print(note)
```

Lanes differ in ways that change what you should do, not just how the client
talks to them. `plan()` states the ones that apply to your job, and the CLI
prints them on every run. `batchlane run ... --dry-run` shows them with the
cost and the chunking and submits nothing.

## What it actually cost

```python
results = list(bl.results(handle))
print(bl.actual_cost(results, handle.provider))
```

`plan().cost` estimates a job before it runs; `actual_cost()` prices usage
reported by the provider. Where a provider returns a service tier, the cost
report carries a caveat if that tier differs from batch pricing.

## Will it fit?

```python
p = bl.plan(rows)
print(p.n_chunks, p.total_bytes, p.limit_bytes)
print(p.cost)
```

The `~` marks a derived rate. Where batch rates are absent from LiteLLM's price
registry, batchlane applies the provider discount in its capability table.
Together has no estimate because its discount varies by model. Output length
is unknown before inference; `max_tokens` bounds the output estimate, while its
absence limits the estimate to input tokens. These estimates are not quotes
or spending limits.

Gemini uses inline requests below 20MB and switches to keyed JSONL file input
for larger batches, with a 2GB provider file limit. `plan()` includes request
and byte limits when splitting work. You can request smaller chunks with
`plan(rows, max_requests_per_batch=1000)` and use the same argument on
`submit_all()`.

## The primitives are still there

For a single batch:

```python
handle = bl.submit(rows)  # -> BatchHandle, JSON-serializable
open("job.json", "w").write(handle.to_json())
bl.wait(handle)  # poll until terminal
list(bl.results(handle))  # joined on your custom_id
bl.cancel(handle)
bl.list_jobs("groq")
```

## Your existing OpenAI batch code, pointed anywhere

```bash
pip install 'batchlane[serve]'
batchlane serve
```

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

f = client.files.create(file=open("rows.jsonl", "rb"), purpose="batch")
batch = client.batches.create(
    input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h"
)
```

Nothing above is batchlane-specific. It is the stock OpenAI SDK, and the rows
name `groq/...` or `gemini/...` models, so the batch runs on a provider the
client has never heard of. An R, JavaScript or curl client works the same way:
change `base_url` and nothing else. The test suite proves this by driving the
real `openai` package against the app rather than a client of our own.

**The gateway stores no jobs.** OpenAI's protocol is already state-passing at
the client boundary, so the `batch_id` carries the compressed handles rather
than pointing at a row: one process, no database, no migrations, nothing lost
on restart. Uploaded files do go to disk, because a file must survive until a
batch references it, and sufficiently large handle collections spill to disk when their encoded
IDs exceed the gateway limit. `GET /v1/batches` returns an
empty list, since a server holding nothing has nothing to enumerate.

**Running it exposes spending authority.** The gateway submits jobs with your
provider keys, so anyone who can reach it can spend your money. It binds
loopback by default and **refuses to serve on any other address without a
key**:

```bash
batchlane serve --host 0.0.0.0 --api-key "$(openssl rand -hex 16)"
```

Clients then send that as their `api_key`. Provider credentials stay in the
gateway's environment and never reach a client.

Note also that a `batch_id` is a bearer capability: it carries the job, so
whoever holds it can poll and read that job's results. That is what makes the
server stateless, and it is the trade being made.

A solo user never has to run any of this; the library alone is enough.

## What it will not do

### It does not emulate a batch

Batchlane does not send concurrent synchronous requests as a substitute for a
provider batch API. It raises `NoBatchLaneError` for providers classified as
having no lane and `AdapterNotShippedError` where an adapter is missing.

A provider whose lane exists but is unimplemented gets a *different* error, so
a refusal never claims a lane is absent when it is merely unwritten.

### It does not enforce a spending budget

`plan().cost` estimates cost and `actual_cost()` prices reported usage. Neither
blocks submissions when a budget is exceeded.

## Supported today

| Provider | Discount | Window | Live-verified | Notes |
|---|---|---|---|---|
| Anthropic | 50% | none | **yes** | inline requests, no file upload |
| Gemini AI Studio | 50% | none | pending | inline or file input; see joining limits below |
| OpenAI | 50% | 24h | no | the reference lane; litellm covers it too |
| Groq | 50% | 24h or 7d | no | model allowlist |
| Mistral | 50% | any Nh | no | model scoped to the job, not the line |
| Fireworks | 50% | 12h to 72h | no | dataset upload; no cancel endpoint |
| Together | up to 50% | 24h fixed | no | some models excluded from batch |
| DeepInfra | 20% | 24h | no | model must be uniform across the file |

"Live-verified" means a real batch was submitted, polled and read back, with
answers checked against their inputs. Take the others as untested: their wire
shapes have been checked line by line against each provider's own API
reference, which is not the same as evidence that they work. The adapter
module docstrings cite the source for each.

Gemini carries a hazard worth stating plainly: its docs say inline results map
to requests **by array index**, not by the key you supply. batchlane joins on
an echoed key wherever the payload carries one, falls back to submission order
otherwise, and refuses outright when the counts disagree, because a quietly
mis-joined batch attaches plausible answers to the wrong rows and nothing
about the output looks wrong.

## What is missing, and why

xAI is skipped on purpose: its lane
discounts 20% rather than 50%, and its own docs exclude the flagship models.

Azure, Vertex AI and Bedrock are unshipped because litellm already reaches
them, so batchlane points you there instead of claiming they have no lane.

Groq, Together, DeepInfra, and Fireworks are reachable through the same interface.

Self-hosted runtimes get a different answer again. Ollama, LM Studio,
llamafile and vLLM have no batch lane because there is no per-token price to
discount: the hardware is yours already. What helps there is throughput, not a
discount, so batchlane says so. For vLLM it names the command that does the
job, `vllm run-batch`, which litellm's own hosted_vllm support will not do
because it assumes an HTTP `/v1/batches` that a stock `vllm serve` does not
expose.

## Inspect a lane before relying on it

```python
>>> bl.capabilities_for("groq").window.allowed
('24h', '7d')
>>> bl.capabilities_for("groq").result_retention
datetime.timedelta(days=30)
```

The descriptor carries the asymmetries that quietly cost you a run: result
retention (Gemini keeps results 6 weeks, Groq 30 days), whether cancel exists
at all (Fireworks has no cancel endpoint), whether the window is yours to set,
and which endpoints the lane covers.

## Why not just use LiteLLM's `/batches`

Use the interface that supports the provider and account you need. Batchlane
provides its own provider adapters, request planning, and resumable submissions.
It uses LiteLLM for request and response conversion; see the
[LiteLLM batch documentation](https://docs.litellm.ai/docs/batches) for its current
provider support.

`batchlane` depends on LiteLLM the *library* and ignores LiteLLM the gateway.
One module, `translate.py`, imports it, and calls nothing but pure synchronous
transforms. A golden-output test pins their results so a version bump fails in
CI rather than corrupting a 50,000-row job.

## Install

```bash
pip install batchlane
```

Requires Python 3.12 or newer.

Credentials come from the usual environment variables (`ANTHROPIC_API_KEY`,
`GEMINI_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, `TOGETHER_API_KEY`,
`DEEPINFRA_TOKEN`), or pass `api_key=` explicitly.

Batch APIs are generally excluded from free tiers: Groq's needs the Developer
plan and Gemini's needs the paid tier. No provider offers a Stripe-style test
key, because inference costs real compute whoever is asking.

## Development

```bash
uv sync --all-groups
uv run pytest              # unit + contract, no network
uv run ruff check .
BATCHLANE_LIVE=1 uv run pytest -m live    # real API calls, real (tiny) spend
```

## License

MIT
