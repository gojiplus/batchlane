# batchlane

Submit asynchronous batch jobs to whichever discount lane a provider actually
runs — and get an honest refusal where none exists.

Most providers sell latency-insensitive inference at a discount: OpenAI,
Anthropic, Gemini, Groq, Mistral, Together, Fireworks and DeepInfra all run an
asynchronous lane, usually at 50% off. LiteLLM, which most people route
through, can only reach six of them. Point it at Groq or Gemini AI Studio for
bulk work and you quietly pay double.

`batchlane` reaches the rest. It borrows LiteLLM's per-provider request and
response translation — the hard, maintained part — and owns the job lifecycle
itself.

```python
import batchlane as bl

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
    print(line.custom_id, result.response["choices"][0]["message"]["content"])
```

One call splits the job to fit the provider's caps, submits however many
batches that takes, waits, and hands back **each input row beside its answer**
— so there is nothing to join on your side. It streams, so a 50,000-row job
never sits in memory.

### Resuming does not re-pay

With `checkpoint=` set, every submitted job is recorded before anything that
could fail. If the process dies, rerun the identical call: batchlane
re-attaches to the jobs it already submitted and re-reads their output rather
than running inference again. That works because providers retain results for
weeks (Anthropic 29 days, Gemini 6 weeks, Groq 30 days), so the checkpoint
stores *handles*, not results. Fetching is free; inference is not.

### Will it fit?

```python
>>> p = bl.plan(rows)
>>> p.n_chunks, p.total_bytes, p.limit_bytes
(1, 423, 209715200)
```

Gemini's inline lane caps at 20MB — an order of magnitude below the
file-based providers and with no request cap at all — so a job that is one
batch on Groq can be a dozen on Gemini. `plan()` tells you before you spend.

batchlane deliberately does **not** try to pick a chunk size for faster
turnaround. That depends on the provider's queue depth, which is invisible and
volatile: the same Anthropic account returned one batch in 200 seconds and the
next in over ten minutes. Fitting the caps is well defined; optimising latency
would be a guess dressed as a feature.

### The primitives are still there

`run()` is built on them, not instead of them:

```python
handle = bl.submit(rows)  # -> BatchHandle, JSON-serialisable
open("job.json", "w").write(handle.to_json())
bl.wait(handle)  # the poll loop nobody should write twice
list(bl.results(handle))  # joined on your custom_id
bl.cancel(handle)
bl.list_jobs("groq")
```

## What it will not do

**It does not emulate a batch.** A discount exists because the provider
backfills otherwise-idle GPUs, which requires owning a fleet with a demand
trough. A reseller buying capacity at retail structurally cannot offer one.
Running your requests concurrently against a synchronous endpoint and calling
the result a "batch" would save nothing while implying half price, isolated
rate limits and a 24-hour window. So it refuses, and says why:

```python
>>> bl.get_adapter("openrouter")
NoBatchLaneError: 'openrouter' has no asynchronous batch lane: It resells
upstream capacity at retail, so it has no idle fleet to backfill and cannot
price a discount lane. Providers with a lane: deepinfra, groq, together_ai.
```

A provider whose lane exists but is unimplemented gets a *different* error, so
a refusal never claims a lane is absent when it is merely unwritten.

**It does not track cost.** No spend gates, no estimates, no database. Use the
gateway you already have.

## Supported today

| Provider | Discount | Window | Live-verified | Notes |
|---|---|---|---|---|
| Anthropic | 50% | none | **yes** | inline requests, no file upload |
| Gemini AI Studio | 50% | none | pending | inline only; model lives in the URL |
| OpenAI | 50% | 24h | no | the reference lane; litellm covers it too |
| Groq | 50% | 24h or 7d | no | model allowlist |
| Together | up to 50% | 24h fixed | no | some models excluded from batch |
| DeepInfra | 20% | 24h | no | model must be uniform across the file |

"Live-verified" means a real batch was submitted, polled and read back, with
answers checked against their inputs. Take the others as untested.

Planned: Mistral, Fireworks. Deliberately skipped: xAI — its lane discounts
20% and its own docs exclude the flagship models. Azure, Vertex AI and Bedrock
are unshipped because litellm already reaches them; batchlane says so rather
than pretending they have no lane.

**Gemini AI Studio is the gap worth knowing about.** LiteLLM can batch Gemini
models through Vertex AI but not through AI Studio — `batches/main.py` never
mentions `gemini`, and there is no `llms/gemini/batches/`. So
`vertex_ai/gemini-2.5-flash` batches fine while `gemini/gemini-2.5-flash`
cannot, despite being the same model behind the same 50% discount.

There is one hazard in that lane worth stating plainly: Gemini documents that
inline results map to requests **by array index**, not by the key you supply.
batchlane joins on an echoed key wherever the payload carries one, falls back
to submission order otherwise, and refuses outright if the counts disagree —
because a silently mis-joined batch is the worst thing this package could
produce.

Anthropic is worth calling out because LiteLLM can *retrieve* an Anthropic
batch but not create one — `litellm.create_batch(custom_llm_provider="anthropic")`
raises `BadRequestError: LiteLLM doesn't support custom_llm_provider=anthropic
for 'create_batch'`. So bulk Claude work routed through the gateway pays full
price today.

Inspect any provider's lane before relying on it:

```python
>>> bl.capabilities_for("groq").window.allowed
('24h', '7d')
>>> bl.capabilities_for("groq").result_retention
datetime.timedelta(days=30)
```

The descriptor carries the asymmetries that silently cost you a run if ignored:
result retention (Gemini keeps results 6 weeks, Groq 30 days), whether cancel
exists at all (Fireworks has no cancel endpoint), whether the window is yours
to set, and which endpoints the lane covers.

## Why not just use LiteLLM's `/batches`

You should, for the six providers it reaches. For the rest there is no path:
its dispatch is a hardcoded `if/elif` chain, its provider registry has one
entry, and `litellm.CustomLLM` exposes no batch hooks, so nothing can be added
from outside. Its `create_batch` also types `completion_window` as
`Literal["24h"]`, which cannot express Groq's 7-day window.

`batchlane` depends on LiteLLM the *library* and ignores LiteLLM the gateway.
One module, `translate.py`, imports it, and calls nothing but pure synchronous
transforms. A golden-output test pins their results so a version bump fails in
CI rather than corrupting a 50,000-row job.

## Install

```bash
pip install batchlane
```

Credentials come from the usual environment variables (`ANTHROPIC_API_KEY`,
`GEMINI_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, `TOGETHER_API_KEY`,
`DEEPINFRA_TOKEN`), or pass `api_key=` explicitly.

Note that batch APIs are generally **excluded from free tiers** — Groq's needs
the Developer plan, Gemini's needs the paid tier. There is no sandbox key in
this space: nobody can fake inference.

## Development

```bash
uv sync --all-extras
uv run pytest              # unit + contract, no network
uv run ruff check .
BATCHLANE_LIVE=1 uv run pytest -m live    # real API calls, real (tiny) spend
```

## License

MIT
