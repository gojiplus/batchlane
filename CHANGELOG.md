# Changelog

## [Unreleased]

## 0.1.0 — 2026-09-07

Initial PyPI release.

- Batch adapters for Anthropic, Gemini AI Studio, OpenAI, Groq, Mistral,
  Fireworks, Together, and DeepInfra.
- Request planning, provider capability checks, cost estimates, and an optional
  OpenAI-compatible HTTP gateway.
- `submit_all()` submits all chunks without waiting; `run()` submits all chunks
  before polling. Checkpoints reject changed inputs and settings.
- Gemini automatically switches from inline requests to keyed file input for
  batches at or above 20MB, and collects keyed JSONL results.
- Only Anthropic has been verified against a live provider API. Other adapters,
  including Gemini file uploads, are covered by mocked contract tests.
- Checkpoint recovery depends on provider support and does not guarantee
  exactly-once submission. Save answers locally beyond provider retention.
