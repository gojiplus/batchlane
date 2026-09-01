# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `run()`, `wait()` and `plan()`: split a job to fit the provider's caps,
  submit the pieces, wait, and stream each input row beside its answer.
  With `checkpoint=` set, a crash resumes against the jobs already submitted
  rather than re-running inference.
- Batch lanes for Anthropic, Gemini AI Studio, OpenAI, Groq, Together and
  DeepInfra.
- `capabilities_for()`, describing each lane precisely enough to gate on:
  result retention, whether cancel exists, whether the completion window is
  the caller's to set, and per-endpoint support.

### Notes

- Gemini AI Studio is the gap this package exists to close. LiteLLM can batch
  Gemini models through Vertex AI but not through AI Studio, so the same model
  behind the same 50% discount is reachable with GCP credentials and
  unreachable with a `GEMINI_API_KEY`.
- Azure, Vertex AI and Bedrock are deliberately unshipped because LiteLLM
  already reaches them; batchlane says so rather than reporting no lane.
- Only Anthropic is live-verified end to end. Treat the other lanes as
  untested against a real API.
