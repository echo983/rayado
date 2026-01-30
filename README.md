# Rayado

High‑throughput, append‑only CLI transcription pipeline for episodic media. The system is optimized for LLM consumption, cache efficiency, and replayable logs with minimal manual intervention.

## Goals
- LLM‑friendly outputs with stable, replayable structure
- Cost control via prompt caching and idempotent requests
- Single‑pass throughput with strict retry policy
- Append‑only GCL as the single source of truth

## Repository Structure
- `AGENTS.md` contributor guidelines
- `docs/spec/` design specs and templates
- `docs/references/` source notes and background material
- `src/rayado/` CLI and pipeline code

## Key Documents
- `docs/spec/SPEC.md` system specification
- `docs/spec/GCL_TEMPLATE.md` GCL minimal and extended templates
- `docs/spec/CACHE_SCHEMA.md` cache storage schema
- `docs/spec/CLI.md` CLI contract

## MVP Usage
- Install locally (editable): `pip install -e .`
- Set API key: `setx DEEPGRAM_API_KEY "<your_key>"`
- Run: `rayado transcribe <input>`
- Default provider is `deepgram` (Nova‑2). Use `--asr-provider mock` or `noop` for dry runs.
- Outputs: `out/<basename>/episode.gcl`, `transcript.txt`, `subtitles.srt`

## E2E Test
- Test file: `docs/testfiles/test.webm`
- Run: `python scripts/run_e2e.py`

## Status
Milestone 1 in progress: end‑to‑end CLI + GCL + TXT/SRT output.
