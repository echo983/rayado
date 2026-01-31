# Rayado

High‑throughput CLI transcription pipeline for episodic media. The system is optimized for LLM consumption, cache efficiency, and replayable logs with minimal manual intervention.

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

## Usage (Refactor)
- Install locally (editable): `pip install -e .`
- Set API key: `setx DEEPGRAM_API_KEY "<your_key>"`
- Set OpenAI key: `setx OPENAI_API_KEY "<your_key>"`
- Phase 1 (audio -> SRT only): `rayado phase1 <input>`
  - Output: `out/<base>.srt`
- Phase 2 (SRT -> object graph): `rayado phase2 <srt>`
  - Output: `out/<base>.graph.txt`
  - Optional external graph: `--graph-in <path>`

## E2E Test
- Test file: `docs/testfiles/test.webm` (not tracked; add your own)
- Run: `python scripts/run_e2e.py` (uses smaller chunk size and disables diarization)
- Debug Deepgram directly: `python scripts/debug_deepgram.py`
- VAD chunk dump (speech-only WAVs): `python scripts/vad_chunk_dump.py "<input>" --target-sec 20 --max-sec 35`

## Status
Milestone 1 in progress: end‑to‑end CLI + GCL + TXT/SRT output.
