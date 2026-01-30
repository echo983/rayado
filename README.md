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

## Key Documents
- `docs/spec/SPEC.md` system specification
- `docs/spec/GCL_TEMPLATE.md` GCL minimal and extended templates
- `docs/spec/CACHE_SCHEMA.md` cache storage schema
- `docs/spec/CLI.md` CLI contract

## Status
Design complete; entering implementation milestones.
