# Refactor Stage Plan

## Scope
This branch refactors the pipeline into two explicit phases:
1) **Transcription** (audio -> SRT only)
2) **Logic modeling** (SRT -> S-ORAL graph -> cleaned SRT)

## Goals (Agreed)
### Phase 1: Transcription
1. Extract/convert audio to a stable working format.
2. VAD-driven chunking with merge targets:
   - `--target-sec 20`
   - `--max-sec 35`
3. Language detection using VoxLingua107 (SpeechBrain) on sampled chunks.
   - Sample positions: head, 25%, 50%, 75%, tail (5 samples).
   - Vote by score to determine dominant language.
4. Use the detected language to call Deepgram (Nova-2) in parallel.
   - Concurrency unchanged (default 64).
   - Retry once on failure.
5. Output **only** a standard SRT at: `out/<base>.srt`.
   - No speaker labels.
   - No additional files (no GCL, no TXT).

### Phase 2: Logic Modeling
1. Load `prompts/SORAL.txt`.
2. Feed the Phase 1 SRT to **GPT-5.2** once to generate the full object graph.
   - Output file format: `.txt`.
   - Prompt caching not required for this single call.
3. Load the object graph file.
   - Allow loading an external object graph via parameter.
   - Example: when processing Episode 2, reuse Episode 1 graph as context.
4. Output the merged object graph as the final Phase 2 artifact.

## OpenAI API Notes (Per References)
- Use **Responses API** for both GPT-5.2 and GPT-5-mini calls.
- Prompt caching:
  - Place static prompt content first.
  - Use `prompt_cache_key` consistently per chunk.

## Deliverables
- Refactored CLI with explicit Phase 1 and Phase 2 commands.
- Deterministic file locations and formats as above.
- Minimal logs; console output stays concise.

## Non-Goals (Explicit)
- No translation or language conversion.
- No speaker labels in output SRT.
- No multi-language mixing fixes (assume single dominant language).
