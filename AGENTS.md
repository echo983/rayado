# Repository Guidelines

## Project Structure & Module Organization
This repository is currently documentation-only. The root contains three text documents: `资料1.txt`, `资料2.txt`, and `GCL.txt`. These appear to describe the GCL pipeline and related processing stages (e.g., AudioPrep, ChunkPlan, ASRPool, CoherenceJobs, RenderExport). There are no source-code, test, or configuration directories at this time.

## Build, Test, and Development Commands
No build, run, or test commands are defined in this repository. If you add code later, document commands here (for example, `npm run build`, `pytest`, or `make test`) and keep them aligned with CI expectations.

## Coding Style & Naming Conventions
There is no code style enforced yet. For documentation edits:
- Preserve the existing language and terminology in the GCL docs.
- Keep headings short and descriptive.
- Avoid reflowing large sections unless the meaning is unchanged.
- The text files appear to use a non-UTF-8 encoding; verify encoding before editing to prevent garbling. If you convert encoding, note it explicitly in the change description.

## Testing Guidelines
No tests are present. If tests are added, include the framework, coverage targets, and naming pattern (e.g., `tests/test_*.py`) in this section.

## Commit & Pull Request Guidelines
This repository is not a Git repository, so no commit history is available. If you initialize Git, adopt clear, imperative commit messages (e.g., “Add GCL rendering notes”) and include:
- A concise summary of changes
- Rationale for content changes
- Links to any related issues or discussions

## Security & Configuration Tips
No secrets or runtime configuration are stored here. Do not add credentials or API keys to documentation files.

## Agent-Specific Instructions
Keep changes minimal and focused. When updating docs, cite the specific file you modified (e.g., `GCL.txt`) and describe the section you touched.
