---
status: done
---

# Phase 1 - relock

## Overview

Move every dependency to its current version and fix what that breaks.
Context: [`00_start.md`](00_start.md).

## Plan

- `uv lock --upgrade`, holding ruff, pyright, nbstripout and detect-secrets at their current lock versions.
- `uv sync --all-extras --all-groups`.
- Review the lock diff for major-version jumps; read the release notes of each one llm-core's code touches.
- Verification: `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright`.
- Fix breakages with the smallest change; record each one below with the version that caused it.

## Done when

- The lock is current and the verification passes.
- Forced code changes and notable jumps are recorded here and in the tracking log.

## Outcome

- `uv lock --upgrade`, then `uv lock -P ruff==0.15.7 -P pyright==1.1.408` to hold the tools (nbstripout and detect-secrets did not move).
- Major jumps: openai 2 to 3, sentence-transformers 5 to 6, protobuf 6 to 7, rich 14 to 15, python-dateutil 2 to 4, torch 2.10 to 2.15 with cu13 wheels.
  langchain-core 1.2.20 to 1.8.0, langchain 1.2 to 1.5, langchain-openai 1.1 to 1.7.
- No code change was needed.
- Tests: 192 passed, before and after. Before, `test_env_vars` failed on g7 only because `~/cred/llm-core/.env` was missing; a stub from `nokeys.env` now provides it.
- ruff check and format pass.
- pyright: 63 errors before, all in `transcription/providers/`, unchanged by this phase.
  After: 67, the 5 new ones in `openai_api.py` where `**kwargs: object` meets openai 3's typed `create()` parameters, one old one there gone. Same kind as the existing ones; left for a pass over the transcription types.
- `openai.resources.audio.transcriptions.Transcriptions.create` exists in openai 3.50 with `file` and `model`; not run against the live API.
- Test warnings: the remaining ones are `asyncio.iscoroutinefunction` deprecations from chromadb and langsmith, upstream.
