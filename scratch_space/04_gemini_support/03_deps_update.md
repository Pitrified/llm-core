---
status: planned
---

# Phase 3 - dependency update pass

## Overview

Refresh llm-core's dependencies to latest before adding the new provider, so phase 4 builds on a
current LangChain rather than a ~3-month-old lock. `uv.lock` was last regenerated 2026-04-30
(langchain 1.2.13, langchain-core 1.2.20, langchain-openai 1.1.11, pydantic 2.12.5);
`pyproject.toml` constraints are loose (`>=`), so this is a re-lock plus verification,
not a constraint rewrite unless something requires it.
Context: [`00_start.md`](00_start.md); independent of phase 2, sequenced before
[`04_llm_core_integration.md`](04_llm_core_integration.md).

## Goals

1. All llm-core dependencies at current versions, lock regenerated, suite green.

## Plan

- `uv lock --upgrade` then `uv sync --all-extras --all-groups` (check the repo's actual sync flags).
- Review the lock diff for major-version jumps (langchain 1.x line, pydantic, chromadb);
  read release notes for anything that crossed a major boundary.
- Run the full verification suite: `uv run pytest && uv run ruff check . && uv run pyright`.
- Fix breakages caused by the upgrades; keep fixes minimal and note them here.
- Only touch `pyproject.toml` constraints if an upgrade forces it (record the reason).

## Out of scope

- Adding `langchain-google-genai` (phase 4).
- Refactors beyond what upgrades force.
- laife's own dependency refresh (happens in phase 5 when it bumps llm-core anyway).

## Done when

- Lock is regenerated to latest and the full verification suite passes.
- Any forced code changes and notable version jumps are recorded here and in the tracking log.
