---
status: in progress
---

# Phase 2 - release 0.3.0

## Overview

Publish the relocked llm-core as 0.3.0 and tell its dependents how to move.
Context: [`00_start.md`](00_start.md).

## Plan

- Raise a lower bound in `pyproject.toml` only where phase 1 found the code needs the newer version.
- `version = "0.3.0"` in `pyproject.toml`, the lock refreshed for it, a `CHANGELOG.md` entry; one release commit, as v0.2.2 was.
- `docs/guides/upgrading_0_3.md`: what changed for a dependent, the steps to move (pin `@v0.3.0`, `uv lock --upgrade`, run the suite),
  the breaking changes phase 1 found in the upstream packages, and that `@v0.2.2` keeps working until then.
  Linked from the docs nav.
- Tag `v0.3.0` locally on the release commit.
- The user pushes the branch, merges, and pushes the tag, from g4.

## Done when

- Tag `v0.3.0` exists on the release commit, and the user has pushed it.
- laife can pin `@v0.3.0` (laife folder 34, phase 2).

## Outcome so far

- `langchain-core>=1.8`, the one bound raised: it is the version that stops the pydantic V1 warning.
- `version = "0.3.0"`, lock refreshed, `CHANGELOG.md` entry, `docs/guides/upgrading_0_3.md` in the nav; `mkdocs build --strict` passes.
- laife relocked on the same stack (still pinned to v0.2.2) with no code change: `make check` and `make headless` pass.
- Left: the release commit, the local tag, the user's push.
