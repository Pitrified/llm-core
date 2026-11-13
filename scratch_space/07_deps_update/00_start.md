---
status: in progress
---

# feat 07 - dependency update and release 0.3.0

Refresh llm-core's dependencies to current, release 0.3.0, and tell the dependents what to do.
Tracked development folder; the index is [`tracking.md`](tracking.md).

Split out of [`04_gemini_support`](../04_gemini_support/00_start.md), where it was phase 3
([`03_deps_update.md`](../04_gemini_support/03_deps_update.md), now superseded by this folder).
Asked for by laife, whose folder 34 needs a langchain-core that does not warn about pydantic V1 on Python 3.14.

## current state

- `uv.lock` was last regenerated 2026-04-30 (langchain-core 1.2.20 here, 1.2.14 in laife's lock).
- `pyproject.toml` constraints are loose lower bounds (`langchain-core>=0.3` and so on).
- langchain-core 1.8.0 imports `pydantic.v1` lazily and no longer warns on import under 3.14
  (checked 2026-11-13 in a scratch venv, `-W always`).
- A dry run of a full upgrade in laife (2026-11-13) shows major jumps in packages llm-core pulls in:
  openai 2 to 3, transformers 4 to 5, sentence-transformers 5 to 6, protobuf 6 to 7,
  the torch CUDA wheels from cu12 to cu13.

## decisions 2026-11-13

Taken with the user.

- The dependency update is its own effort, not a phase of the Gemini support; Gemini is deferred.
- The release is 0.3.0: dependents may break. They keep their old pin until their own upgrade pass; 0.x is for this.
- Lower bounds in `pyproject.toml` are raised only where llm-core's code needs the newer version, not to every version the new lock was tested with.
- A guide in `docs/guides/` tells dependents what changed and how to move.
- The pushes, of the branch and of the tag, are the user's, from g4.
- The exactly pinned dev tools (ruff, pyright, nbstripout, detect-secrets) are held at their current lock versions:
  this pass is about what dependents install, and the tools' drift from their hook revs is a separate, cross-repo item.

## phases

1. **relock** - [`01_relock.md`](01_relock.md): `uv lock --upgrade`, suite green, breakages fixed and recorded.
2. **release** - [`02_release.md`](02_release.md): lower bounds, 0.3.0 version and changelog, dependents guide, tag.
