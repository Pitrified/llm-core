# implementation tracking

Refresh llm-core's dependencies, release 0.3.0, and guide the dependents.
Analysis and decisions in [`00_start.md`](00_start.md).

## Key decisions

- Split out of the Gemini support, which is deferred.
- Release 0.3.0; dependents keep `@v0.2.2` until their own upgrade pass.
- Lower bounds raised only where the code needs it.
- Dev tools held at their current lock versions.

## Phases

| #   | Phase   | Plan                               | Status      |
| --- | ------- | ---------------------------------- | ----------- |
| 1   | relock  | [`01_relock.md`](01_relock.md)     | done        |
| 2   | release | [`02_release.md`](02_release.md)   | in progress |

Status values: draft / planned / in progress / done / superseded / discarded.

## Log

Append-only. Newest at the bottom.

- 2026-11-13 : split out of `04_gemini_support` phase 3 with the user; decisions in `00_start.md`.
- 2026-11-13 : phase 1 done: relocked, no code change, tests and ruff pass, pyright errors in the transcription providers 63 to 67, all pre-existing in kind.
- 2026-11-13 : phase 2 prepared: version 0.3.0, `langchain-core>=1.8`, changelog, dependents guide; laife verified on the same stack. Commit and tag pending.
