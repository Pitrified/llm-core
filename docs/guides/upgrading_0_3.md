# Upgrading to 0.3

0.3.0 changes no llm-core API.
It moves the dependencies to current versions and requires `langchain-core>=1.8`,
so a dependent that moves to it relocks most of its own environment too.

## Why

`langchain-core` before 1.8 imports `pydantic.v1` when it is imported and warns on Python 3.14:

```
UserWarning: Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.
```

1.8 imports it lazily, and the warning is gone.

## Staying on 0.2

Nothing forces the move.
A dependent pinned to `@v0.2.2` keeps resolving to it until its own upgrade pass.

## Moving

1. Pin the new tag in `pyproject.toml`:

    ```toml
    "llm-core[all] @ git+https://github.com/Pitrified/llm-core@v0.3.0",
    ```

    Keep the extras the dependent already uses.

2. Relock everything, not only llm-core: `uv lock --upgrade`.
   A repo that pins ruff, pyright, nbstripout or detect-secrets exactly, to match its pre-commit hook revs, keeps them where they are;
   moving them is a separate step with its hook revs.
3. Sync and run the repo's full check.

## What can break

The relock crosses these major versions among the packages llm-core pulls in:

- `openai` 2 to 3. llm-core's own call, `client.audio.transcriptions.create` in the OpenAI transcription provider, still exists with the arguments llm-core passes; it has not been run against the live API.
  Code that calls the `openai` client directly should check its own calls.
- `sentence-transformers` 5 to 6. `transformers` is at 5; a dependent whose own lock still has 4 crosses that major too, as laife did.
- `torch` 2.10 to 2.15, with the CUDA wheels moving from `cu12` to `cu13` packages: a large download on the first sync.
- `protobuf` 6 to 7, `rich` 14 to 15, `python-dateutil` 2 to 4.

laife moved with no code changes.

## Warnings that remain

`chromadb` and `langsmith` still call `asyncio.iscoroutinefunction`, deprecated in Python 3.14, and a test run shows a `DeprecationWarning` for each.
Both are upstream.
