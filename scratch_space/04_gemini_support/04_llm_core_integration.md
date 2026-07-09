---
status: planned
---

# Phase 4 - llm-core integration

## Overview

Add Gemini as a chat provider in llm-core, mirroring the existing provider pattern exactly.
Context: [`00_start.md`](00_start.md), depends on [`02_manual_google_setup.md`](02_manual_google_setup.md)
for a working key and on [`03_deps_update.md`](03_deps_update.md) for a current LangChain (tests that hit the network, if any, follow the repo's existing testing conventions).

## Goals

1. `GoogleGenAIChatConfig` usable via `create_chat_model()` like every other provider.
2. Dependency, exports, tests, and docs all updated.

## Plan

- Add `langchain-google-genai` to `pyproject.toml` (verify current version and that
  `init_chat_model` dispatches provider string `google_genai` to it).
- Add `src/llm_core/chat/config/google_genai.py`: `GoogleGenAIChatConfig(ChatConfig)` with
  `model = "gemini-2.5-flash"`, `model_provider = "google_genai"`,
  `api_key: SecretStr | None` via `secret_from_env("GOOGLE_API_KEY", default=None)` -
  same shape as `ChatOpenAIConfig`.
- Export it from `src/llm_core/chat/__init__.py`.
- Tests in `tests/` mirroring the other chat config tests.
- Docs: add the provider to the relevant `docs/library/` page; include the free-tier caveats
  (data training outside EU on free tier; binding limits are RPM/RPD, not TPM).

## Out of scope

- Embeddings (draft feature [`06_gemini_embeddings`](../06_gemini_embeddings/00_start.md)).
- laife changes (phase 5).
- Native `google-genai` SDK usage.

## Done when

- `uv run pytest && uv run ruff check . && uv run pyright` all pass.
- `GoogleGenAIChatConfig().create_chat_model()` returns a model that answers a prompt
  with the phase 2 key (manual check is fine).
- Docs updated, including the free-tier caveat note.
