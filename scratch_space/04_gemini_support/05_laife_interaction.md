---
status: draft
---

# Phase 5 - laife interaction

## Overview

Make Gemini selectable in laife through the existing `ChatParams` pattern.
Context: [`00_start.md`](00_start.md), depends on [`04_llm_core_integration.md`](04_llm_core_integration.md).
Decided in 00_start: selection stays a code swap of `ChatParams.default`; no runtime picking.

## Goals

1. laife can run on Gemini by swapping one line.

## Plan (to detail when phase starts)

- Bump laife's llm-core dependency to the version carrying `GoogleGenAIChatConfig`.
- Add `self.google = GoogleGenAIChatConfig()` to `ChatParams.load_params()`
  in `src/laife/params/llm_services/chat.py`.
- Ensure `GOOGLE_API_KEY` reaches laife's env (`~/cred/laife/.env` or shared cred loading - check
  how laife's `load_env()` sources overlap with llm-core's).
- Update laife docs on available providers and how to swap `default`.

## Out of scope

- Runtime/env-driven provider selection.
- Actually switching laife's default to Gemini permanently (phase 6 informs that call).

## Done when

- With `default = self.google`, laife's verification suite passes and a basic chat call works.
