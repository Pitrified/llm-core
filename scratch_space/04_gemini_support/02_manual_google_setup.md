---
status: planned
---

# Phase 2 - manual google setup

## Overview

Human-in-the-loop phase: get a working Gemini API key on the free tier, owned by a dedicated
ecosystem Google account, and prove it works before any llm-core code is written.
Context: [`00_start.md`](00_start.md), findings in [`01_initial_research.md`](01_initial_research.md).

## Goals

1. A dedicated ecosystem Google account exists and owns an AI Studio project.
2. A Gemini API key on the free tier (no billing account), stored per convention.
3. Key verified with a real call to `gemini-2.5-flash`.

## Plan

- User creates (or designates) the dedicated ecosystem Google account.
- User generates an API key in AI Studio (https://aistudio.google.com), free tier, no billing account.
- Store the key as `GOOGLE_API_KEY` in `~/cred/llm_core/.env` (loaded by `load_env()`).
- Confirm the account's region and the resulting data-training stance; note it here for phase 4 docs.
- Smoke-test the key with a one-off script in `scratch_space/04_gemini_support/`
  (native `google-genai` SDK or raw REST is fine here; it is throwaway).
- Note the actual rate limits shown for this account at https://aistudio.google.com/rate-limit.

## Out of scope

- Any llm-core source changes (phases 3-4).
- Billing setup / prepaid credits.

## Done when

- The smoke-test script gets a non-error completion from `gemini-2.5-flash` using the stored key.
- Region, data-training stance, and observed rate limits are recorded in this file.
