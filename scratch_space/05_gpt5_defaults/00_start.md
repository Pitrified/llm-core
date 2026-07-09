---
status: draft
---

# feat 05 - update openai defaults to gpt-5 family

## draft

Spun off from [`04_gemini_support`](../04_gemini_support/00_start.md) while deciding the Gemini
default model: llm-core's OpenAI chat config still defaults to the gpt-4 family
(`ChatOpenAIConfig.model = "gpt-4o-mini"` in `src/llm_core/chat/config/openai.py`).

Update the defaults to the gpt-5 family, keeping the "cheap default" spirit
(pick the gpt-5 equivalent of a mini model).
Check other spots that hardcode gpt-4-era names: docs, sample configs, tests,
and consumers that override the model (e.g. laife's `ChatParams` pins `gpt-5.2-chat` on Azure already).

## analysis
