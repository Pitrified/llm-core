---
status: draft
---

# feat 06 - gemini embeddings support

## draft

Spun off from [`04_gemini_support`](../04_gemini_support/00_start.md), which covers chat only.

Add a Google embeddings config to `src/llm_core/embeddings/config/`, mirroring the existing
embeddings provider pattern, backed by `GoogleGenerativeAIEmbeddings` from `langchain-google-genai`
(dependency already added by feature 04 phase 4; same `GOOGLE_API_KEY` credential).

Pick up only when a consumer actually needs it; until then this stays a draft note.

## analysis
