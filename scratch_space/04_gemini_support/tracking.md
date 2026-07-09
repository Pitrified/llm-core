# implementation tracking

Add Google Gemini as a chat provider in llm-core via `langchain-google-genai` and wire it into
laife as the swappable default. Analysis and decisions in [`00_start.md`](00_start.md).

## Key decisions

- Go through LangChain (`init_chat_model`, provider `google_genai`), matching every existing provider;
  native `google-genai` SDK only if LangChain lags on a needed feature.
- Free tier first, default model `gemini-2.5-flash`; prepaid credits only if rate limits bite.
- Key owned by a dedicated ecosystem Google account, stored per convention in `~/cred/`, env var `GOOGLE_API_KEY`.
- Free-tier data training accepted for laife; caveat documented in llm-core docs (phase 4).
- laife selection stays a code swap of `ChatParams.default`; no runtime picking.
- Spin-offs: gpt-5 defaults update ([`05_gpt5_defaults`](../05_gpt5_defaults/00_start.md)),
  Gemini embeddings ([`06_gemini_embeddings`](../06_gemini_embeddings/00_start.md)); both draft.

## Phases

| #   | Phase                 | Plan                                                     | Status  |
| --- | --------------------- | -------------------------------------------------------- | ------- |
| 1   | research              | [`01_initial_research.md`](01_initial_research.md)       | done    |
| 2   | manual google setup   | [`02_manual_google_setup.md`](02_manual_google_setup.md) | planned |
| 3   | dependency update pass | [`03_deps_update.md`](03_deps_update.md)                | planned |
| 4   | llm-core integration  | [`04_llm_core_integration.md`](04_llm_core_integration.md) | planned |
| 5   | laife interaction     | [`05_laife_interaction.md`](05_laife_interaction.md)     | draft   |
| 6   | e2e verification      | [`06_e2e_verification.md`](06_e2e_verification.md)       | draft   |

Status values: draft / planned / in progress / done / superseded / discarded.

## Log

Append-only. Newest at the bottom.

- 2026-07-10 : phase 1 - web research on access, credentials, billing; recorded in `01_initial_research.md`
  (originally `00_start.md`). Key findings: LangChain path is `langchain-google-genai` / `google_genai`;
  free tier needs only an API key; billing default is prepaid credits, postpay only at Tier 3;
  free-tier binding limits are RPM/RPD, not the headline 1M TPM.
- 2026-07-10 : bootstrapped tracked development; renamed research file to `01_initial_research.md`,
  wrote new `00_start.md`, resolved all open questions with the user (flash default, dedicated Google
  account, code-swap selection, data training accepted), spun off draft features 05 and 06,
  scaffolded tracking.md and sub-plans for phases 2-5.
- 2026-07-10 : inserted phase 3 (dependency update pass) per user request - uv.lock dated 2026-04-30
  (langchain 1.2.13); renumbered integration/laife/e2e sub-plans to 04/05/06.
