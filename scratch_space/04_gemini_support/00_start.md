# feat 04 - gemini support

Add Google Gemini as a chat provider in llm-core, then wire it into laife as a selectable provider.

Tracked development folder; the index is [`tracking.md`](tracking.md).
Initial web research (rate limits, credentials, billing) already done in
[`01_initial_research.md`](01_initial_research.md).

## the idea

Gemini's free tier (API key from AI Studio, no credit card) and cheap Flash pricing make it a good
extra provider for llm-core consumers, laife first.
The integration should follow the existing provider pattern, not invent a new one.

## current state

- llm-core chat providers are `ChatConfig` subclasses in `src/llm_core/chat/config/`
  (openai, azure_openai, huggingface, ollama), each a thin Pydantic config dispatched through
  LangChain's `init_chat_model` via `model_provider`.
  Secrets use `SecretStr` + `secret_from_env` (e.g. `OPENAI_API_KEY`).
- laife holds a `ChatParams` class (`src/laife/params/llm_services/chat.py`) that instantiates one
  config per provider and exposes `self.default: ChatConfig`; switching provider = swapping `default`.
- LangChain supports Gemini through the `langchain-google-genai` package,
  provider string `google_genai`, env var `GOOGLE_API_KEY` (the native SDK also accepts `GEMINI_API_KEY`).

## decisions

- Go through LangChain (`langchain-google-genai` + `init_chat_model`), not the native `google-genai` SDK,
  to match every existing provider in llm-core. The native SDK remains an option for features
  LangChain lags on, but that would be a separate effort.
- Free tier first: API key from AI Studio, no billing account.
  Prepaid credits only if rate limits (10-15 RPM on free) actually bite in laife usage.
- Credentials follow the ecosystem convention: `~/cred/<package>/.env` loaded by `load_env()`,
  key exposed as `GOOGLE_API_KEY`.
- Default model is `gemini-2.5-flash`: cheap default, same spirit as `gpt-4o-mini`.
- Free-tier data training is acceptable for laife content; the caveat gets documented in llm-core docs
  (phase 4) rather than forcing paid tier.
- laife provider selection stays a code swap of `ChatParams.default`; no runtime picking mechanism.
- Embeddings support moves out of this feature entirely: tracked as draft feature
  [`06_gemini_embeddings`](../06_gemini_embeddings/00_start.md).
- Side note spun off during review: llm-core's OpenAI defaults still point at the gpt-4 family;
  tracked as draft feature [`05_gpt5_defaults`](../05_gpt5_defaults/00_start.md).

## open questions

- Which default model for the config: `gemini-2.5-flash` (cheap, free tier) or a pro model?
  Leaning flash, matching the "cheap default" spirit of `gpt-4o-mini`.
  ANS: cheap default
  ANS: create a draft feature 05/00_start with a note to update gpt from 4 to 5 family. draft status
- Free-tier data training: outside EU Google trains on free-tier prompts.
  Is that acceptable for laife content, or does it push us to paid tier sooner? (EU account may be exempt.)
  ANS: not an issue for laife. note it in llm core docs.
- Does laife need runtime provider picking (env var / config flag) or is swapping
  `ChatParams.default` in code enough for now?
  ANS: code swap is enough for now
- Should embeddings (`GoogleGenerativeAIEmbeddings`) ride along, or stay out of scope? Proposed: later phase, draft only.
  ANS: later phase, create a draft feature 06/00_start with a brief note. draft status

New (surfaced while folding in the answers above):

- Phase 5: which laife flow is the e2e probe - a single player-brain turn, or a full world-runner tick?
  A full tick exercises tool calling and structured output harder but burns more of the 10-15 RPM free budget.
  ANS: decide when writing the e2e verification sub-plan (now phase 6).
- Phase 2: which Google account owns the key - personal account, or a dedicated one for the box ecosystem?
  Affects where the AI Studio project lives and who can rotate the key.
  ANS: dedicated ecosystem account.

## phases (draft)

1. **research** - web research on access, credentials, billing.
   Done; recorded in [`01_initial_research.md`](01_initial_research.md).
   Remaining: verify `langchain-google-genai` current version and `init_chat_model` dispatch.
2. **manual google setup** - human-in-the-loop: create the dedicated ecosystem Google account,
   generate the API key in AI Studio, store it under `~/cred/`, confirm region/data-training stance,
   smoke-test the key with a one-off script in `scratch_space/`.
3. **dependency update pass** - refresh llm-core deps to latest (`uv.lock` is ~3 months old);
   re-lock, review major jumps, verification suite green. Sequenced before the integration so the
   new provider lands on a current LangChain.
4. **llm-core integration** - add `langchain-google-genai` dependency, `GoogleGenAIChatConfig`
   in `src/llm_core/chat/config/google_genai.py`, export it, tests mirroring the other providers,
   docs update. Full verification suite green.
5. **laife interaction** - add the Gemini config to laife's `ChatParams` and set it as the swappable
   `default` (decided: code swap, no runtime picking), update laife docs.
6. **end-to-end verification** - run a real laife flow (player brain / world runner) on Gemini,
   watch for rate-limit errors and response-format quirks (tool calling, structured output),
   record findings. Cheap insurance that phases 4+5 work beyond unit tests.

Embeddings support was proposed as a phase but spun off to draft feature
[`06_gemini_embeddings`](../06_gemini_embeddings/00_start.md) instead.

## notes

- `01_initial_research.md` predates this tracked setup; it carries `status: done` frontmatter
  and stands as phase 1's record.
