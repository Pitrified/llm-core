---
status: done
---

# Phase 1 - initial research

Originally the feature's `00_start.md`; renamed when the folder became tracked development.
Serves as phase 1's record. Index: [`tracking.md`](tracking.md).

## draft

https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemini-cli-open-source-ai-agent/
https://pasqualepillitteri.it/en/news/6683/gemini-2-5-flash-free-1m-tokens

do a web search for all these claims

1. how do you access those 1M token per minute via python?
2. which credentials do you need? can it be api keys?
3. which is the billing model? you load money on that account and then the api key uses the credits, or will you receive an invoice at the end of the month?

## analysis

Researched 2026-07-10 via web search. The two draft links describe two different products:
the Gemini CLI (an open-source terminal agent, free via Google account OAuth, no API key involved)
and the Gemini API free tier (API key, this is where the "1M tokens per minute" figure comes from).
For Python integration we care about the API, not the CLI.

### 1. Accessing the tokens via Python

Official SDK is `google-genai` (the old `google-generativeai` is deprecated).

```python
from google import genai

client = genai.Client()  # picks up GEMINI_API_KEY or GOOGLE_API_KEY from env
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Why is the sky blue?",
)
print(response.text)
```

Caveat on the "1M tokens per minute" claim: the TPM figure is real but almost never the binding limit.
The free tier is also capped per requests per minute and per day
(figures reported for 2.5 Flash range from 10-15 RPM and 250-1500 RPD depending on period and account;
current values are shown per-account at https://aistudio.google.com/rate-limit).
The article itself admits "the million per minute just sits there, untouched".
Some 2026 reports say free TPM for 2.5 Flash was later reduced to 250K.
Limits are per Google Cloud project (not per API key), so multiple keys do not add quota.

### 2. Credentials

Yes, plain API keys work. Get one from Google AI Studio (https://aistudio.google.com) with just a Google account;
no credit card is needed for the free tier.
The SDK reads it from the `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) env var, or pass `api_key=` to `genai.Client`.
Vertex AI is the alternative path (service account / ADC credentials) but is not needed for this use case.

Free-tier trade-off: outside EU/UK/CH, Google may use free-tier prompts and responses to improve its models.
Paid tier (and EU users) are excluded from training.

### 3. Billing model

Both models exist, and the answer changed in 2026: **prepaid is now the default**.

- Free tier: no billing account at all, just the API key.
- Paid (Tier 1+): you link a Cloud Billing account in AI Studio and **load prepaid credits**
  (min $10, max $5,000; credits expire after 1 year, non-refundable; optional auto-reload).
  The API key then burns those credits. Exactly the "load money, key uses credits" model.
- Monthly invoice (postpay) still exists but only for Tier 3 accounts
  (requires $1,000 paid + 30 days of history), as an opt-in switch.
- Tiers: Tier 1 = billing account linked ($250/month spend cap),
  Tier 2 = $100 paid + 3 days, Tier 3 = $1,000 paid + 30 days. Higher tiers raise rate limits.

Pricing for 2.5 Flash (standard tier, per 1M tokens): $0.30 input (text), $2.50 output;
batch tier halves that. The draft article's $0.15/$0.60 figures are outdated.

Sources:

- https://ai.google.dev/gemini-api/docs/rate-limits
- https://ai.google.dev/gemini-api/docs/pricing
- https://ai.google.dev/gemini-api/docs/billing
- https://ai.google.dev/gemini-api/docs/quickstart
- https://googleapis.github.io/python-genai/
- https://blog.google/innovation-and-ai/technology/developers-tools/prepay-gemini-api/
- https://pasqualepillitteri.it/en/news/6683/gemini-2-5-flash-free-1m-tokens

