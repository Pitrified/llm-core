---
status: draft
---

# Phase 6 - end-to-end verification

## Overview

Run a real laife flow on Gemini and record how it behaves beyond unit tests:
rate-limit errors, tool calling, structured output quirks.
Context: [`00_start.md`](00_start.md), depends on [`05_laife_interaction.md`](05_laife_interaction.md).

## Goals

1. Confidence that Gemini works in a real laife flow, or a recorded list of what breaks.

## Plan (to detail when phase starts)

- Decide the probe here (deferred from 00_start): single player-brain turn vs full world-runner tick.
  Full tick exercises tool calling and structured output harder but burns more of the free RPM/RPD budget;
  use the actual limits recorded in phase 2 to decide.
- Run the chosen flow with `ChatParams.default = self.google`.
- Watch for: 429s under the free-tier RPM, structured-output/tool-calling deviations vs OpenAI,
  latency differences.
- Record findings here and in the tracking log, including failures - they are the point of this phase.

## Out of scope

- Fixing laife logic bugs unrelated to the provider swap.
- Moving to paid tier (only note if the free budget proves unusable).

## Done when

- The chosen flow completed (or its failure modes are documented) and findings are logged.
- A recommendation is recorded: keep Gemini as an option, make it default, or park it.
