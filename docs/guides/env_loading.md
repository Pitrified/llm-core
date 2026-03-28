# Environment Loading

This guide explains when and where to call `load_env()`, and why libraries
must never do it automatically on import.

## The rule

**Only the top-level application calls `load_env()`.** Library packages
(including `llm-core` itself) must not call it in `__init__.py` or anywhere
else that runs on import.

## Why

`load_env()` reads `~/cred/llm-core/.env` and merges its values into the
process environment via `python-dotenv`. This is a side effect: it mutates
global state the moment the package is imported.

When a consuming application (e.g. `kit-hub`) depends on `llm-core`, it
calls `load_env()` once, from its own `__init__.py`, to load its own
credentials. If `llm-core.__init__` also calls `load_env()`, the application
gets two credential files merged together - or a confusing log line about a
file it did not ask to load:

```
DEBUG | kit_hub.params.load_env:load_env:15 - Loaded environment variables from /home/pmn/cred/kit-hub/.env
DEBUG | llm_core.params.load_env:load_env:15 - Loaded environment variables from /home/pmn/cred/llm-core/.env
```

Libraries that auto-load env on import are impossible to use cleanly inside
any application that manages its own credentials.

## Where `load_env()` belongs

| Context | Call `load_env()`? | Location |
|---|---|---|
| Application `__init__.py` | Yes | `src/my_app/__init__.py` |
| Library `__init__.py` | **No** | - |
| Standalone script | Yes | top of the script, after imports |
| Test suite | Yes | `tests/conftest.py` (module level) |

## Pattern for tests

`tests/conftest.py` is loaded by pytest before any test module is imported,
making it the correct place to call `load_env()` for a standalone run of
`llm-core`'s own test suite.

Stub values for secrets must be registered with `os.environ.setdefault`
**before** calling `load_env()`. `load_dotenv` does not override variables
that are already present in the process environment, so the stubs take
precedence in both CI (where the real `.env` is absent) and local dev
(where the real `.env` might contain a different value):

```python
"""Pytest configuration and shared fixtures."""

import os

from llm_core.params.load_env import load_env

# Stub values for CI environments where ~/cred/llm-core/.env is absent.
# Set BEFORE load_env() so load_dotenv does not override them.
os.environ.setdefault("SAMPLE_API_KEY", "test-api-key-do-not-use-in-prod")

load_env()
```

This gives tests deterministic stub credentials while still allowing
`load_env()` to populate every other variable from the real `.env` for
tests that need live credentials.

## `load_env()` is still available for standalone use

The function itself lives in `llm_core.params.load_env` and is exported
from there. Any standalone script or application that uses `llm-core`
in isolation can call it explicitly:

```python
from llm_core.params.load_env import load_env

load_env()
```
