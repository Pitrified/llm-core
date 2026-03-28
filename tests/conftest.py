"""Pytest configuration and shared fixtures."""

import os

from llm_core.params.load_env import load_env

# Stub values for CI environments where ~/cred/llm-core/.env is absent.
# These are set BEFORE load_env() so that load_dotenv (which does not
# override by default) leaves them untouched even when the real .env exists.
os.environ.setdefault("SAMPLE_API_KEY", "test-api-key-do-not-use-in-prod")

load_env()
