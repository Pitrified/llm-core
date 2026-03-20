"""Test that the environment variables are available."""

import os


def test_env_vars() -> None:
    """The environment var LLM_CORE_SAMPLE_ENV_VAR is available."""
    assert "LLM_CORE_SAMPLE_ENV_VAR" in os.environ
    assert os.environ["LLM_CORE_SAMPLE_ENV_VAR"] == "sample"
