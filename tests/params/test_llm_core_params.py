"""Test the LlmCoreParams class."""

from llm_core.params.llm_core_params import LlmCoreParams
from llm_core.params.llm_core_params import get_llm_core_params
from llm_core.params.llm_core_paths import LlmCorePaths
from llm_core.params.sample_params import SampleParams


def test_llm_core_params_singleton() -> None:
    """Test that LlmCoreParams is a singleton."""
    params1 = LlmCoreParams()
    params2 = LlmCoreParams()
    assert params1 is params2
    assert get_llm_core_params() is params1


def test_llm_core_params_init() -> None:
    """Test initialization of LlmCoreParams."""
    params = LlmCoreParams()
    assert isinstance(params.paths, LlmCorePaths)
    assert isinstance(params.sample, SampleParams)


def test_llm_core_params_str() -> None:
    """Test string representation."""
    params = LlmCoreParams()
    s = str(params)
    assert "LlmCoreParams:" in s
    assert "LlmCorePaths:" in s
    assert "SampleParams:" in s
