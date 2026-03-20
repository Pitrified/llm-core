"""Test the llm_core paths."""

from llm_core.params.llm_core_params import get_llm_core_paths


def test_llm_core_paths() -> None:
    """Test the llm_core paths."""
    llm_core_paths = get_llm_core_paths()
    assert llm_core_paths.src_fol.name == "llm_core"
    assert llm_core_paths.root_fol.name == "llm-core"
    assert llm_core_paths.data_fol.name == "data"
