"""Tests for PromptLoader and PromptLoaderConfig."""

from pathlib import Path

import pytest

from llm_core.prompts.prompt_loader import NoPromptVersionFoundError
from llm_core.prompts.prompt_loader import PromptLoader
from llm_core.prompts.prompt_loader import PromptLoaderConfig


def _write_prompt(base: Path, name: str, version: int, content: str) -> Path:
    prompt_dir = base / name
    prompt_dir.mkdir(parents=True, exist_ok=True)
    path = prompt_dir / f"v{version}.jinja"
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# PromptLoaderConfig
# ---------------------------------------------------------------------------


def test_config_defaults(tmp_path: Path) -> None:
    """PromptLoaderConfig defaults version to 'auto'."""
    cfg = PromptLoaderConfig(base_prompt_fol=tmp_path, prompt_name="my_prompt")
    assert cfg.version == "auto"


def test_config_explicit_version(tmp_path: Path) -> None:
    """PromptLoaderConfig stores explicit version string."""
    cfg = PromptLoaderConfig(
        base_prompt_fol=tmp_path, prompt_name="my_prompt", version="3"
    )
    assert cfg.version == "3"


# ---------------------------------------------------------------------------
# Auto-version discovery
# ---------------------------------------------------------------------------


def test_auto_picks_highest_version(tmp_path: Path) -> None:
    """Auto mode returns content of the highest-numbered vN.jinja."""
    _write_prompt(tmp_path, "p", 1, "v1 content")
    _write_prompt(tmp_path, "p", 3, "v3 content")
    _write_prompt(tmp_path, "p", 2, "v2 content")
    cfg = PromptLoaderConfig(base_prompt_fol=tmp_path, prompt_name="p")
    loader = PromptLoader(cfg)
    assert loader.load_prompt() == "v3 content"


def test_auto_single_version(tmp_path: Path) -> None:
    """Auto mode works correctly when only one version exists."""
    _write_prompt(tmp_path, "p", 1, "only version")
    cfg = PromptLoaderConfig(base_prompt_fol=tmp_path, prompt_name="p")
    assert PromptLoader(cfg).load_prompt() == "only version"


def test_auto_non_sequential_versions(tmp_path: Path) -> None:
    """Auto should pick v5 even if v2, v3, v4 are absent."""
    _write_prompt(tmp_path, "p", 1, "v1")
    _write_prompt(tmp_path, "p", 5, "v5 content")
    cfg = PromptLoaderConfig(base_prompt_fol=tmp_path, prompt_name="p")
    assert PromptLoader(cfg).load_prompt() == "v5 content"


def test_auto_raises_when_no_versions(tmp_path: Path) -> None:
    """Auto mode raises NoPromptVersionFoundError when no vN.jinja files exist."""
    (tmp_path / "p").mkdir()
    cfg = PromptLoaderConfig(base_prompt_fol=tmp_path, prompt_name="p")
    with pytest.raises(NoPromptVersionFoundError):
        PromptLoader(cfg).load_prompt()


def test_auto_ignores_non_versioned_files(tmp_path: Path) -> None:
    """Files that do not match vN.jinja are ignored by auto mode."""
    prompt_dir = tmp_path / "p"
    prompt_dir.mkdir()
    (prompt_dir / "notes.jinja").write_text("ignored", encoding="utf-8")
    (prompt_dir / "vabc.jinja").write_text("also ignored", encoding="utf-8")
    cfg = PromptLoaderConfig(base_prompt_fol=tmp_path, prompt_name="p")
    with pytest.raises(NoPromptVersionFoundError):
        PromptLoader(cfg).load_prompt()


# ---------------------------------------------------------------------------
# Explicit version
# ---------------------------------------------------------------------------


def test_explicit_version_loads_correct_file(tmp_path: Path) -> None:
    """Explicit version string selects the matching vN.jinja file."""
    _write_prompt(tmp_path, "p", 1, "v1 content")
    _write_prompt(tmp_path, "p", 2, "v2 content")
    cfg = PromptLoaderConfig(base_prompt_fol=tmp_path, prompt_name="p", version="1")
    assert PromptLoader(cfg).load_prompt() == "v1 content"


def test_explicit_version_missing_raises(tmp_path: Path) -> None:
    """Explicit version that points to a missing file raises FileNotFoundError."""
    (tmp_path / "p").mkdir()
    cfg = PromptLoaderConfig(base_prompt_fol=tmp_path, prompt_name="p", version="99")
    with pytest.raises(FileNotFoundError):
        PromptLoader(cfg).load_prompt()


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_caching_returns_same_object(tmp_path: Path) -> None:
    """Subsequent load_prompt() calls return the same cached string object."""
    _write_prompt(tmp_path, "p", 1, "cached content")
    cfg = PromptLoaderConfig(base_prompt_fol=tmp_path, prompt_name="p")
    loader = PromptLoader(cfg)
    first = loader.load_prompt()
    second = loader.load_prompt()
    assert first is second


def test_caching_does_not_reread_after_file_change(tmp_path: Path) -> None:
    """Cache is not invalidated when the underlying file is modified."""
    path = _write_prompt(tmp_path, "p", 1, "original")
    cfg = PromptLoaderConfig(base_prompt_fol=tmp_path, prompt_name="p")
    loader = PromptLoader(cfg)
    loader.load_prompt()
    path.write_text("modified", encoding="utf-8")
    assert loader.load_prompt() == "original"


# ---------------------------------------------------------------------------
# Template content is returned as-is
# ---------------------------------------------------------------------------


def test_template_returned_verbatim(tmp_path: Path) -> None:
    """load_prompt returns raw Jinja template string without rendering."""
    content = "Hello {{ name }}!\n{% if flag %}yes{% endif %}"
    _write_prompt(tmp_path, "p", 1, content)
    cfg = PromptLoaderConfig(base_prompt_fol=tmp_path, prompt_name="p")
    assert PromptLoader(cfg).load_prompt() == content
