# Block 8 - v1 release and consumer migration

Parent: [01-track-progress.md](01-track-progress.md)
Depends on: Block 7 (polish)

---

## Release checklist

- [ ] All blocks 1-7 complete and passing `uv run pytest && uv run ruff check . && uv run pyright`
- [ ] Write `CHANGELOG.md` for v0.1.0
- [ ] Review all `__init__.py` public exports
- [ ] Ensure `README.md` has install + quickstart examples
- [ ] Git tag `v0.1.0`
- [ ] Verify install from git tag: `uv pip install "llm-core[openai] @ git+...@v0.1.0"`

---

## Consumer migration plan

### laife

**Scope:** replace `src/laife/llm_services/` and parts of `src/laife/llm/`

Files to delete (replaced by llm-core imports):
- `src/laife/llm_services/chat/config/` (all files)
- `src/laife/llm_services/embeddings/config/` (all files)
- `src/laife/llm_services/vectorstores/` (all files)
- `src/laife/llm/structured_chain.py`
- `src/laife/llm/prompt_loader.py`
- `src/laife/entities/vectorable.py`

Files to keep (domain-specific):
- `src/laife/llm/player_brain.py`
- `src/laife/llm/mission.py`
- `src/laife/llm/mission_generator.py`
- `src/laife/llm/player_planner.py`
- `src/laife/llm/player_replier.py`
- All prompt `vN.jinja` files

Update `pyproject.toml`:
```toml
dependencies = [
    "llm-core[all] @ git+https://github.com/<org>/llm-core@v0.1.0",
    # ... other deps
]
```

### recipamatic

**Scope:** replace chat config and chain boilerplate

Files to update:
- `py/src/recipamatic/cook/recipe_core/` - replace `ChatOpenAI` config with `ChatOpenAIConfig`
- Replace LCEL chain with `StructuredLLMChain`
- Keep `RecipeCore` and all Pydantic output models

Update `pyproject.toml`:
```toml
dependencies = [
    "llm-core[openai] @ git+https://github.com/<org>/llm-core@v0.1.0",
]
```

### recipinator

**Scope:** replace custom VectorDB with CChroma + EntityStore

Files to update:
- `backend/be/src/be/data/vector_db.py` - replace with `CChroma` import
- Implement `Vectorable` protocol on existing entity classes

Update `pyproject.toml`:
```toml
dependencies = [
    "llm-core[chroma] @ git+https://github.com/<org>/llm-core@v0.1.0",
]
```

---

## Local development template

### Makefile target (copy into each consumer project)

```makefile
LLM_CORE_PATH ?= ../llm-core

.PHONY: dev-llm-core
dev-llm-core:  ## Swap llm-core to a local editable install
	uv pip install -e "$(LLM_CORE_PATH)[all]"
	@echo "llm-core installed from $(LLM_CORE_PATH)"
	@echo "Run 'uv sync' to revert to the pinned git version."
```

### Usage

```bash
make dev-llm-core                              # uses ../llm-core
make dev-llm-core LLM_CORE_PATH=~/dev/llm-core # custom path
uv sync                                        # revert to pinned version
```

---

## Extension contract (document for consumers)

Adding a new provider (e.g. Anthropic, Groq, Bedrock):

1. Subclass `ChatConfig` in your project:
   ```python
   class ChatAnthropicConfig(ChatConfig):
       model: str = "claude-3-sonnet"
       model_provider: str = "anthropic"
       api_key: SecretStr | None = None
   ```
2. Pass to `StructuredLLMChain` as normal - it accepts `ChatConfig` base type
3. If generally useful, open a PR to upstream into llm-core

Same pattern for `EmbeddingsConfig` and `VectorStoreConfig`.

---

## Post-release monitoring

- Track which consumers are on which version
- Write "consumers must update X" callouts in CHANGELOG.md
- Consider Renovate bot for auto-PR on new tags
