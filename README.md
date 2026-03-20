# llm-core

Reusable LLM tooling library for Python projects. Provides multi-provider chat/embeddings
configs, a generic structured-output chain, versioned Jinja2 prompts, and vector store
abstractions - all built on LangChain and the `BaseModelKwargs` config pattern.

## Installation

Setup [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync --all-extras --all-groups
```

## Docs

Docs are available at [https://pitrified.github.io/llm-core/](https://pitrified.github.io/llm-core/).

## Setup

### Environment Variables

Create `~/cred/llm-core/.env` with the required keys (see `nokeys.env` for the full list).

For VSCode to pick up the env file, add to your workspace settings:

```json
"python.envFile": "/home/${env:USER}/cred/llm-core/.env"
```

### Pre-commit

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

### Linting and type checking

```bash
uv run ruff check --fix
uv run ruff format
uv run pyright
```

### Testing

```bash
uv run pytest
```
