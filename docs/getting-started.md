# Getting Started

This guide will help you set up your development environment and get started with llm-core.

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/pitrified/llm-core.git
cd llm-core
```

### 2. Install Dependencies

```bash
# Install all dependencies (including dev tools)
uv sync --group dev

# Or install specific groups
uv sync --group test    # Testing only
uv sync --group lint    # Linting only
uv sync --group docs    # Documentation only
```

### 3. Verify Installation

```bash
uv run pytest
uv run ruff check .
uv run pyright
```

## Development Workflow

### Running Tests

```bash
uv run pytest
uv run pytest -v
uv run pytest tests/config/   # Run a specific directory
```

### Code Quality

```bash
uv run ruff check .
uv run ruff format .
uv run pyright
```

### Pre-commit Hooks

```bash
# Install hooks (first time only)
uv run pre-commit install

# Run manually on all files
uv run pre-commit run --all-files
```

## Environment Configuration

The project expects environment variables in `~/cred/llm-core/.env`.
See `nokeys.env` in the repository root for the list of required keys.


!!! warning "Security"
    Never commit `.env` files or secrets to the repository.

## Building Documentation

```bash
# Install docs dependencies
uv sync --group docs

# Start local server with hot reload
uv run mkdocs serve

# Build static site
uv run mkdocs build
```

## Renaming the Project

Use the included renaming utility to customize the project for your needs:

```bash
uv run rename-project
```

See the [meta README](https://github.com/YOUR_USERNAME/llm-core/blob/main/meta/README.md) for detailed instructions.
