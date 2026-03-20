# First tasks for llm_core

This file contains the first tasks to do after renaming the template project.
Read it alongside `README_POST_CREATE.md`.

## Update the Copilot instructions

Edit `.github/copilot-instructions.md`:

- Replace the generic project description with a real description of what llm_core does.
- Remove any references to the template or example features that do not apply.
- Update the architecture layers table to reflect the actual structure.
- Keep the style rules, testing, and linting sections intact.

## Review `docs/`

- `docs/index.md` - replace the template overview with a real project description.
- `docs/getting-started.md` - update setup steps for the actual project.
- `docs/contributing.md` - update if contribution guidelines differ.

## Decide which optional features to keep

Ask the user the following questions and act on the answers.

### Webapp (FastAPI)

> "Do you want to keep the FastAPI webapp?"
no

If **no**, remove:
- `src/llm_core/webapp/`
- `src/llm_core/config/webapp/`
- The `webapp` optional dependency group in `pyproject.toml`
- Any webapp imports or middleware wiring in the params/app entrypoints.
- Update `README.md` to remove webapp references.

### MkDocs documentation site

> "Do you want to keep MkDocs?"
yes

If **no**, remove:
- `mkdocs.yml`
- `docs/`
- The `mkdocs` and `mkdocs-material` dependencies.

### Google OAuth and rate limiting

> "Do you want to keep Google OAuth and rate limiting middleware?"
no

If **no**, remove those from `src/llm_core/config/webapp/` and the webapp routers/middleware.

### Haystack / AI layer

> "Do you want to keep the Haystack / LLM dependencies?"
yes

If **no**, remove the relevant entries from `pyproject.toml` and any AI-related source files.

## Update `nokeys.env`

Replace the sample environment variable names with the real ones needed by llm_core.
Remove any variables that are no longer relevant.

## Update `.github/agents/`

Review each agent definition file. Either:
- Update the agent description and commands to reflect llm_core, or
- Remove agents that are not relevant to this project.

## Clean up scratch_space

Review the `scratch_space/` directory and remove stale notebooks, vibes, or other files
that are not relevant to llm_core. Keep only what will be actively used.

## Update `README.md`

Replace the template boilerplate with a real description of llm_core:
- What the project does.
- How to install and run it.
- Key environment variables (reference `nokeys.env`).

## Set up pre-commit hooks

```bash
uv run pre-commit install
```

## Run the verification suite

Confirm everything is working after the cleanup:

```bash
uv run pytest && uv run ruff check . && uv run pyright
```
