# CLAUDE.md

## Development

- Use `uv` for Python version management, dependency management, and script execution
  - `uv run python ...` to execute scripts
  - `uv run ruff check .` / `uv run ruff format .` for lint and formatting
  - `uv add <package>` / `uv add --dev <package>` for dependencies

## Coding Conventions

- Use `logging.getLogger(__name__)` and `logger.info()` instead of `print()`
  - Example scripts should include `setup_logging()` with `format="%(message)s"`
- Linter: ruff (configured in pyproject.toml)
