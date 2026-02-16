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

## Pitfalls

- `ruff check --fix` can over-delete imports: if `from enum import Enum, IntEnum` has unused `IntEnum`, fix may remove the entire import line including `Enum`. Always run `ruff check` after `--fix` to catch breakage.
- `print()` → `logger.info()` conversion: bare `print()` (for blank lines) must become `logger.info("")`, not `logger.info()` which raises `TypeError`.
