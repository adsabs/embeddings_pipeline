# AGENT.md - Development Guide

## Commands
- **Environment**: Always activate `.venv` if present: `source .venv/bin/activate` before running commands
- **Dependencies**: `rye sync` or `rye install` (preferred) or `pip install -r requirements.txt`
- **Test**: `pytest tests/` or `python -m pytest tests/`
- **Test single**: `pytest tests/test_filename.py::test_function_name`
- **Lint/Format**: `ruff check` and `ruff format .`
- **Servers**: Detach long-running processes: `nohup python server.py &` or use `screen`/`tmux`

## Architecture
- **Structure**: Separate `src/`, `tests/`, `docs/`, `config/` directories
- **Modules**: Distinct files for models, services, controllers, utilities
- **Config**: Use environment variables, store in `config/` or `.env` files
- **Data**: Training/test data in `data/`, models in `models/` or `checkpoints/`

## Code Style (AI-Optimized)
- **Naming**: Descriptive names, snake_case for functions/variables, PascalCase for classes
- **Types**: Mandatory type hints for all function parameters and returns
- **Docstrings**: Detailed Google/NumPy style with examples and context
- **Error handling**: Rich context capture, specific exception types, comprehensive logging
- **Comments**: Detailed explanations for complex logic to aid AI understanding
- **Imports**: Standard library, third-party, local imports (sorted)
