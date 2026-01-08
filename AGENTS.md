# Repository Guidelines

This repository hosts a local Ollama-based "Planning with Files" agent plus supporting experiments and skill documentation. Use this guide to orient contributions.

## Project Structure & Module Organization

- `ollama_planning_agent.py`: primary Python entry point for the agent and file/plan management.
- `test_ollama_planning_agent.py` and `tests/`: unit tests covering response parsing and file extraction.
- `planning-with-files/`: skill documentation (`SKILL.md`, `reference.md`, `examples.md`).
- `ollama communication code/`: prototype clients and helpers (TypeScript/JavaScript and Python).
- Generated runtime files (`task_plan.md`, `notes.md`) are gitignored.

## Build, Test, and Development Commands

- `python ollama_planning_agent.py --goal "..." --model <model>` start a new planning session.
- `python ollama_planning_agent.py --continue --turns 5` resume from an existing `task_plan.md`.
- `python -m unittest` run all unit tests from the repository root.
- `python -m unittest discover -s tests` run only tests under `tests/`.

## CLI Options

- `--goal "..."`: required when starting a new plan; sets the objective.
- `--model <name>`: Ollama model to use (default `ministral-3:14b`).
- `--turns <n>`: maximum number of agent turns per run (default `5`).
- `--continue`: resume from an existing `task_plan.md`.
- `--workdir <path>`: change the working directory for plans, notes, and command execution.

## Coding Style & Naming Conventions

- Python: 4-space indentation, snake_case, and `Path` from `pathlib` for filesystem work.
- TypeScript/JavaScript in `ollama communication code/`: 2-space indentation and no semicolons; follow existing file patterns.
- Files: prefer descriptive lowercase names with underscores for Python.

## Testing Guidelines

- Framework: `unittest`.
- Test files follow `test_*.py`.
- Add parsing tests when changing response handling or file/command extraction behavior.

## Commit & Pull Request Guidelines

- Commit messages typically use a `type: summary` prefix such as `feat:`, `test:`, or `refactor:`.
- PRs should include a brief summary, the tests run, and any user-visible behavior changes; link related issues when relevant.

## Configuration & Runtime Notes

- Ollama defaults to `http://localhost:11434` (`OLLAMA_HOST` in `ollama_planning_agent.py`).
- The agent writes `task_plan.md` and `notes.md`; do not commit these artifacts.
