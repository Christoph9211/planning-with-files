# planning-with-files

Local tooling for the "Planning with Files" workflow using Ollama. The core script manages a `task_plan.md` file as long-term memory and can execute shell commands with explicit user approval.

## Quick Start

- Start a new plan: `python ollama_planning_agent.py --goal "Build a simple weather dashboard" --model mistral`
- Resume an existing plan: `python ollama_planning_agent.py --continue --turns 5`
- Run tests: `python -m unittest`

## CLI Options

- `--goal "..."`: required when starting a new plan; sets the objective.
- `--model <name>`: Ollama model to use (default `ministral-3:14b`).
- `--turns <n>`: maximum number of agent turns per run (default `5`).
- `--continue`: resume from an existing `task_plan.md`.
- `--workdir <path>`: change the working directory for plans, notes, and command execution.

## Repository Layout

- `ollama_planning_agent.py`: main agent and file/plan manager.
- `test_ollama_planning_agent.py` and `tests/`: parsing and robustness tests.
- `planning-with-files/`: skill documentation (`SKILL.md`, `reference.md`, `examples.md`).
- `ollama communication code/`: experimental API clients and helpers.

## Notes

- Runtime artifacts (`task_plan.md`, `notes.md`) are gitignored.
- Ollama defaults to `http://localhost:11434` (see `OLLAMA_HOST`).
