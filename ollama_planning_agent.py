#!/usr/bin/env python3
"""
ollama_planning_agent.py - Local "Planning with Files" Agent
============================================================

This script implements the "Planning with Files" workflow (Manus-style) using local
Ollama models. It forces the model to maintain a `task_plan.md` file as its
primary context and memory, updating it after every execution step.

Usage:
    python ollama_planning_agent.py --goal "create a comprehensive design document for a evolving ecosystem within a falling sand simulation-sandbox environment" --model huihui_ai/glm-4.7-flash-abliterated:latest --turns 10
    python ollama_planning_agent.py --continue --turns 5

"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Generator, List, Optional
import requests

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

DEFAULT_MODEL = "huihui_ai/glm-4.7-flash-abliterated:latest"
OLLAMA_HOST = "http://localhost:11434"
TASK_FILE = Path("task_plan.md")
NOTES_FILE = Path("notes.md")
ERROR_FILE = Path("error.md")

SYSTEM_PROMPT = """
You are an autonomous AI agent that follows the "Planning with Files" workflow.
Your goal is to complete the user's objective by iteratively planning, executing, and updating your plan.

# CRITICAL RULES:

1. **The Plan is Truth**: You must heavily rely on `task_plan.md`. It is your long-term memory.
2. **One Step at a Time**: Read the plan, identify the next incomplete phase/step, execute it, and then UPDATE the plan.
3. **File Operations**:
   - To update the plan, you MUST overwrite `task_plan.md` using a file block.
   - To save research or findings, write to `notes.md`.
   - To create deliverables (code, text), write to their respective files.
   - **Error Tracking**: If you make a mistake, or realize you made one in a previous turn, append it to `error.md`.
4. **Format**:
   - Return code or file content in fenced code blocks:
     ```markdown task_plan.md
     ... content ...
     ```
     ```python script.py
     ... content ...
     ```

# ERROR TRACKING:
Check `error.md` to see a list of past mistakes. You MUST avoid repeating these errors.
If you encounter an error during execution, document it in `error.md` for future reference.

# TASK PLAN TEMPLATE:
When initializing a new task, use this structure:

# Task Plan: [Title]
## Goal
[Description]
## Phases
- [ ] Phase 1: Planning & Setup
- [ ] Phase 2: Execution
- [ ] Phase 3: Review
## Status
[Current status description]

"""

# -----------------------------------------------------------------------------
# Ollama Client
# -----------------------------------------------------------------------------

class OllamaClient:
    def __init__(self, host: str = OLLAMA_HOST, model: str = DEFAULT_MODEL):
        self.host = host
        self.model = model

    def generate(self, prompt: str, system: str = "", stream: bool = True) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": stream,
            "options": {"temperature": 0.7}
        }
        
        try:
            response = requests.post(url, json=payload, stream=stream, timeout=300)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama: {e}")
            sys.exit(1)

        full_response = []
        if stream:
            print(f"\n🤖 {self.model} is thinking...\n")
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    print(token, end="", flush=True)
                    full_response.append(token)
                    if data.get("done"):
                        break
            print("\n")
        else:
            data = response.json()
            full_response.append(data.get("response", ""))
            
        return "".join(full_response)

# -----------------------------------------------------------------------------
# File & Plan Manager
# -----------------------------------------------------------------------------

class TaskManager:
    def __init__(self):
        self.task_file = TASK_FILE
        self.notes_file = NOTES_FILE
        self.error_file = ERROR_FILE

    def exists(self) -> bool:
        return self.task_file.exists()

    def read_plan(self) -> str:
        if not self.exists():
            return "No plan exists yet."
        return self.task_file.read_text(encoding="utf-8")

    def read_errors(self) -> str:
        if not self.error_file.exists():
            return "No errors recorded yet."
        return self.error_file.read_text(encoding="utf-8")

    def initialize_plan(self, goal: str, client: OllamaClient):
        print(f"Creating initial plan for goal: '{goal}'")
        
        # Inject the goal dynamically into the system prompt for strongest effect
        dynamic_system = SYSTEM_PROMPT.replace("[Description]", f"The user's specific goal is: {goal}")
        
        prompt = (
            f"You are starting a new task. The User's Goal is: '{goal}'\n\n"
            "INSTRUCTIONS:\n"
            "1. You must acknowledge this goal and create the initial `task_plan.md`.\n"
            "2. Define phases to achieve exactly this goal. Use the template provided.\n"
            "3. Create an initial empty `error.md` file to track future mistakes.\n"
            "4. Respond ONLY with the markdown code blocks for `task_plan.md` and `error.md`."
        )
        response = client.generate(prompt, system=dynamic_system, stream=True)
        self.save_files_from_response(response)

    def save_files_from_response(self, response: str) -> List[str]:
        """Extracts ```lang filename ... ``` blocks and saves them."""
        # Regex to capture ```[lang] [filename]\n[content]```
        # Robust pattern: matches optional language, then filename (which might be on same line or next), then content
        pattern = re.compile(r"```(?:\w+)?(?:[ \t]+)([\w./-]+)[ \t]*\n(.*?)```", re.DOTALL)
        matches = pattern.findall(response)
        
        # Fallback: if no filename found, look for ```markdown\n...``` and assume it's task_plan.md if it looks like one
        if not matches:
             fallback_pattern = re.compile(r"```(?:markdown)?\s*\n(.*?)```", re.DOTALL)
             fallback_matches = fallback_pattern.findall(response)
             for content in fallback_matches:
                 if "# Task Plan" in content or "## Phases" in content:
                     matches.append(("task_plan.md", content))

        saved_files = []
        for filename, content in matches:
            # Clean filename
            filename = filename.strip()
            if not filename or filename == "---": # generic separator detected as filename
                continue
            
            path = Path(filename)
            # Security check: prevent writing outside cwd (basic)
            if ".." in str(filename) or str(filename).startswith("/"):
                print(f"⚠️  Skipping unsafe filename: {filename}")
                continue
                
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                saved_files.append(str(path))
                print(f"📝 Wrote to {path}")
            except Exception as e:
                print(f"❌ Failed to write to {path}: {e}")
            
        return saved_files

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------

def run_agent(goal: str, model: str, turns: int, continue_mode: bool):
    client = OllamaClient(model=model)
    manager = TaskManager()

    # Initialization Phase
    if not continue_mode and not manager.exists():
        if not goal:
            print("Error: No goal provided and no existing plan found.")
            sys.exit(1)
        manager.initialize_plan(goal, client)
    elif goal and not continue_mode and manager.exists():
        print(f"⚠️  {TASK_FILE} already exists. Use --continue to resume or delete it to start over.")
        sys.exit(1)

    # Execution Loop
    for turn in range(1, turns + 1):
        print(f"--- Turn {turn}/{turns} ---")
        
        current_plan = manager.read_plan()
        current_errors = manager.read_errors()
        
        # Assemble Prompt
        prompt = (
            f"Here is the current state of `task_plan.md`:\n\n{current_plan}\n\n"
            f"Here are the known errors to avoid (`error.md`):\n\n{current_errors}\n\n"
            "INSTRUCTIONS:\n"
            "1. Analyze the plan to determine the next immediate step.\n"
            "2. Perform the work for that step (write code, create notes, etc.).\n"
            "3. **CRITICAL**: You MUST output a new version of `task_plan.md` in a code block "
            "that marks the step as completed or updates the status.\n"
            "4. If you encounter an error or make a mistake, update `error.md`.\n\n"
            "Go."
        )

        response = client.generate(prompt, system=SYSTEM_PROMPT, stream=True)
        
        # Save any files generated (including the updated plan)
        saved = manager.save_files_from_response(response)
        
        if str(TASK_FILE) not in saved:
            print(f"⚠️  Warning: Model did not update {TASK_FILE} this turn.")
        
        # Optional: check if done
        if "[x] Phase 4" in manager.read_plan() or "Status: Complete" in manager.read_plan():
            print("✅ Task appears complete based on plan status.")
            return

        time.sleep(1) # Brief pause

    # End of loop - Check if we need to auto-finalize
    current_plan = manager.read_plan()
    if "Status: Complete" not in current_plan and "[x] Phase 4" not in current_plan:
        print("\n⚠️  Loop ended but plan not marked complete. Auto-finalizing...")
        prompt = (
            f"The task execution loop has finished. Here is the last known plan:\n{current_plan}\n\n"
            "INSTRUCTIONS:\n"
            "1. Review the work done.\n"
            "2. If the work is actually complete, output the FINAL `task_plan.md` with all items marked [x] and Status: Complete.\n"
            "3. If not complete, mark the current status accurately."
        )
        response = client.generate(prompt, system=SYSTEM_PROMPT, stream=True)
        manager.save_files_from_response(response)

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Agent with File-based Memory")
    parser.add_argument("--goal", help="The main objective (required if starting new)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--turns", type=int, default=5, help="Maximum number of turns to run")
    parser.add_argument("--continue", dest="continue_mode", action="store_true", help="Continue from existing plan")
    
    args = parser.parse_args()
    
    run_agent(args.goal, args.model, args.turns, args.continue_mode)
