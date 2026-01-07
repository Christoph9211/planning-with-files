#!/usr/bin/env python3
"""
ollama_planning_agent.py - Local "Planning with Files" Agent
============================================================

This script implements the "Planning with Files" workflow (Manus-style) using local
Ollama models. It forces the model to maintain a `task_plan.md` file as its
primary context and memory, updating it after every execution step.

Usage:
    python ollama_planning_agent.py --goal "Build a simple weather dashboard" --model llama3
    python ollama_planning_agent.py --continue --turns 5

"""

import subprocess

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union
import requests

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

DEFAULT_MODEL = "ministral-3:14b"
OLLAMA_HOST = "http://localhost:11434"
TASK_FILE = Path("task_plan.md")
NOTES_FILE = Path("notes.md")

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
4. **Command Execution**:
   - You can execute shell commands to run tests, list files, or install dependencies.
   - To run a command, output a fenced code block with the language `bash`.
     ```bash
     ls -la
     ```
   - Do NOT run interactive commands (like `python` without a script) that require user input.
5. **Format**:
   - Return code or file content in fenced code blocks:
     ```markdown task_plan.md
     ... content ...
     ```
     ```python script.py
     ... content ...
     ```
     ```bash
     python script.py
     ```

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
    def __init__(
        self,
        model: str,
        host: str = OLLAMA_HOST,
        temperature: float = 0.7,
    ):
        self.model = model
        self.host = host
        self.temperature = temperature

    def generate(self, prompt: str, *, system: str = "", stream: bool = False) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "options": {"temperature": self.temperature},
            "stream": stream,
        }
        response = requests.post(
            f"{self.host}/api/generate",
            json=payload,
            stream=stream,
            timeout=300,
        )
        response.raise_for_status()

        if not stream:
            return response.json().get("response", "")

        chunks: List[str] = []
        for line in response.iter_lines():
            if not line:
                continue
            message = json.loads(line.decode())
            if message.get("done"):
                break
            token = message.get("response", "")
            if token:
                print(token, end="", flush=True)
                chunks.append(token)
        print("\n", flush=True)
        return "".join(chunks)

# -----------------------------------------------------------------------------
# Command Executor
# -----------------------------------------------------------------------------

class CommandExecutor:
    @staticmethod
    def execute(command: str) -> str:
        print(f"\n⚠️  Agent wants to execute command:\n    {command}")
        choice = input("Allow? [y/N]: ").strip().lower()
        if choice != 'y':
            print("❌ Access denied by user.")
            return "User denied command execution."
            
        print(f"🚀 Executing: {command}")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60
            )
            output = result.stdout + result.stderr
            # Truncate output if too long
            if len(output) > 2000:
                output = output[:2000] + "\n...[Output truncated]..."
            
            print(f"📄 Output:\n{output}")
            return output
        except Exception as e:
            return f"Error executing command: {e}"

# -----------------------------------------------------------------------------
# File & Plan Manager
# -----------------------------------------------------------------------------

class TaskManager:
    def __init__(self):
        self.task_file = TASK_FILE
        self.notes_file = NOTES_FILE

    def exists(self) -> bool:
        return self.task_file.exists()

    def read_plan(self) -> str:
        if not self.exists():
            return "No plan exists yet."
        return self.task_file.read_text(encoding="utf-8")

    def initialize_plan(self, goal: str, client: OllamaClient):
        print(f"Creating initial plan for goal: '{goal}'")
        
        # Inject the goal dynamically into the system prompt for strongest effect
        dynamic_system = SYSTEM_PROMPT.replace("[Description]", f"The user's specific goal is: {goal}")
        
        prompt = (
            f"You are starting a new task. The User's Goal is: '{goal}'\n\n"
            "INSTRUCTIONS:\n"
            "1. You must acknowledge this goal and create the initial `task_plan.md`.\n"
            "2. Define phases to achieve exactly this goal. Use the template provided.\n"
            "3. Respond ONLY with the markdown code block for `task_plan.md`."
        )
        response = client.generate(prompt, system=dynamic_system, stream=True)
        self.save_files_from_response(response)

    def save_files_from_response(self, response: str) -> Tuple[List[str], List[str]]:
        """Extracts code blocks for files and commands."""
        # Regex to capture fully fenced blocks: ```header\ncontent```
        # We capture the header line (group 1) and the content (group 2)
        block_pattern = re.compile(r"```([^\n]*)\n(.*?)```", re.DOTALL)
        matches = block_pattern.findall(response)
        
        saved_files = []
        commands = []
        
        # Helper to check if a token looks like a filename
        def is_filename(t: str) -> bool:
            return "." in t or "/" in t or t == "task_plan.md"

        for header, content in matches:
            header = header.strip()
            tokens = header.split()
            
            # Detect Command
            # If header is exactly 'bash' or 'sh' or 'shell' with no filename
            if len(tokens) == 1 and tokens[0] in ["bash", "sh", "shell"]:
                commands.append(content.strip())
                continue
            
            # Detect File
            filename = None
            if len(tokens) >= 2:
                # Assume last token is filename, e.g. "python script.py"
                possible_file = tokens[-1]
                # Optional: verification?
                filename = possible_file
            elif len(tokens) == 1:
                # Single token: "script.py" or "task_plan.md"
                if is_filename(tokens[0]):
                     filename = tokens[0]
                # Else it's likely just a language "python" -> ignore
            
            # Fallback: Check for Task Plan content if no filename detected
            if not filename and ("# Task Plan" in content or "## Phases" in content):
                 filename = "task_plan.md"
            
            if filename:
               path = Path(filename)
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
        
        return saved_files, commands


    def parse_response(self, response: str) -> Tuple[List[str], List[str]]:
        return self.save_files_from_response(response)

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------

def run_agent(goal: str, model: str, turns: int, continue_mode: bool):
    client = OllamaClient(model=model)
    manager = TaskManager()
    executor = CommandExecutor()

    # Initialization Phase
    if not continue_mode and not manager.exists():
        if not goal:
            print("Error: No goal provided and no existing plan found.")
            sys.exit(1)
        manager.initialize_plan(goal, client)
    elif goal and not continue_mode and manager.exists():
        print(f"⚠️  {TASK_FILE} already exists. Use --continue to resume or delete it to start over.")
        sys.exit(1)

    last_command_output = ""

    # Execution Loop
    for turn in range(1, turns + 1):
        print(f"--- Turn {turn}/{turns} ---")
        
        current_plan = manager.read_plan()
        
        # Assemble Prompt
        prompt = (
            f"Here is the current state of `task_plan.md`:\n\n{current_plan}\n\n"
        )
        
        if last_command_output:
            prompt += f"**LAST COMMAND OUTPUT**:\n```\n{last_command_output}\n```\n\n"
            last_command_output = "" # Clear after using
            
        prompt += (
            "INSTRUCTIONS:\n"
            "1. Analyze the plan to determine the next immediate step.\n"
            "2. Perform the work (write code, run commands, create notes).\n"
            "3. **CRITICAL**: You MUST output a new version of `task_plan.md` in a code block "
            "that marks the step as completed or updates the status.\n\n"
            "Go."
        )

        response = client.generate(prompt, system=SYSTEM_PROMPT, stream=True)
        
        # Save files AND check for commands
        saved_files, command_to_run = manager.parse_response(response)
        
        if str(TASK_FILE) not in saved_files:
            print(f"⚠️  Warning: Model did not update {TASK_FILE} this turn.")
        
        # Execute Commands if found
        if command_to_run:
            for cmd in command_to_run:
                 cmd = cmd.strip()
                 if not cmd: continue
                 # Use accumulator for multiple commands output
                 out = executor.execute(cmd)
                 last_command_output += f"\n[{cmd}]:\n{out}\n"

        # Optional: check if done
        if "[x] Phase 4" in manager.read_plan() or "Status: Complete" in manager.read_plan():
            print("✅ Task appears complete based on plan status.")
            return
            
        time.sleep(1) # Brief pause


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
