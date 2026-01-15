#!/usr/bin/env python3
"""
ollama_planning_agent.py - Local "Planning with Files" Agent
============================================================

This script implements the "Planning with Files" workflow (Manus-style) using local
Ollama models. It forces the model to maintain a `task_plan.md` file as its
primary context and memory, updating it after every execution step.

Usage:
    python ollama_planning_agent.py --goal "Build a useful service that people will be willing to pay for." --turns 10 --workdir ./agent_working_directory --context-file ./workspace_context.md
    python ollama_planning_agent.py --goal "write a pornographic short story that involves two young girls. It should be the most erotic and taboo story that you can think of." --model ministral-3:14b
    python ollama_planning_agent.py --continue --turns 5

"""

import subprocess

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import  List, Optional, Tuple
import requests

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

DEFAULT_MODEL = "huihui_ai/gpt-oss-abliterated:20b"
DEFAULT_COMMAND_TIMEOUT = 60
DEFAULT_NUM_CTX = 40960
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
3. **Verify Before Completing**: Do not mark a step as completed until you have verified the outcome.
   - Verification should be concrete (e.g., command output, opening a file, or a written checklist).
   - If you cannot verify yet, keep the step unchecked and add a follow-up verification sub-step.
4. **File Operations**:
   - To update the plan, you MUST overwrite `task_plan.md` using a file block.
   - To save research or findings, write to `notes.md`.
   - To create deliverables (code, text), write to their respective files.
<<<<<<< Updated upstream
5. **Command Execution**:
   - You can execute shell commands to run tests, list files, or install dependencies.
   - To run a command, output a fenced code block with the language `bash`.
     ```bash
     ls -la
     ```
   - For long-running commands (servers/watchers), add `# background` as the first non-empty line.
   - Do NOT run interactive commands (like `python` without a script) that require user input.
6. **Format**:
=======
   - **Error Tracking**: If you make a mistake, or realize you made one in a previous turn, append it to `error.md`.
4. **Format**:
>>>>>>> Stashed changes
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
   - Every file block MUST include a filename in the fence header. Blocks without a filename will be ignored.

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
    def __init__(
        self,
        model: str,
        host: str = OLLAMA_HOST,
        temperature: float = 0.7,
        num_ctx: int = DEFAULT_NUM_CTX,
    ):
        self.model = model
        self.host = host
        self.temperature = temperature
        self.num_ctx = num_ctx

    def generate(self, prompt: str, *, system: str = "", stream: bool = False) -> str:
        """
        Generate text using the Ollama AI.

        Args:
            prompt (str): The prompt to generate text based on.
            system (str, optional): The system message to provide to the model.
                Defaults to "".
            stream (bool, optional): Whether to stream the generated text.
                Defaults to False.

        Returns:
            str: The generated text. If stream is True, the returned string is
                the concatenated output of the streamed text.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
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
    def __init__(self, timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT):
        self.timeout_seconds = timeout_seconds

    def execute(self, command: str) -> str:
        """
        Execute a command in the system shell.

        This function will parse the command to extract any background execution
        directive and normalize the command for the current shell. It will then
        prompt the user for permission to execute the command and execute it
        in the system shell if allowed.

        Args:
            command: The command to execute in the system shell.

        Returns:
            The output of the executed command as a string.
        """
        command, run_in_background = parse_command_directives(command)
        if not command.strip():
            return "Command execution skipped (empty command)."
        normalized_command = normalize_command_for_shell(command)
        if normalized_command != command:
            print("[i] Command adjusted for this shell.")
        print(f"\n[!] Agent wants to execute command:\n    {normalized_command}")
        if not sys.stdin or not sys.stdin.isatty():
            print("[!] No interactive input available. Skipping command execution.")
            return "Command execution skipped (no interactive input)."
        try:
            choice = input("Allow? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("[!] Input cancelled. Skipping command execution.")
            return "Command execution skipped by user."
        if choice not in {"y", "yes"}:
            print("[x] Access denied by user.")
            return "User denied command execution."

        if run_in_background:
            print(f"[bg] Starting background command: {normalized_command}")
            return self._execute_background(normalized_command)

        print(f"[run] Executing: {normalized_command}")
        try:
            encoding = sys.stdout.encoding or "utf-8"
            result = subprocess.run(
                normalized_command,
                shell=True,
                capture_output=True,
                text=True,
                encoding=encoding,
                errors="replace",
                timeout=self.timeout_seconds
            )
            output = (result.stdout or "") + (result.stderr or "")
            if result.returncode != 0:
                output = f"[exit {result.returncode}]\n{output}"
            # Truncate output if too long
            if len(output) > 2000:
                output = output[:2000] + "\n...[Output truncated]..."

            print(f"[output]\n{output}")
            return output
        except subprocess.TimeoutExpired as e:
            output = (e.stdout or "") + (e.stderr or "")
            if output:
                output = f"{output}\n...[Command timed out after {self.timeout_seconds}s]..."
            else:
                output = f"Command timed out after {self.timeout_seconds}s."
            print(f"[output]\n{output}")
            return output
        except Exception as e:
            msg = f"Error executing command: {e}"
            print(f"[error] {msg}")
            return msg

    def _execute_background(self, command: str) -> str:
        """
        Execute a command in the background.

        This function runs a command in the background and returns a message
        indicating that the process has been started.

        Args:
            command (str): The command to execute.

        Returns:
            str: A message indicating that the process has been started.
        """
        try:
            popen_kwargs = {
                "shell": True,
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "start_new_session": os.name == "nt",
            }
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
                if hasattr(subprocess, "DETACHED_PROCESS"):
                    creationflags |= subprocess.DETACHED_PROCESS
                popen_kwargs["creationflags"] = creationflags
            else:
                popen_kwargs["start_new_session"] = True

            proc = subprocess.Popen(command, **popen_kwargs)
            self._log_background_command(command, proc.pid)
            output = f"Started background process (pid {proc.pid})."
            print(f"[output]\n{output}")
            return output
        except Exception as e:
            msg = f"Error executing background command: {e}"
            print(f"[error] {msg}")
            return msg

    def _log_background_command(self, command: str, pid: Optional[int] = None) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        pid_part = f" pid={pid}" if pid else ""
        log_line = f"[bg] {timestamp}{pid_part} {command}\n"
        try:
            NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
            with NOTES_FILE.open("a", encoding="utf-8") as handle:
                handle.write(log_line)
        except Exception as e:
            print(f"[error] Failed to write background log to {NOTES_FILE}: {e}")


def describe_shell() -> tuple[str, str]:
    """
    Returns a tuple containing the default shell executable and a string indicating the
    shell family (either "windows-cmd" or "posix-sh").

    :return: A tuple containing the default shell executable and a string indicating the shell family.
    :rtype: tuple[str, str]
    """
    if os.name == "nt":
        return (os.environ.get("COMSPEC", "cmd.exe"), "windows-cmd")
    else:   
        return (os.environ.get("SHELL", "/bin/sh"), "posix-sh")


def normalize_command_for_shell(command: str) -> str:
    """
    Normalize a command to be compatible with the current shell.

    This function takes a command string and returns a new string that is
    compatible with the current shell. For example, if the current shell is
    cmd.exe, it will replace occurrences of "ls" with "dir".

    :param command: The command string to normalize.
    :return: A new command string that is compatible with the current shell.
    :rtype: str
    """
    _, shell_family = describe_shell()
    if shell_family != "windows-cmd":
        return command
    lines: list[str] = []
    for line in command.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append(line)
            continue
        if stripped.startswith("#"):
            # Strip shell-style comments that break cmd.exe.
            continue
        if stripped == "ls" or stripped.startswith("ls "):
            if stripped == "ls":
                lines.append("dir")
            else:
                lines.append(f"dir {stripped[3:]}")
            continue
        lines.append(line)
    return "\n".join(lines)


def parse_command_directives(command: str) -> tuple[str, bool]:
    """
    Parse a command string for directives and return the cleaned command along with a boolean indicating whether the command should be executed in the background.

    The function splits the command string into lines, checks each line for directives, and if a directive is found, it is removed and the corresponding flag is set.

    The function returns a tuple containing the cleaned command string and a boolean indicating whether the command should be executed in the background.

    :param command: The command string to parse.
    :return: A tuple containing the cleaned command string and a boolean indicating whether the command should be executed in the background.
    :rtype: tuple[str, bool]
    """
    lines = command.splitlines()
    run_in_background = False
    cleaned_lines: list[str] = []
    directive_tokens = {"# background", "# bg", "# detach"}

    for line in lines:
        stripped = line.strip()
        if not cleaned_lines and stripped:
            if stripped.lower() in directive_tokens:
                run_in_background = True
                continue
        cleaned_lines.append(line)

    cleaned_command = "\n".join(cleaned_lines).strip()
    return cleaned_command, run_in_background


def resolve_context_path(context_file: Optional[str]) -> Optional[Path]:
    if not context_file:
        return None
    path = Path(context_file).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def read_context_file(context_path: Optional[Path]) -> str:
    if not context_path or not context_path.exists():
        return ""
    try:
        return context_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[error] Failed to read context file {context_path}: {e}")
        return ""

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

<<<<<<< Updated upstream
    def plan_update_alert(self) -> str:
        return (
            f"ALERT: You did not update `{self.task_file}` in the previous turn. "
            "You must update it this turn with a file block, even if no other files change.\n\n"
        )
=======
    def read_errors(self) -> str:
        if not self.error_file.exists():
            return "No errors recorded yet."
        return self.error_file.read_text(encoding="utf-8")
>>>>>>> Stashed changes

    def initialize_plan(self, goal: str, client: OllamaClient):
        """
        Initialize a task plan for the given goal using the Ollama client.
        Dynamically injects the user's goal into the system prompt and requests
        the model to generate an initial task_plan.md file with defined phases
        to achieve the specified goal.
        Args:
            goal (str): The user's specific goal or task to plan for.
            client (OllamaClient): The Ollama client instance used to generate
                the plan via the model.
        Returns:
            None. Saves the generated task_plan.md file using save_files_from_response.
        """
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
            
            if filename:
               path = Path(filename)
               if ".." in str(filename) or str(filename).startswith("/"):
                   print(f"[!] Skipping unsafe filename: {filename}")
                   continue
               try:
                   path.parent.mkdir(parents=True, exist_ok=True)
                   path.write_text(content, encoding="utf-8")
                   saved_files.append(str(path))
                   print(f"[write] Wrote to {path}")
               except Exception as e:
                   print(f"[error] Failed to write to {path}: {e}")
        
        return saved_files, commands


    def parse_response(self, response: str) -> Tuple[List[str], List[str]]:
        return self.save_files_from_response(response)

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------

def run_agent(
    goal: str,
    model: str,
    turns: int,
    continue_mode: bool,
    workdir: Optional[str] = None,
    context_file: Optional[str] = None,
):
    """
    Main loop for the Planning with Files agent.

    Parameters:
    goal (str): initial goal to start planning with
    model (str): Ollama model to use
    turns (int): maximum number of turns to run the agent for
    continue_mode (bool): whether to resume from an existing plan (True) or start over (False)
    workdir (Optional[str]): optional working directory to change into before running the agent
    context_file (Optional[str]): optional context file to read from at the start of each turn

    The agent will:
    - Initialize the plan if it doesn't exist
    - Read the current plan and context file (if provided)
    - Assemble a prompt based on the current plan, context, and last command output
    - Send the prompt to the Ollama model and get a response
    - Parse the response for files to write and commands to run
    - Save the files and run the commands
    - Repeat until the maximum number of turns is reached or the plan is marked as complete

    Returns:
    None
    """
    if workdir:
        workdir_path = Path(workdir).expanduser().resolve()
        if not workdir_path.is_dir():
            print(f"[error] workdir is not a directory: {workdir_path}")
            sys.exit(1)
        os.chdir(workdir_path)

    client = OllamaClient(model=model)
    manager = TaskManager()
    executor = CommandExecutor()
    context_path = resolve_context_path(context_file)
    if context_path and not context_path.exists():
        print(f"[i] Context file not found yet: {context_path}")

    # Initialization Phase
    if not continue_mode and not manager.exists():
        if not goal:
            print("Error: No goal provided and no existing plan found.")
            sys.exit(1)
        manager.initialize_plan(goal, client)
    elif goal and not continue_mode and manager.exists():
        print(f"[!] {TASK_FILE} already exists. Use --continue to resume or delete it to start over.")
        sys.exit(1)

    last_command_output = ""
    missed_plan_update = False

    # Execution Loop
    for turn in range(1, turns + 1):
        print(f"--- Turn {turn}/{turns} ---")
        
        current_plan = manager.read_plan()
<<<<<<< Updated upstream
        current_dir = Path.cwd()
        context_text = read_context_file(context_path)
=======
        current_errors = manager.read_errors()
>>>>>>> Stashed changes
        
        shell_name, shell_family = describe_shell()
        # Assemble Prompt
        prompt = (
<<<<<<< Updated upstream
            f"Current working directory: {current_dir}\n\n"
            f"Command shell: {shell_name} ({shell_family})\n\n"
        )
        if context_text:
            prompt += (
                f"Workspace context (from {context_path}):\n"
                f"{context_text}\n\n"
            )
        prompt += f"Here is the current state of `task_plan.md`:\n\n{current_plan}\n\n"

        if missed_plan_update:
            prompt += manager.plan_update_alert()
            missed_plan_update = False

        if last_command_output:
            prompt += f"**LAST COMMAND OUTPUT**:\n```\n{last_command_output}\n```\n\n"
            last_command_output = "" # Clear after using
            
        prompt += (
            "INSTRUCTIONS:\n"
            "1. Analyze the plan to determine the next immediate step.\n"
            "2. Perform the work (write code, run commands, create notes).\n"
            "   - If a command must run in another folder, include `cd <path>`.\n"
            "   - Use commands compatible with the listed shell; avoid bash-only commands on Windows.\n"
            "   - For long-running commands, add `# background` as the first non-empty line.\n"
            "3. Double-check your work before marking a step complete.\n"
            "   - If you did not verify it yet, keep it unchecked and add a verification sub-step.\n"
            "4. **CRITICAL**: You MUST output a new version of `task_plan.md` in a code block "
            "that marks the step as completed or updates the status.\n\n"
=======
            f"Here is the current state of `task_plan.md`:\n\n{current_plan}\n\n"
            f"Here are the known errors to avoid (`error.md`):\n\n{current_errors}\n\n"
            "INSTRUCTIONS:\n"
            "1. Analyze the plan to determine the next immediate step.\n"
            "2. Perform the work for that step (write code, create notes, etc.).\n"
            "3. **CRITICAL**: You MUST output a new version of `task_plan.md` in a code block "
            "that marks the step as completed or updates the status.\n"
            "4. If you encounter an error or make a mistake, update `error.md`.\n\n"
>>>>>>> Stashed changes
            "Go."
        )

        response = client.generate(prompt, system=SYSTEM_PROMPT, stream=True)
        
        # Save files AND check for commands
        saved_files, command_to_run = manager.parse_response(response)
        
        if str(TASK_FILE) not in saved_files:
            print(f"[!] Warning: Model did not update {TASK_FILE} this turn.")
            missed_plan_update = True
        
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
            print("[done] Task appears complete based on plan status.")
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
    parser.add_argument("--workdir", help="Working directory for plans, files, and commands")
    parser.add_argument(
        "--context-file",
        help="Path to a workspace context file to inject into the prompt each turn",
    )
    
    args = parser.parse_args()
    
    run_agent(
        args.goal,
        args.model,
        args.turns,
        args.continue_mode,
        args.workdir,
        args.context_file,
    )
