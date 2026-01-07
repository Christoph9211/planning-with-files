#!/usr/bin/env python3
"""
ollama_dual_model_build.py  —  collaborative build utilities
============================================================
This module lets two locally hosted Ollama models collaborate on code.

The original flow paired a *Lead Architect* with an *Implementation
Engineer*.  This file now also provides a **Scrum style workflow** where a
*Scrum Master* drafts extremely specific stories and a *Developer* implements
them one at a time.  In addition, a **phased** workflow first generates a
project plan with detailed steps and then executes those steps sequentially.

In both modes the script extracts fenced code blocks (```lang filename```) and
saves them into a ``generated/`` directory.  Optionally unit tests can be
executed after each developer cycle and failing output is fed back into the
conversation.

Command line flags allow choosing the workflow and enabling streaming or test
execution.

---
Usage example
-------------
```bash
# Pull or ensure models exist
ollama pull phi3
ollama pull deepseek-r1:14b

python ollama_dual_model_collab.py \
  --model-a phi3 \
  --model-b deepseek-r1:14b \
  --prompt "Code an addictive strategy game in Python. Brainstorm some ideas for interesting features. Use fenced code blocks" \
  --turns 3 \
  --stream \
  --tests tests/ \
  --output round1.json
  
python ollama_dual_model_build.py --model-a devstral:latest --model-b devstral:latest --prompt "create a real time strategy game in Python. Use fenced code blocks" --turns 3 --stream --output round4.json

# Run the Scrum workflow
python ollama_dual_model_build.py \
  --model-a phi3 \
  --model-b deepseek-r1:14b \
  --prompt "build a mini web server" \
  --workflow scrum \
  --turns 5

"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Generator, List

import requests

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
OLLAMA_HOST = "http://localhost:11434"
GENERATED_DIR = Path("generated")
FENCE_RE = re.compile(r"```\s*(\w+)?\s+([\w./-]+)\s*\n(.*?)```", re.DOTALL)

# ------------------------------------------------------------
# Ollama wrapper
# ------------------------------------------------------------

def generate(
    model: str,
    prompt: str,
    *,
    system: str = "",
    temperature: float = 0.7,
    stream: bool = False,
) -> str | Generator[str, None, None]:
    """Call `/api/generate` on a local Ollama server."""
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "options": {"temperature": temperature},
        "stream": stream,
    }
    r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, stream=stream, timeout=300)
    r.raise_for_status()
    if stream:
        for line in r.iter_lines():
            if not line:
                continue
            msg = json.loads(line.decode())
            if msg.get("done"):
                break
            yield msg["response"]
    else:
        return r.json()["response"]


def full_generate(model: str, prompt: str, *, temperature: float = 0.7) -> str:
    """Return the full response (non-streaming)."""
    return generate(model, prompt, temperature=temperature, stream=False)  # type: ignore[arg-type]


def stream_generate_collect(model: str, prompt: str, *, temperature: float = 0.7) -> str:
    """Stream tokens from the model, print them live, and return the concatenated string.

    Args:
        model (str): The model identifier to use for generation.
        prompt (str): The prompt to provide to the model.
        temperature (float, optional): Sampling temperature to use for generation.

    Returns:
        str: The full response string after streaming and concatenation.
    """

    chunks: list[str] = []
    for tok in generate(model, prompt, temperature=temperature, stream=True):  # type: ignore[arg-type]
        print(tok, end="", flush=True)
        chunks.append(tok)
    print("\n", flush=True)
    return "".join(chunks)

# ------------------------------------------------------------
# File extraction & test runner
# ------------------------------------------------------------

def save_code_blocks(text: str, out_dir: Path = GENERATED_DIR) -> list[str]:
    """Save ```lang filename\n...``` fences. Return written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for _, fname, body in FENCE_RE.findall(text):
        path = out_dir / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        written.append(str(path))
    return written


def run_pytest(tests_path: Path) -> tuple[bool, str]:
    """Run pytest. Return (passed, output)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(tests_path), "-q"],
        capture_output=True,
        text=True,
    )
    passed = result.returncode == 0
    output = result.stdout + result.stderr
    return passed, output


def generate_tests(
    model: str,
    project_desc: str,
    tests_dir: Path,
    *,
    temperature: float = 0.7,
    stream: bool = False,
) -> str:
    """Use ``model`` to create pytest suites for the project."""

    tests_dir.mkdir(parents=True, exist_ok=True)
    prompt = (
        "Write pytest unit tests for the following project. "
        "Respond with fenced code blocks like ```python tests/test_example.py```.\n"
        f"Project description:\n{project_desc}"
    )
    response = (
        stream_generate_collect(model, prompt, temperature=temperature)
        if stream
        else full_generate(model, prompt, temperature=temperature)
    )
    save_code_blocks(response, out_dir=tests_dir)
    return response


def summarize_history(
    history: List[str], *, model: str, temperature: float = 0.7, max_bullets: int = 3
) -> str:
    """Summarize conversation using ``model`` to keep prompts short."""
    convo = "\n".join(history)
    prompt = (
        f"Summarize the following conversation in {max_bullets} concise bullet points:\n"
        f"{convo}"
    )
    return full_generate(model, prompt, temperature=temperature)


def finalize_collaboration(
    model: str,
    history: List[str],
    *,
    temperature: float = 0.7,
    stream: bool = False,
) -> str:
    """Ask ``model`` to produce the final consolidated deliverable."""

    prompt = (
        "Collaboration is complete. Below is the conversation transcript:\n" "'''\n"
        + "\n".join(history)
        + "\n'''\nProvide the final answer or deliverable only."
    )
    response = (
        stream_generate_collect(model, prompt, temperature=temperature)
        if stream
        else full_generate(model, prompt, temperature=temperature)
    )
    return response


def scrum_cycle(
    *,
    sm_model: str,
    dev_model: str,
    project_prompt: str,
    cycles: int = 3,
    temperature: float = 0.7,
    stream: bool = False,
    tests_dir: Path | None = None,
    summary_model: str | None = None,
) -> List[str]:
    """Run a Scrum-style workflow between Scrum Master and Developer."""

    history: list[str] = []
    summary = f"Project: {project_prompt}"
    summarizer = summary_model or sm_model
    dev_last = ""

    if tests_dir and not any(tests_dir.glob("*.py")):
        tests_msg = generate_tests(
            dev_model,
            project_prompt,
            tests_dir,
            temperature=temperature,
            stream=stream,
        )
        history.append(f"Generated tests:\n{tests_msg}")

    for _ in range(cycles):
        # ---- Scrum Master draft ----
        sm_prompt = (
            f"{summary}\n\n"
            f"Developer previously responded:\n'''{dev_last}'''\n"
            "Write the next user with extremely specific context."
        )
        sm_reply = (
            stream_generate_collect(sm_model, sm_prompt, temperature=temperature)
            if stream
            else full_generate(sm_model, sm_prompt, temperature=temperature)
        )
        if not stream:
            print(f"{sm_model}: {sm_reply}\n")
        history.append(f"SM: {sm_reply}")

        # ---- Developer implements code ----
        dev_prompt = (
            "You are the developer. Implement the following code.\n"
            f"Code previously responded:\n'''{sm_reply}'''\n"
            "Respond with code in fenced blocks (```python filename.py)."
        )
        dev_reply = (
            stream_generate_collect(dev_model, dev_prompt, temperature=temperature)
            if stream
            else full_generate(dev_model, dev_prompt, temperature=temperature)
        )
        if not stream:
            print(f"{dev_model}: {dev_reply}\n")
        history.append(f"Dev: {dev_reply}")
        dev_last = dev_reply
        written = save_code_blocks(dev_reply)
        if written:
            print("📝  Saved:", *written)

        if tests_dir and written:
            passed, output = run_pytest(tests_dir)
            print(output)
            history.append(output)
            if not passed:
                dev_last = output

        summary = summarize_history(
            history,
            model=summarizer,
            temperature=temperature,
            max_bullets=3,
        )

    final = finalize_collaboration(
        sm_model,
        history,
        temperature=temperature,
        stream=stream,
    )
    history.append(f"FINAL: {final}")

    return history

# ------------------------------------------------------------
# Collaboration loop
# ------------------------------------------------------------

def collaborative_build(
    *,
    model_a: str,
    model_b: str,
    initial_prompt: str,
    turns: int,
    temperature: float,
    stream: bool,
    tests_dir: Path | None,
) -> List[str]:
    """Execute a collaborative coding session between two models.

    Model A acts as the lead architect, proposing file structures and
    functions, while Model B implements code changes. Code blocks
    provided by Model B are saved to files, and optional unit tests can
    be run after each iteration. Failed tests are fed back into the loop
    for correction. The process alternates between the models for a
    specified number of turns.

    Args:
        model_a (str): Identifier for the lead architect model.
        model_b (str): Identifier for the implementation engineer model.
        initial_prompt (str): Initial prompt provided to Model A.
        turns (int): Number of interaction cycles between the models.
        temperature (float): Sampling temperature for text generation.
        stream (bool): Whether to stream tokens live.
        tests_dir (Path | None): Directory path for unit tests to run,
                                 if provided.

    Returns:
        List[str]: A list of conversation history between the models.
    """

    history: list[str] = []

    if tests_dir and not any(tests_dir.glob("*.py")):
        tests_msg = generate_tests(
            model_b,
            initial_prompt,
            tests_dir,
            temperature=temperature,
            stream=stream,
        )
        history.append(f"Generated tests:\n{tests_msg}")

    # ── round 0 — architect kicks things off ───────────────────────────────
    lead_reply = (
        stream_generate_collect(model_a, initial_prompt, temperature=temperature)
        if stream else
        full_generate(model_a, initial_prompt, temperature=temperature)
    )
    if not stream:
        print(f"{model_a}: {lead_reply}\n")
    history.append(f"{model_a}: {lead_reply}")
    last_resp = lead_reply                         # seed for coder

    # ── main collaboration loop ────────────────────────────────────────────
    for idx in range(turns):
        # ---- coder turn ---------------------------------------------------
        prompt_b = (
            "Your collaborator responded:\n'''\n" + last_resp + "\n'''\n"
            "Write or update code as fenced blocks (```python filename.py)."
        )
        coder_reply = (
            stream_generate_collect(model_b, prompt_b, temperature=temperature)
            if stream else
            full_generate(model_b, prompt_b, temperature=temperature)
        )
        if not stream:
            print(f"{model_b}: {coder_reply}\n")
        history.append(f"{model_b}: {coder_reply}")
        written = save_code_blocks(coder_reply)
        if written:
            print("📝  Saved:", *written)

        # ---- optional test phase -----------------------------------------
        if tests_dir and written:
            passed, output = run_pytest(tests_dir)
            print(output)
            if not passed:
                last_resp = output          # skip architect, let coder fix
                continue

        # ---- architect turn ----------------------------------------------
        prompt_a = (
            "The engineer wrote code (see below). Review and refine next steps.\n"
            "'''\n" + coder_reply + "\n'''"
        )
        arch_reply = (
            stream_generate_collect(model_a, prompt_a, temperature=temperature)
            if stream else
            full_generate(model_a, prompt_a, temperature=temperature)
        )
        if not stream:
            print(f"{model_a}: {arch_reply}\n")
        history.append(f"{model_a}: {arch_reply}")
        last_resp = arch_reply

        # ---- last-round check --------------------------------------------
        if idx == turns - 1:
            final_answer = finalize_collaboration(
                model_a,
                history,
                temperature=temperature,
                stream=stream,
            )
            if not stream:
                print(f"{model_a} (FINAL): {final_answer}\n")
            history.append(f"{model_a} (FINAL): {final_answer}")

    return history


def phased_workflow(
    *,
    planner_model: str,
    dev_model: str,
    project_prompt: str,
    turns: int = 3,
    temperature: float = 0.7,
    stream: bool = False,
    tests_dir: Path | None = None,
) -> List[str]:
    """Plan first, then execute steps sequentially."""

    history: list[str] = []

    if tests_dir and not any(tests_dir.glob("*.py")):
        tests_msg = generate_tests(
            dev_model,
            project_prompt,
            tests_dir,
            temperature=temperature,
            stream=stream,
        )
        history.append(f"Generated tests:\n{tests_msg}")

    # ---- Phase 1: planning ----------------------------------------------
    outline_prompt = (
        "Create a project plan outline listing the small steps required for completion. "
        "Number the steps.\nProject:\n" + project_prompt
    )
    outline = (
        stream_generate_collect(planner_model, outline_prompt, temperature=temperature)
        if stream
        else full_generate(planner_model, outline_prompt, temperature=temperature)
    )
    if not stream:
        print(f"{planner_model}: {outline}\n")
    history.append(f"Outline:\n{outline}")

    steps: list[str] = []
    for line in outline.splitlines():
        m = re.match(r"\s*(?:\d+\.\s*|[-*]\s*)?(.*\S)", line)
        if m:
            step = m.group(1).strip()
            if step:
                steps.append(step)
    if not steps:
        steps = [line.strip() for line in outline.splitlines() if line.strip()]

    detailed_steps: list[tuple[str, str]] = []
    for idx, step in enumerate(steps, 1):
        detail_prompt = (
            f"Provide a detailed plan for step {idx}: {step}\n"
            "Explain the specific actions needed to accomplish it."
        )
        detail = (
            stream_generate_collect(planner_model, detail_prompt, temperature=temperature)
            if stream
            else full_generate(planner_model, detail_prompt, temperature=temperature)
        )
        if not stream:
            print(f"Step {idx} detail: {detail}\n")
        history.append(f"Step {idx} detail: {detail}")
        detailed_steps.append((step, detail))

    # ---- Phase 2: execution ---------------------------------------------
    for idx, (_, detail) in enumerate(detailed_steps, 1):
        exec_prompt = (
            f"Implement step {idx}:\n{detail}\n"
            "Respond with code in fenced blocks where applicable."
        )
        convo = collaborative_build(
            model_a=planner_model,
            model_b=dev_model,
            initial_prompt=exec_prompt,
            turns=turns,
            temperature=temperature,
            stream=stream,
            tests_dir=tests_dir,
        )
        history.extend(convo)

    final = finalize_collaboration(
        planner_model,
        history,
        temperature=temperature,
        stream=stream,
    )
    history.append(f"FINAL: {final}")

    return history

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main() -> None:
    """Run collaborative build workflows with two local Ollama models.

    The default ``architect`` mode runs the classic architect→developer cycle.
    Selecting ``scrum`` enables the Scrum Master/Developer workflow with code
    generation and implementation loops.  The transcript of all turns can be
    optionally saved to disk.
    """
    parser = argparse.ArgumentParser(
        description="Two-model collaborative build workflows via Ollama"
    )
    parser.add_argument("--model-a", required=True, help="Model used for the lead role or Scrum Master")
    parser.add_argument("--model-b", required=True, help="Model used for the secondary role or Developer")
    parser.add_argument("--prompt", required=True, help="Initial project prompt")
    parser.add_argument(
        "--turns", type=int, default=3, help="Number of cycles to run"
    )
    parser.add_argument(
        "--workflow",
        choices=["architect", "scrum", "phased"],
        default="architect",
        help="Choose collaboration style (architect, scrum, or phased)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--stream", action="store_true", help="Stream tokens live to stdout"
    )
    parser.add_argument(
        "--tests-dir", help="Path to unit tests directory to run after each coder turn"
    )
    parser.add_argument(
        "--output", "-o", help="Save full transcript JSON to this file"
    )
    args = parser.parse_args()

    # Choose the collaboration workflow based on the command line argument
    # The default 'architect' workflow is a simple cycle between two models,
    # while 'scrum' enables the Scrum Master/Developer workflow with code
    # generation and implementation loops.  The 'phased' workflow is a more
    # structured cycle of (1) planning, (2) development, and (3) finalization.
    if args.workflow == "scrum":
        # Run the Scrum Master/Developer workflow
        # The Scrum Master generates tasks and the Developer implements them
        # in a loop.  The loop runs for the number of turns specified, and
        # the whole conversation is saved to a list.
        transcript = scrum_cycle(
            sm_model=args.model_a,
            dev_model=args.model_b,
            project_prompt=args.prompt,
            cycles=args.turns,
            temperature=args.temperature,
            stream=args.stream,
            tests_dir=Path(args.tests_dir) if args.tests_dir else None,
            summary_model=None,
        )
    elif args.workflow == "phased":
        # Run the phased workflow, which consists of
        # (1) planning: generate a high-level plan (architect)
        # (2) development: generate code for each task in the plan (developer)
        # (3) finalization: summarize the plan and code, and make final changes
        transcript = phased_workflow(
            planner_model=args.model_a,
            dev_model=args.model_b,
            project_prompt=args.prompt,
            turns=args.turns,
            temperature=args.temperature,
            stream=args.stream,
            tests_dir=Path(args.tests_dir) if args.tests_dir else None,
        )
    else:
        # Run the architect/developer workflow
        # The architect generates code and the developer implements it
        # in a loop.  The loop runs for the number of turns specified, and
        # the whole conversation is saved to a list.
        transcript = collaborative_build(
            model_a=args.model_a,
            model_b=args.model_b,
            initial_prompt=args.prompt,
            turns=args.turns,
            temperature=args.temperature,
            stream=args.stream,
            tests_dir=Path(args.tests_dir) if args.tests_dir else None,
        )

    if args.output:
        Path(args.output).write_text(
            json.dumps(transcript, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
