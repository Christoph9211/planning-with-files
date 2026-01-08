import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ollama_planning_agent import TASK_FILE, TaskManager, run_agent


class TestSaveFilesFromResponse(unittest.TestCase):
    def test_saves_files_and_extracts_command(self):
        response = (
            "```markdown task_plan.md\n"
            "# Task Plan: Demo\n"
            "## Goal\n"
            "Test the parser\n"
            "## Phases\n"
            "- [ ] Phase 1: Setup\n"
            "## Status\n"
            "In progress\n"
            "```\n"
            "\n"
            "```bash\n"
            "ls -la\n"
            "```\n"
        )

        manager = TaskManager()
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                saved_files, commands = manager.save_files_from_response(response)
                self.assertIn("task_plan.md", saved_files)
                self.assertEqual(["ls -la"], commands)
                self.assertTrue(Path(tmpdir, "task_plan.md").exists())
            finally:
                os.chdir(original_cwd)


class TestPlanUpdateAlert(unittest.TestCase):
    def test_alert_inserted_when_task_plan_missing(self):
        prompts = []
        original_cwd = os.getcwd()

        def fake_generate(self, prompt: str, *, system: str = "", stream: bool = False) -> str:
            prompts.append(prompt)
            return ""

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                Path(tmpdir, "task_plan.md").write_text(
                    "# Task Plan: Demo\n"
                    "## Goal\n"
                    "Test alert insertion\n"
                    "## Phases\n"
                    "- [ ] Phase 1: Setup\n"
                    "## Status\n"
                    "In progress\n",
                    encoding="utf-8",
                )

                with patch("ollama_planning_agent.OllamaClient.generate", new=fake_generate), \
                    patch(
                        "ollama_planning_agent.TaskManager.parse_response",
                        side_effect=[([], []), ([str(TASK_FILE)], [])],
                    ), \
                    patch("ollama_planning_agent.time.sleep"):
                    run_agent(
                        goal=None,
                        model="test-model",
                        turns=2,
                        continue_mode=True,
                        workdir=tmpdir,
                        context_file=None,
                    )
            finally:
                os.chdir(original_cwd)

        self.assertGreaterEqual(len(prompts), 2)
        self.assertIn(
            "ALERT: You did not update `task_plan.md` in the previous turn.",
            prompts[1],
        )


if __name__ == "__main__":
    unittest.main()
