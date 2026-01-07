import os
import tempfile
import unittest
from pathlib import Path

from ollama_planning_agent import TaskManager


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


if __name__ == "__main__":
    unittest.main()
