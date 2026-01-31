
import unittest
from ollama_planning_agent import TaskManager
import os
import shutil
import tempfile

class TestParserRobustness(unittest.TestCase):
    def setUp(self):
        self.manager = TaskManager()
        self.test_dir = "test_artifacts"
        os.makedirs(self.test_dir, exist_ok=True)
        # Mock task file path for the manager if needed, 
        # but save_files_from_response doesn't rely on self.task_file for saving (it uses content)
        # However, checking existence etc might.

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_standard_format(self):
        response = """
Here is the plan:
```markdown task_plan.md
# Plan
```
And here is a script:
```python script.py
print("Hello")
```
        """
        files, cmd = self.manager.save_files_from_response(response)
        self.assertIn("task_plan.md", files)
        self.assertIn("script.py", files)


    def test_missing_filename_in_fence(self):
        # Model forgets filename in fence
        response = """
```python
print("Where do I go?")
```
        """
        # We want to detect this!
        files, cmds = self.manager.save_files_from_response(response)
        # Ideally, we should maybe save this to a 'lost_and_found' or log a warning.
        # For now, let's just assert it is empty but print if we change behavior.
        self.assertEqual(files, [])

    def test_multiple_commands(self):
        response = """
First list:
```bash
ls
```
Then echo:
```bash
echo hello
```
        """
        files, cmds = self.manager.save_files_from_response(response)
        # We WANT to support multiple commands
        # Current behavior is: returns single string "ls"
        # We will change API to return list
        if isinstance(cmds, list):
             self.assertEqual(len(cmds), 2)
             self.assertEqual(cmds[0], "ls")
             self.assertEqual(cmds[1], "echo hello")
        else:
             # Current behavior
             self.assertEqual(cmds, "ls")

    def test_malformed_fence(self):
        # Case: ``` python script.py
        response = """
``` python script.py
print("Space before python")
```
        """
        files, cmds = self.manager.save_files_from_response(response)
        # Should parse correctly
        self.assertIn("script.py", files)

    def test_missing_filename_no_fallback(self):
        # Case: Model forgets filename but provides plan content
        response = """
```markdown
# Task Plan: Fallback
## Phases
- [ ] P1
```
        """
        files, cmds = self.manager.save_files_from_response(response)
        self.assertEqual(files, [])

    def test_section_update_status(self):
        response = """
```markdown task_plan.md#Status
Status line one.
Status line two.
```
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                plan_path = os.path.join(tmpdir, "task_plan.md")
                with open(plan_path, "w", encoding="utf-8") as handle:
                    handle.write("# Task Plan\n## Goal\nDemo\n## Status\nOld\n")
                files, cmds = self.manager.save_files_from_response(response)
                self.assertIn("task_plan.md", files)
                with open(plan_path, "r", encoding="utf-8") as handle:
                    updated = handle.read()
                self.assertIn("## Status", updated)
                self.assertIn("Status line one.", updated)
                self.assertIn("Status line two.", updated)
                self.assertNotIn("Old", updated)
            finally:
                os.chdir(original_cwd)


if __name__ == '__main__':
    unittest.main()
