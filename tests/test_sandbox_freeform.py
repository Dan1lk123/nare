"""Regression tests for the HYBRID-path freeform-code execution helpers.

The benchmark log showed that HYBRID was effectively SLOW: the LLM emitted
\"\"\"
```python
import re
text = '...'
emails = re.findall(...)
print(emails)
```
\"\"\"
and the agent stored the raw fenced block as the answer. These helpers fix
that by extracting the fenced block, executing it, and using the captured
stdout as the answer.
"""

from __future__ import annotations

import unittest

from nare.execution.sandboxes.base import (
    SecurityError,
    extract_python_block,
    safe_execute_freeform,
)


class TestExtractPythonBlock(unittest.TestCase):
    def test_extract_python_fenced(self):
        text = "Some prose\n```python\nprint(1+1)\n```\nMore prose"
        self.assertEqual(extract_python_block(text), "print(1+1)")

    def test_extract_unspecified_fenced(self):
        # Backtick fence without language tag should still be extracted.
        text = "```\nprint('x')\n```"
        self.assertEqual(extract_python_block(text), "print('x')")

    def test_extract_no_block_returns_empty(self):
        self.assertEqual(extract_python_block("just a string"), "")
        self.assertEqual(extract_python_block(""), "")
        self.assertEqual(extract_python_block(None), "")  # type: ignore[arg-type]

    def test_extract_first_block_when_multiple(self):
        text = "```python\nprint(1)\n```\n```python\nprint(2)\n```"
        self.assertEqual(extract_python_block(text), "print(1)")


class TestSafeExecuteFreeform(unittest.TestCase):
    def test_print_captured_as_answer(self):
        result = safe_execute_freeform("print(2 + 2)")
        self.assertEqual(result, "4")

    def test_last_print_line_returned(self):
        code = "print('warmup')\nprint('actual answer')"
        self.assertEqual(safe_execute_freeform(code), "actual answer")

    def test_result_variable_fallback(self):
        code = "result = 42"
        self.assertEqual(safe_execute_freeform(code), "42")

    def test_no_output_returns_empty(self):
        code = "x = 1"
        self.assertEqual(safe_execute_freeform(code), "")

    def test_runtime_error_returns_error_prefix(self):
        result = safe_execute_freeform("print(1 / 0)")
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("ZeroDivisionError", result)

    def test_security_error_propagates(self):
        # ``open`` is a forbidden bare call.
        with self.assertRaises(SecurityError):
            safe_execute_freeform("open('/etc/passwd').read()")

    def test_email_extraction_realistic(self):
        # Mimics the kind of code the benchmark log showed HYBRID emitting.
        code = (
            "import re\n"
            "text = 'Send to a@x.com and b@x.com'\n"
            "print(', '.join(re.findall(r'[a-z]+@[a-z.]+', text)))"
        )
        self.assertEqual(safe_execute_freeform(code), "a@x.com, b@x.com")

    def test_output_truncation(self):
        code = "print('A' * 10000)"
        result = safe_execute_freeform(code, max_output_chars=128)
        self.assertEqual(len(result), 128)


if __name__ == "__main__":
    unittest.main()
