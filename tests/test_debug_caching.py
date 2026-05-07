"""
Debug test to understand the caching issue.
"""

import os
import tempfile
from pathlib import Path

from nare.tools.safety import get_safety, _safety_instances


def test_debug_caching():
    """Debug test for caching."""

    with tempfile.TemporaryDirectory() as tmpdir:
        wd1 = os.path.join(tmpdir, "project1")
        os.makedirs(wd1)

        print(f"Working dir: {wd1}")
        print(f"Absolute path: {os.path.abspath(wd1)}")
        print(f"Initial cache: {_safety_instances}")

        s1a = get_safety(wd1)
        print(f"After first call - cache: {_safety_instances}")
        print(f"s1a id: {id(s1a)}")

        s1b = get_safety(wd1)
        print(f"After second call - cache: {_safety_instances}")
        print(f"s1b id: {id(s1b)}")

        print(f"s1a is s1b: {s1a is s1b}")
        print(f"s1a == s1b: {s1a == s1b}")


if __name__ == "__main__":
    test_debug_caching()
