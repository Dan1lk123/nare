"""
Integration test to verify all 5 bug fixes work together.
"""

import sys
import os

# Ensure we import from the local directory, not installed package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
from pathlib import Path

from nare.tools.safety import get_safety


def test_integration_all_fixes():
    """Integration test verifying all 5 fixes work together."""

    print("\n=== Integration Test: All 5 Bug Fixes ===\n")

    # Create temp directories
    with tempfile.TemporaryDirectory() as tmpdir:
        wd1 = os.path.join(tmpdir, "project1")
        wd2 = os.path.join(tmpdir, "project2")
        os.makedirs(wd1)
        os.makedirs(wd2)

        # Test Bug #1: SafetyLayer caching
        print("[+] Bug #1: Testing SafetyLayer caching...")
        s1a = get_safety(wd1)
        s1b = get_safety(wd1)
        s2 = get_safety(wd2)

        assert s1a is s1b, "Same working_dir should return same instance"
        assert s1a is not s2, "Different working_dirs should return different instances"
        print("  [OK] Caching works correctly")

        # Test Bug #1: Snapshots preserved
        test_file1 = Path(wd1) / "test.txt"
        test_file1.write_text("original content")

        s1a.snapshot(str(test_file1))
        assert len(s1a._snapshots) == 1

        s1c = get_safety(wd1)
        assert len(s1c._snapshots) == 1, "Snapshots should be preserved"
        print("  [OK] Snapshots preserved across calls")

        # Test Bug #2: Command validation
        print("\n[+] Bug #2: Testing enhanced command validation...")

        dangerous_cmds = [
            'r""m -rf /',
            'rm -rf \\x2f',
            '$(rm -rf /)',
            'rm -rf $HOME',
        ]

        for cmd in dangerous_cmds:
            allowed, reason = s1a.check_command(cmd)
            assert not allowed, f"Should block: {cmd}"
        print(f"  [OK] Blocked {len(dangerous_cmds)} bypass attempts")

        safe_cmds = ['ls -la', 'cat file.txt', 'echo hello']
        for cmd in safe_cmds:
            allowed, reason = s1a.check_command(cmd)
            assert allowed, f"Should allow: {cmd}"
        print(f"  [OK] Allowed {len(safe_cmds)} safe commands")

        # Test Bug #3: Temp file cleanup
        print("\n[+] Bug #3: Testing temp file cleanup...")

        test_file2 = Path(wd1) / "test2.txt"
        test_file2.write_text("content")

        s1a.snapshot(str(test_file2))
        assert len(s1a._temp_files) == 2, "Should track 2 temp files"

        temp_files = s1a._temp_files.copy()
        for tf in temp_files:
            assert os.path.exists(tf), "Temp file should exist"

        s1a._cleanup_temp_files()
        for tf in temp_files:
            assert not os.path.exists(tf), "Temp file should be deleted"
        print("  [OK] Temp files cleaned up successfully")

        # Test Bug #4: Windows translation (if on Windows)
        print("\n[+] Bug #4: Testing Windows command translation...")
        from nare.tools.builtin.builtin import _translate_unix_to_windows

        test_cases = [
            ("ls -la | grep test", ["dir", "findstr", "|"]),
            ("cat file.txt", ["type"]),
            ("find . -name '*.py'", ["dir", "/s/b"]),
        ]

        for unix_cmd, expected_parts in test_cases:
            result = _translate_unix_to_windows(unix_cmd)
            for part in expected_parts:
                assert part in result, f"Expected '{part}' in translation of '{unix_cmd}'"
        print(f"  [OK] Translated {len(test_cases)} commands correctly")

        # Test Bug #5: Thread safety (basic check)
        print("\n[+] Bug #5: Testing thread safety...")
        import threading

        errors = []

        def worker():
            try:
                s = get_safety(wd1)
                test_file = Path(wd1) / f"thread_{threading.current_thread().ident}.txt"
                test_file.write_text("content")
                s.snapshot(str(test_file))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        print("  [OK] Thread-safe operations completed successfully")

    print("\n=== All Integration Tests Passed ===\n")


if __name__ == "__main__":
    test_integration_all_fixes()
    print("SUCCESS: All 5 bug fixes verified!")
