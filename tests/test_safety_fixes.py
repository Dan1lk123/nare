"""
Unit tests for SafetyLayer bug fixes.

Tests all 5 critical bugs:
1. SafetyLayer amnesia (caching per working_dir)
2. Shell injection bypass prevention
3. Temporary file cleanup
4. Windows command translation
5. Thread safety
"""

import os
import tempfile
import threading
import time
from pathlib import Path

import pytest

from nare.tools.safety import SafetyLayer, get_safety
from nare.tools.builtin.builtin import _translate_unix_to_windows


class TestBug1_SafetyLayerCaching:
    """Test Bug #1: SafetyLayer instances are cached per working_dir."""

    def test_same_working_dir_returns_same_instance(self, tmp_path):
        """Verify same working_dir returns the same SafetyLayer instance."""
        wd = str(tmp_path)

        s1 = get_safety(wd)
        s2 = get_safety(wd)

        assert s1 is s2, "Should return same instance for same working_dir"

    def test_different_working_dirs_return_different_instances(self, tmp_path):
        """Verify different working_dirs return different instances."""
        wd1 = str(tmp_path / "dir1")
        wd2 = str(tmp_path / "dir2")
        os.makedirs(wd1, exist_ok=True)
        os.makedirs(wd2, exist_ok=True)

        s1 = get_safety(wd1)
        s2 = get_safety(wd2)

        assert s1 is not s2, "Should return different instances for different working_dirs"

    def test_snapshots_preserved_across_calls(self, tmp_path):
        """Verify snapshots are preserved when get_safety is called multiple times."""
        wd = str(tmp_path)
        test_file = tmp_path / "test.txt"
        test_file.write_text("original content")

        # First call: create snapshot
        s1 = get_safety(wd)
        s1.snapshot(str(test_file))
        snapshot_count_1 = len(s1._snapshots)

        # Second call: should return same instance with snapshots intact
        s2 = get_safety(wd)
        snapshot_count_2 = len(s2._snapshots)

        assert snapshot_count_1 == snapshot_count_2 == 1, "Snapshots should be preserved"
        assert str(test_file.resolve()) in s2._snapshots, "Snapshot should exist"


class TestBug2_CommandValidationBypass:
    """Test Bug #2: Enhanced command validation catches bypass attempts."""

    def test_quoted_rm_blocked(self):
        """Test that quoted dangerous commands are blocked."""
        safety = SafetyLayer()

        # Various bypass attempts
        bypass_attempts = [
            'r""m -rf /',
            "r''m -rf /",
            'r`m -rf /',
            '"rm" -rf /',
            "'rm' -rf /",
        ]

        for cmd in bypass_attempts:
            allowed, reason = safety.check_command(cmd)
            assert not allowed, f"Should block: {cmd}"

    def test_encoded_rm_blocked(self):
        """Test that encoded dangerous commands are blocked."""
        safety = SafetyLayer()

        # Hex-encoded slash
        allowed, reason = safety.check_command('rm -rf \\x2f')
        assert not allowed, "Should block hex-encoded slash"

    def test_variable_expansion_rm_blocked(self):
        """Test that variable expansion attempts are blocked."""
        safety = SafetyLayer()

        dangerous_cmds = [
            'rm -rf $HOME',
            'rm -rf ~',
            'rm -rf ${HOME}',
        ]

        for cmd in dangerous_cmds:
            allowed, reason = safety.check_command(cmd)
            assert not allowed, f"Should block: {cmd}"

    def test_command_injection_blocked(self):
        """Test that command injection patterns are blocked."""
        safety = SafetyLayer()

        injection_attempts = [
            '$(rm -rf /)',
            '`rm -rf /`',
            'echo test && rm -rf /',
            'echo test || rm -rf /',
            'echo test; rm -rf /',
        ]

        for cmd in injection_attempts:
            allowed, reason = safety.check_command(cmd)
            assert not allowed, f"Should block injection: {cmd}"

    def test_safe_commands_allowed(self):
        """Test that safe commands are still allowed."""
        safety = SafetyLayer()

        safe_cmds = [
            'ls -la',
            'cat file.txt',
            'echo "hello world"',
            'grep pattern file.txt',
            'find . -name "*.py"',
        ]

        for cmd in safe_cmds:
            allowed, reason = safety.check_command(cmd)
            assert allowed, f"Should allow safe command: {cmd}"


class TestBug3_TempFileCleanup:
    """Test Bug #3: Temporary files are cleaned up properly."""

    def test_temp_files_tracked(self, tmp_path):
        """Test that temp files are tracked in _temp_files list."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        safety = SafetyLayer(working_dir=str(tmp_path))
        safety.snapshot(str(test_file))

        assert len(safety._temp_files) == 1, "Should track temp file"
        assert os.path.exists(safety._temp_files[0]), "Temp file should exist"

    def test_cleanup_removes_temp_files(self, tmp_path):
        """Test that _cleanup_temp_files removes all temp files."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        safety = SafetyLayer(working_dir=str(tmp_path))
        safety.snapshot(str(test_file))

        temp_files = safety._temp_files.copy()
        safety._cleanup_temp_files()

        for temp_file in temp_files:
            assert not os.path.exists(temp_file), f"Temp file should be deleted: {temp_file}"
        assert len(safety._temp_files) == 0, "Temp files list should be empty"

    def test_rollback_removes_from_tracking(self, tmp_path):
        """Test that rollback removes temp file from tracking list."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("original")

        safety = SafetyLayer(working_dir=str(tmp_path))
        safety.snapshot(str(test_file))

        # Modify file
        test_file.write_text("modified")

        # Rollback
        safety.rollback(str(test_file))

        assert len(safety._temp_files) == 0, "Should remove from tracking after rollback"

    def test_clear_snapshots_removes_from_tracking(self, tmp_path):
        """Test that clear_snapshots removes temp files from tracking."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        safety = SafetyLayer(working_dir=str(tmp_path))
        safety.snapshot(str(test_file))

        safety.clear_snapshots()

        assert len(safety._temp_files) == 0, "Should remove from tracking after clear"


class TestBug4_WindowsTranslation:
    """Test Bug #4: Robust Windows command translation."""

    def test_pipe_translation(self):
        """Test that pipes are handled correctly."""
        cmd = "ls -la | grep test"
        result = _translate_unix_to_windows(cmd)

        assert "dir" in result, "Should translate ls to dir"
        assert "findstr" in result, "Should translate grep to findstr"
        assert "|" in result, "Should preserve pipe"

    def test_redirection_translation(self):
        """Test that redirections are handled correctly."""
        cmd = "ls > output.txt"
        result = _translate_unix_to_windows(cmd)

        assert "dir" in result, "Should translate ls to dir"
        assert ">" in result, "Should preserve redirection"
        assert "output.txt" in result, "Should preserve target file"

    def test_complex_find_translation(self):
        """Test complex find commands."""
        test_cases = [
            ("find . -name '*.py'", "dir /s/b *.py*"),
            ("find . -type f", "dir /s/b /a-d"),
            ("find . -type d", "dir /s/b /ad"),
            ("find .", "dir /s/b"),
        ]

        for unix_cmd, expected_pattern in test_cases:
            result = _translate_unix_to_windows(unix_cmd)
            assert expected_pattern in result, f"Failed to translate: {unix_cmd}"

    def test_basic_command_translation(self):
        """Test basic command translations."""
        test_cases = [
            ("cat file.txt", "type file.txt"),
            ("pwd", "cd"),
            ("cp src dst", "copy src dst"),
            ("mv src dst", "move src dst"),
        ]

        for unix_cmd, expected in test_cases:
            result = _translate_unix_to_windows(unix_cmd)
            assert result == expected, f"Failed: {unix_cmd} -> {result} (expected {expected})"


class TestBug5_ThreadSafety:
    """Test Bug #5: Thread safety of SafetyLayer."""

    def test_concurrent_get_safety_calls(self, tmp_path):
        """Test that concurrent get_safety calls are thread-safe."""
        wd1 = str(tmp_path / "dir1")
        wd2 = str(tmp_path / "dir2")
        wd3 = str(tmp_path / "dir3")
        os.makedirs(wd1, exist_ok=True)
        os.makedirs(wd2, exist_ok=True)
        os.makedirs(wd3, exist_ok=True)

        results = []

        def worker(wd, worker_id):
            for _ in range(10):
                s = get_safety(wd)
                results.append((worker_id, wd, id(s)))
                time.sleep(0.001)

        threads = []
        for i in range(10):
            wd = [wd1, wd2, wd3][i % 3]
            t = threading.Thread(target=worker, args=(wd, i))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Verify same working_dir always returns same instance ID
        wd_to_ids = {}
        for worker_id, wd, instance_id in results:
            if wd not in wd_to_ids:
                wd_to_ids[wd] = set()
            wd_to_ids[wd].add(instance_id)

        for wd, ids in wd_to_ids.items():
            assert len(ids) == 1, f"Working dir {wd} should have only one instance ID"

    def test_concurrent_snapshot_operations(self, tmp_path):
        """Test that concurrent snapshot operations are thread-safe."""
        # Create test files
        test_files = []
        for i in range(10):
            f = tmp_path / f"test{i}.txt"
            f.write_text(f"content {i}")
            test_files.append(str(f))

        safety = SafetyLayer(working_dir=str(tmp_path))
        errors = []

        def worker(file_path):
            try:
                safety.snapshot(file_path)
            except Exception as e:
                errors.append(e)

        threads = []
        for file_path in test_files:
            t = threading.Thread(target=worker, args=(file_path,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Should not have errors: {errors}"
        assert len(safety._snapshots) == 10, "Should have 10 snapshots"

    def test_concurrent_rollback_operations(self, tmp_path):
        """Test that concurrent rollback operations are thread-safe."""
        # Create and snapshot test files
        test_files = []
        for i in range(5):
            f = tmp_path / f"test{i}.txt"
            f.write_text(f"original {i}")
            test_files.append(str(f))

        safety = SafetyLayer(working_dir=str(tmp_path))

        # Create snapshots
        for file_path in test_files:
            safety.snapshot(file_path)

        # Modify files
        for i, file_path in enumerate(test_files):
            Path(file_path).write_text(f"modified {i}")

        errors = []

        def worker(file_path):
            try:
                safety.rollback(file_path)
            except Exception as e:
                errors.append(e)

        threads = []
        for file_path in test_files:
            t = threading.Thread(target=worker, args=(file_path,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Should not have errors: {errors}"

        # Verify all files were rolled back
        for i, file_path in enumerate(test_files):
            content = Path(file_path).read_text()
            assert content == f"original {i}", f"File {i} should be rolled back"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
