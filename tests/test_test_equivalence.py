"""Unit tests to verify pensieve_ppo/test.py behavior matches src/test.py.

This module tests equivalence between:
- pensieve_ppo/test.py (new implementation) vs src/test.py (original)

Both implementations are run with the same pretrained model and test traces.
The test compares the output log files to ensure behavioral equivalence.

Testing strategy:
- Run src/test.py via subprocess (command line)
- Run pensieve_ppo/test.py via code (testing function)
- Compare the generated log files line by line
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

# Import our implementation
from pensieve_ppo.test import prepare_testing, testing as run_testing
from pensieve_ppo.defaults import VIDEO_BIT_RATE

# Paths
SRC_DIR = os.path.join(os.path.dirname(__file__), '..', 'src')
PRETRAINED_MODEL = os.path.join(SRC_DIR, 'pretrain', 'nn_model_ep_155400.pth')
TEST_TRACES = os.path.join(SRC_DIR, 'test')

# Constants matching src/test.py
# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L23
RANDOM_SEED = 42


class TestTestEquivalence(unittest.TestCase):
    """Test that pensieve_ppo/test.py produces identical output as src/test.py."""

    @classmethod
    def setUpClass(cls):
        """Set up test directories and verify prerequisites."""
        if not os.path.exists(PRETRAINED_MODEL):
            raise unittest.SkipTest(f"Pretrained model not found: {PRETRAINED_MODEL}")

        if not os.path.exists(TEST_TRACES):
            raise unittest.SkipTest(f"Test traces not found: {TEST_TRACES}")

        # Create temporary directories for log outputs
        cls.temp_dir = tempfile.mkdtemp(prefix='pensieve_test_')
        cls.src_log_dir = os.path.join(cls.temp_dir, 'src_logs')
        cls.our_log_dir = os.path.join(cls.temp_dir, 'our_logs')
        os.makedirs(cls.src_log_dir, exist_ok=True)
        os.makedirs(cls.our_log_dir, exist_ok=True)

        # Store original working directory
        cls.original_cwd = os.getcwd()

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directories."""
        os.chdir(cls.original_cwd)
        if hasattr(cls, 'temp_dir') and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def _run_src_test(self, log_dir: str) -> None:
        """Run src/test.py via subprocess.

        Args:
            log_dir: Directory for log output files.
        """
        # Create a modified test.py in the src directory (so imports work)
        modified_test_py = os.path.join(SRC_DIR, '_test_modified_temp.py')

        # Read original src/test.py
        src_test_path = os.path.join(SRC_DIR, 'test.py')
        with open(src_test_path, 'r') as f:
            src_content = f.read()

        # Replace LOG_FILE path with our temp directory
        log_prefix = os.path.join(log_dir, 'log_sim_ppo').replace('\\', '/')
        modified_content = src_content.replace(
            "LOG_FILE = './test_results/log_sim_ppo'",
            f"LOG_FILE = '{log_prefix}'"
        )

        # Write modified test.py to src directory
        try:
            with open(modified_test_py, 'w') as f:
                f.write(modified_content)

            # Run modified test.py from src directory
            result = subprocess.run(
                [sys.executable, modified_test_py, PRETRAINED_MODEL],
                cwd=SRC_DIR,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"src/test.py failed with return code {result.returncode}\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )
        finally:
            # Clean up temporary file
            if os.path.exists(modified_test_py):
                os.remove(modified_test_py)

    def _run_our_test(self, log_dir: str) -> None:
        """Run pensieve_ppo/test.py via code.

        Args:
            log_dir: Directory for log output files.
        """
        # Change to src directory so trace paths resolve correctly
        os.chdir(SRC_DIR)

        # Set random seed to match src/test.py
        np.random.seed(RANDOM_SEED)

        # Prepare env and agent
        env, agent = prepare_testing(
            trace_folder='./test/',
            model_path=PRETRAINED_MODEL,
            agent_name='ppo',
            levels_quality=VIDEO_BIT_RATE,
        )

        # Run testing
        log_file_prefix = os.path.join(log_dir, 'log_sim_ppo')
        run_testing(
            env=env,
            agent=agent,
            log_file_prefix=log_file_prefix,
        )

    def _parse_log_line(self, line: str) -> dict:
        """Parse a log line into components.

        Log format: time_stamp, bit_rate, buffer_size, rebuf, video_chunk_size, delay, entropy, reward

        Args:
            line: Log line string.

        Returns:
            Dict with parsed values, or None for empty lines.
        """
        line = line.strip()
        if not line:
            return None

        parts = line.split('\t')
        if len(parts) != 8:
            return None

        return {
            'time_stamp': float(parts[0]),
            'bit_rate': float(parts[1]),
            'buffer_size': float(parts[2]),
            'rebuf': float(parts[3]),
            'video_chunk_size': float(parts[4]),
            'delay': float(parts[5]),
            'entropy': float(parts[6]),
            'reward': float(parts[7]),
        }

    def _compare_logs(self, src_log_path: str, our_log_path: str) -> list:
        """Compare two log files and return differences.

        Args:
            src_log_path: Path to src/test.py log file.
            our_log_path: Path to our log file.

        Returns:
            List of difference descriptions.
        """
        differences = []

        # Read both logs
        with open(src_log_path, 'r') as f:
            src_lines = f.readlines()
        with open(our_log_path, 'r') as f:
            our_lines = f.readlines()

        # Compare line counts
        if len(src_lines) != len(our_lines):
            differences.append(
                f"Line count mismatch: src={len(src_lines)}, ours={len(our_lines)}"
            )

        # Compare each line
        min_lines = min(len(src_lines), len(our_lines))
        for i in range(min_lines):
            src_parsed = self._parse_log_line(src_lines[i])
            our_parsed = self._parse_log_line(our_lines[i])

            # Skip empty lines
            if src_parsed is None and our_parsed is None:
                continue

            if src_parsed is None or our_parsed is None:
                differences.append(f"Line {i+1}: One is empty, other is not")
                continue

            # Compare each field with tolerance
            for key in src_parsed.keys():
                src_val = src_parsed[key]
                our_val = our_parsed[key]

                # Use relative tolerance for floating point comparison
                if not np.isclose(src_val, our_val, rtol=1e-6, atol=1e-9):
                    differences.append(
                        f"Line {i+1}, {key}: src={src_val}, ours={our_val}, diff={abs(src_val-our_val)}"
                    )

        return differences

    def test_full_test_equivalence(self):
        """Test that full testing run produces identical logs."""
        # Run src/test.py
        self._run_src_test(self.src_log_dir)

        # Run our test
        self._run_our_test(self.our_log_dir)

        # Get list of log files
        src_logs = sorted([f for f in os.listdir(self.src_log_dir) if f.startswith('log_sim_ppo_')])
        our_logs = sorted([f for f in os.listdir(self.our_log_dir) if f.startswith('log_sim_ppo_')])

        # Should have same number of log files
        self.assertEqual(len(src_logs), len(our_logs),
                         f"Number of log files mismatch: src={len(src_logs)}, ours={len(our_logs)}")

        # Compare each log file
        all_differences = {}
        for src_log, our_log in zip(src_logs, our_logs):
            # File names should match
            self.assertEqual(src_log, our_log,
                             f"Log file name mismatch: src={src_log}, ours={our_log}")

            src_path = os.path.join(self.src_log_dir, src_log)
            our_path = os.path.join(self.our_log_dir, our_log)

            differences = self._compare_logs(src_path, our_path)
            if differences:
                all_differences[src_log] = differences

        # Report all differences
        if all_differences:
            error_msg = "Log differences found:\n"
            for log_file, diffs in all_differences.items():
                error_msg += f"\n{log_file}:\n"
                for diff in diffs[:10]:  # Limit to first 10 differences per file
                    error_msg += f"  {diff}\n"
                if len(diffs) > 10:
                    error_msg += f"  ... and {len(diffs) - 10} more differences\n"
            self.fail(error_msg)

    def test_first_trace_equivalence(self):
        """Test equivalence for just the first trace (faster test)."""
        # This test is a subset of full test but useful for quick debugging

        # Run both tests
        self._run_src_test(self.src_log_dir)
        self._run_our_test(self.our_log_dir)

        # Compare first log file
        src_logs = sorted([f for f in os.listdir(self.src_log_dir) if f.startswith('log_sim_ppo_')])
        our_logs = sorted([f for f in os.listdir(self.our_log_dir) if f.startswith('log_sim_ppo_')])

        self.assertTrue(len(src_logs) > 0, "No src log files generated")
        self.assertTrue(len(our_logs) > 0, "No our log files generated")

        first_src = src_logs[0]
        first_our = our_logs[0]

        self.assertEqual(first_src, first_our,
                         f"First log file name mismatch: src={first_src}, ours={first_our}")

        src_path = os.path.join(self.src_log_dir, first_src)
        our_path = os.path.join(self.our_log_dir, first_our)

        differences = self._compare_logs(src_path, our_path)
        if differences:
            error_msg = f"First trace ({first_src}) differences:\n"
            for diff in differences:
                error_msg += f"  {diff}\n"
            self.fail(error_msg)


class TestLogFormat(unittest.TestCase):
    """Test that log format matches src/test.py expectations."""

    @classmethod
    def setUpClass(cls):
        """Set up for log format tests."""
        if not os.path.exists(PRETRAINED_MODEL):
            raise unittest.SkipTest(f"Pretrained model not found: {PRETRAINED_MODEL}")

        cls.temp_dir = tempfile.mkdtemp(prefix='pensieve_log_format_')
        cls.original_cwd = os.getcwd()

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        os.chdir(cls.original_cwd)
        if hasattr(cls, 'temp_dir') and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_log_file_naming(self):
        """Test that log files are named correctly."""
        os.chdir(SRC_DIR)
        np.random.seed(RANDOM_SEED)

        log_dir = os.path.join(self.temp_dir, 'naming_test')
        os.makedirs(log_dir, exist_ok=True)

        env, agent = prepare_testing(
            trace_folder='./test/',
            model_path=PRETRAINED_MODEL,
            agent_name='ppo',
        )

        log_prefix = os.path.join(log_dir, 'log_sim_ppo')
        run_testing(env=env, agent=agent, log_file_prefix=log_prefix)

        # Check log files exist
        log_files = os.listdir(log_dir)
        self.assertTrue(len(log_files) > 0, "No log files created")

        # All files should start with 'log_sim_ppo_'
        for log_file in log_files:
            self.assertTrue(
                log_file.startswith('log_sim_ppo_'),
                f"Log file has wrong prefix: {log_file}"
            )

    def test_log_line_format(self):
        """Test that each log line has correct format."""
        os.chdir(SRC_DIR)
        np.random.seed(RANDOM_SEED)

        log_dir = os.path.join(self.temp_dir, 'format_test')
        os.makedirs(log_dir, exist_ok=True)

        env, agent = prepare_testing(
            trace_folder='./test/',
            model_path=PRETRAINED_MODEL,
            agent_name='ppo',
        )

        log_prefix = os.path.join(log_dir, 'log_sim_ppo')
        run_testing(env=env, agent=agent, log_file_prefix=log_prefix)

        # Read first log file
        log_files = sorted(os.listdir(log_dir))
        self.assertTrue(len(log_files) > 0)

        with open(os.path.join(log_dir, log_files[0]), 'r') as f:
            lines = f.readlines()

        # Should have 48 chunks + 1 empty line
        non_empty_lines = [l for l in lines if l.strip()]
        self.assertEqual(len(non_empty_lines), 48,
                         f"Expected 48 data lines, got {len(non_empty_lines)}")

        # Check format of each line
        for i, line in enumerate(non_empty_lines):
            parts = line.strip().split('\t')
            self.assertEqual(len(parts), 8,
                             f"Line {i+1}: Expected 8 tab-separated fields, got {len(parts)}")

            # Verify each field is a valid number
            for j, part in enumerate(parts):
                try:
                    float(part)
                except ValueError:
                    self.fail(f"Line {i+1}, field {j+1}: Invalid number '{part}'")

    def test_chunk_count_per_trace(self):
        """Test that each trace produces exactly 48 log lines."""
        os.chdir(SRC_DIR)
        np.random.seed(RANDOM_SEED)

        log_dir = os.path.join(self.temp_dir, 'chunk_count_test')
        os.makedirs(log_dir, exist_ok=True)

        env, agent = prepare_testing(
            trace_folder='./test/',
            model_path=PRETRAINED_MODEL,
            agent_name='ppo',
        )

        log_prefix = os.path.join(log_dir, 'log_sim_ppo')
        run_testing(env=env, agent=agent, log_file_prefix=log_prefix)

        # Check each log file
        log_files = os.listdir(log_dir)
        for log_file in log_files:
            with open(os.path.join(log_dir, log_file), 'r') as f:
                lines = f.readlines()

            non_empty_lines = [l for l in lines if l.strip()]
            self.assertEqual(len(non_empty_lines), 48,
                             f"{log_file}: Expected 48 lines, got {len(non_empty_lines)}")


class TestReproducibility(unittest.TestCase):
    """Test that testing is reproducible with same seed."""

    @classmethod
    def setUpClass(cls):
        """Set up for reproducibility tests."""
        if not os.path.exists(PRETRAINED_MODEL):
            raise unittest.SkipTest(f"Pretrained model not found: {PRETRAINED_MODEL}")

        cls.temp_dir = tempfile.mkdtemp(prefix='pensieve_repro_')
        cls.original_cwd = os.getcwd()

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        os.chdir(cls.original_cwd)
        if hasattr(cls, 'temp_dir') and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_same_seed_same_output(self):
        """Test that same random seed produces same output."""
        os.chdir(SRC_DIR)

        log_dir1 = os.path.join(self.temp_dir, 'run1')
        log_dir2 = os.path.join(self.temp_dir, 'run2')
        os.makedirs(log_dir1, exist_ok=True)
        os.makedirs(log_dir2, exist_ok=True)

        # Run 1
        np.random.seed(RANDOM_SEED)
        env1, agent1 = prepare_testing(
            trace_folder='./test/',
            model_path=PRETRAINED_MODEL,
            agent_name='ppo',
        )
        run_testing(env=env1, agent=agent1, log_file_prefix=os.path.join(log_dir1, 'log'))

        # Run 2
        np.random.seed(RANDOM_SEED)
        env2, agent2 = prepare_testing(
            trace_folder='./test/',
            model_path=PRETRAINED_MODEL,
            agent_name='ppo',
        )
        run_testing(env=env2, agent=agent2, log_file_prefix=os.path.join(log_dir2, 'log'))

        # Compare outputs
        logs1 = sorted(os.listdir(log_dir1))
        logs2 = sorted(os.listdir(log_dir2))

        self.assertEqual(logs1, logs2, "Different log files generated")

        for log_name in logs1:
            with open(os.path.join(log_dir1, log_name), 'r') as f1, \
                 open(os.path.join(log_dir2, log_name), 'r') as f2:
                content1 = f1.read()
                content2 = f2.read()
                self.assertEqual(content1, content2,
                                 f"Log {log_name} differs between runs")


if __name__ == '__main__':
    unittest.main(verbosity=2)
