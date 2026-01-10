"""Unit tests to verify Simulator with random wrappers matches core.py Environment.

This module tests equivalence between:
- pensieve_ppo.core.simulator.Simulator (with RandomTraceSimulator + NoiseTraceSimulator)
- src/core.py Environment

These tests involve random behavior and require special handling to ensure
both implementations consume the same random values.
"""

import unittest
import os
import numpy as np

# Import original implementations from src
import load_trace as src_load_trace
import fixed_env as src_fixed_env
from core import Environment as CoreEnvironment

# Import our implementations
from pensieve_ppo.core.simulator.simulator import Simulator
from pensieve_ppo.core.video.player import VideoPlayer
from pensieve_ppo.core.video.data import VideoData
from pensieve_ppo.core.trace.ext.random import RandomTraceSimulator
from pensieve_ppo.core.trace.ext.noise import NoiseTraceSimulator
from pensieve_ppo.core.trace.simulator import TraceSimulator
from pensieve_ppo.core.trace.data import TraceData

# Use constants from original src/fixed_env.py
BITRATE_LEVELS = src_fixed_env.BITRATE_LEVELS
TOTAL_VIDEO_CHUNKS = src_fixed_env.TOTAL_VIDEO_CHUNCK
VIDEO_SIZE_FILE = src_fixed_env.VIDEO_SIZE_FILE

# Use paths from src/test.py and src/train.py
TEST_TRACES = './test/'


def load_video_data() -> VideoData:
    """Load video chunk sizes using the same method as src/fixed_env.py Environment.__init__."""
    video_size = {}
    for bitrate in range(BITRATE_LEVELS):
        video_size[bitrate] = []
        with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
            for line in f:
                video_size[bitrate].append(int(line.split()[0]))

    return VideoData(
        video_size=video_size,
        bitrate_levels=BITRATE_LEVELS,
        num_chunks=TOTAL_VIDEO_CHUNKS,
    )


class TestSimulatorMatchesCoreEnv(unittest.TestCase):
    """Test that Simulator with wrappers matches core.py Environment exactly.

    core.py uses np.random for:
    1. Random trace selection in __init__ and when video ends
    2. Random starting point within each trace
    3. Multiplicative noise on download delay

    To test equivalence, both Simulator and CoreEnvironment must consume
    the same random values in the same order. This is achieved by:
    - Using random_seed=None to make wrappers use global np.random
    - Saving/restoring random state before each step
    """

    @classmethod
    def setUpClass(cls):
        """Load test data."""
        cls.original_cwd = os.getcwd()
        src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
        os.chdir(src_dir)

        cls.all_cooked_time, cls.all_cooked_bw, cls.all_file_names = \
            src_load_trace.load_trace(TEST_TRACES)

        cls.trace_data = TraceData(
            all_cooked_time=cls.all_cooked_time,
            all_cooked_bw=cls.all_cooked_bw,
            all_file_names=cls.all_file_names
        )
        cls.video_data = load_video_data()

    @classmethod
    def tearDownClass(cls):
        """Restore original working directory."""
        os.chdir(cls.original_cwd)

    def _compare_results(self, result, env_result, msg_prefix=""):
        """Compare all fields between Simulator result and Environment result."""
        (delay, sleep_time, buffer_size, rebuf,
         video_chunk_size, next_sizes, end_of_video, chunks_remain) = env_result

        self.assertAlmostEqual(result.delay, delay, places=6,
                               msg=f"{msg_prefix}Delay mismatch")
        self.assertAlmostEqual(result.sleep_time, sleep_time, places=6,
                               msg=f"{msg_prefix}Sleep time mismatch")
        self.assertAlmostEqual(result.buffer_size, buffer_size, places=6,
                               msg=f"{msg_prefix}Buffer size mismatch")
        self.assertAlmostEqual(result.rebuffer, rebuf, places=6,
                               msg=f"{msg_prefix}Rebuffer mismatch")
        self.assertEqual(result.video_chunk_size, video_chunk_size,
                         msg=f"{msg_prefix}Video chunk size mismatch")
        self.assertEqual(result.next_video_chunk_sizes, next_sizes,
                         msg=f"{msg_prefix}Next chunk sizes mismatch")
        self.assertEqual(result.end_of_video, end_of_video,
                         msg=f"{msg_prefix}End of video mismatch")
        self.assertEqual(result.video_chunk_remain, chunks_remain,
                         msg=f"{msg_prefix}Chunks remaining mismatch")

    def _create_simulator_matching_core(self):
        """Create Simulator that uses global np.random to match core.py exactly.

        Both RandomTraceSimulator and NoiseTraceSimulator are configured to use
        the global np.random module (by passing random_seed=None), which matches
        how core.py uses np.random for all random operations.
        """
        video_player = VideoPlayer(self.video_data)
        base_trace_sim = TraceSimulator(self.trace_data)

        # Use None to make wrappers use global np.random
        random_trace_sim = RandomTraceSimulator(base_trace_sim, random_seed=None)
        noisy_trace_sim = NoiseTraceSimulator(random_trace_sim, random_seed=None)

        return Simulator(video_player, noisy_trace_sim)

    def test_single_step_match(self):
        """Test single step matches core.py Environment."""
        seed = 42

        np.random.seed(seed)
        simulator = self._create_simulator_matching_core()

        np.random.seed(seed)
        core_env = CoreEnvironment(self.all_cooked_time, self.all_cooked_bw, random_seed=seed)

        # Save random state, run simulator, restore, then run core_env
        # This ensures both consume the same random values
        state = np.random.get_state()
        result = simulator.step(2)
        np.random.set_state(state)
        env_result = core_env.get_video_chunk(2)
        self._compare_results(result, env_result)

    def test_multiple_steps_match(self):
        """Test multiple sequential steps match core.py Environment."""
        seed = 42

        np.random.seed(seed)
        simulator = self._create_simulator_matching_core()

        np.random.seed(seed)
        core_env = CoreEnvironment(self.all_cooked_time, self.all_cooked_bw, random_seed=seed)

        for i in range(20):
            quality = i % BITRATE_LEVELS
            with self.subTest(step=i, quality=quality):
                # Save/restore random state so both use same random values
                state = np.random.get_state()
                result = simulator.step(quality)
                np.random.set_state(state)
                env_result = core_env.get_video_chunk(quality)
                self._compare_results(result, env_result, f"Step {i}: ")

    def test_complete_video_match(self):
        """Test complete video playback matches core.py Environment."""
        seed = 42

        np.random.seed(seed)
        simulator = self._create_simulator_matching_core()

        np.random.seed(seed)
        core_env = CoreEnvironment(self.all_cooked_time, self.all_cooked_bw, random_seed=seed)

        for i in range(TOTAL_VIDEO_CHUNKS):
            quality = i % BITRATE_LEVELS
            with self.subTest(chunk=i, quality=quality):
                # Save/restore random state so both use same random values
                state = np.random.get_state()
                result = simulator.step(quality)
                np.random.set_state(state)
                env_result = core_env.get_video_chunk(quality)
                self._compare_results(result, env_result, f"Chunk {i}: ")


if __name__ == '__main__':
    unittest.main(verbosity=2)
