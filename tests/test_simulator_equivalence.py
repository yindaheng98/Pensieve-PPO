"""Unit tests to verify Simulator behavior matches fixed_env.py Environment.

This module tests equivalence between:
- pensieve_ppo.core.simulator.Simulator vs src/fixed_env.py Environment

All tests in this file are deterministic (no random behavior).
For tests involving random wrappers (core.py equivalence), see test_random_simulator_equivalence.py.
"""

import unittest
import os

# Import original implementations from src
import load_trace as src_load_trace
import fixed_env as src_fixed_env

# Import our implementations
from pensieve_ppo.core.simulator.simulator import Simulator
from pensieve_ppo.core.video.player import VideoPlayer
from pensieve_ppo.core.video.data import VideoData
from pensieve_ppo.core.trace.simulator import TraceSimulator
from pensieve_ppo.core.trace.data import TraceData

# Use constants from original src/fixed_env.py
BITRATE_LEVELS = src_fixed_env.BITRATE_LEVELS
TOTAL_VIDEO_CHUNKS = src_fixed_env.TOTAL_VIDEO_CHUNCK
VIDEO_SIZE_FILE = src_fixed_env.VIDEO_SIZE_FILE
RANDOM_SEED = src_fixed_env.RANDOM_SEED

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


# ==============================================================================
# TEST CLASS 1: Exact match with fixed_env.py (deterministic, no noise)
# ==============================================================================

class TestSimulatorMatchesFixedEnv(unittest.TestCase):
    """Test that Simulator + TraceSimulator matches fixed_env.py Environment exactly."""

    @classmethod
    def setUpClass(cls):
        """Load shared test data."""
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

    def setUp(self):
        """Create fresh instances for each test."""
        video_player = VideoPlayer(self.video_data)
        trace_simulator = TraceSimulator(self.trace_data)
        self.simulator = Simulator(video_player, trace_simulator)

        self.fixed_env = src_fixed_env.Environment(
            self.all_cooked_time,
            self.all_cooked_bw,
            random_seed=RANDOM_SEED
        )

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

    def test_single_step_quality_0(self):
        """Test single step with quality 0."""
        result = self.simulator.step(0)
        env_result = self.fixed_env.get_video_chunk(0)
        self._compare_results(result, env_result)

    def test_single_step_quality_5(self):
        """Test single step with highest quality."""
        result = self.simulator.step(5)
        env_result = self.fixed_env.get_video_chunk(5)
        self._compare_results(result, env_result)

    def test_all_quality_levels(self):
        """Test each quality level individually."""
        for quality in range(BITRATE_LEVELS):
            with self.subTest(quality=quality):
                self.setUp()
                result = self.simulator.step(quality)
                env_result = self.fixed_env.get_video_chunk(quality)
                self._compare_results(result, env_result, f"Quality {quality}: ")

    def test_multiple_sequential_steps(self):
        """Test multiple sequential steps with varying qualities."""
        qualities = [0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0]

        for i, quality in enumerate(qualities):
            with self.subTest(step=i, quality=quality):
                result = self.simulator.step(quality)
                env_result = self.fixed_env.get_video_chunk(quality)
                self._compare_results(result, env_result, f"Step {i}: ")

    def test_complete_video_playback(self):
        """Test complete video playback (all 48 chunks)."""
        for i in range(TOTAL_VIDEO_CHUNKS):
            quality = i % BITRATE_LEVELS
            with self.subTest(chunk=i, quality=quality):
                result = self.simulator.step(quality)
                env_result = self.fixed_env.get_video_chunk(quality)
                self._compare_results(result, env_result, f"Chunk {i}: ")

        self.assertTrue(result.end_of_video, "Video should end after 48 chunks")

    def test_multiple_video_sessions(self):
        """Test multiple complete video sessions to verify trace cycling."""
        num_sessions = 3

        for session in range(num_sessions):
            for chunk in range(TOTAL_VIDEO_CHUNKS):
                quality = (session + chunk) % BITRATE_LEVELS
                with self.subTest(session=session, chunk=chunk):
                    result = self.simulator.step(quality)
                    env_result = self.fixed_env.get_video_chunk(quality)
                    self._compare_results(result, env_result,
                                          f"Session {session}, Chunk {chunk}: ")

    def test_buffer_overflow_scenario(self):
        """Test buffer overflow scenario with low quality chunks."""
        for i in range(20):
            with self.subTest(step=i):
                result = self.simulator.step(0)
                env_result = self.fixed_env.get_video_chunk(0)
                self._compare_results(result, env_result, f"Step {i}: ")

    def test_internal_state_consistency(self):
        """Test that internal states remain consistent with fixed_env."""
        for i in range(10):
            quality = i % BITRATE_LEVELS
            self.simulator.step(quality)
            self.fixed_env.get_video_chunk(quality)

            with self.subTest(step=i):
                self.assertAlmostEqual(
                    self.simulator.trace_simulator.buffer_size,
                    self.fixed_env.buffer_size,
                    places=6,
                    msg=f"Step {i}: Buffer size state mismatch"
                )
                self.assertEqual(
                    self.simulator.trace_simulator.trace_idx,
                    self.fixed_env.trace_idx,
                    msg=f"Step {i}: Trace index mismatch"
                )

    def test_trace_pointer_consistency(self):
        """Test that mahimahi pointer states match fixed_env."""
        for i in range(15):
            quality = (i * 2) % BITRATE_LEVELS
            self.simulator.step(quality)
            self.fixed_env.get_video_chunk(quality)

            trace_sim = self.simulator.trace_simulator
            with self.subTest(step=i):
                self.assertEqual(
                    trace_sim.mahimahi_ptr,
                    self.fixed_env.mahimahi_ptr,
                    msg=f"Step {i}: mahimahi_ptr mismatch"
                )
                self.assertAlmostEqual(
                    trace_sim.last_mahimahi_time,
                    self.fixed_env.last_mahimahi_time,
                    places=6,
                    msg=f"Step {i}: last_mahimahi_time mismatch"
                )

    def test_video_end_transition(self):
        """Test behavior at video end boundary matches fixed_env."""
        # Play through video to last chunk
        for _ in range(TOTAL_VIDEO_CHUNKS - 1):
            self.simulator.step(3)
            self.fixed_env.get_video_chunk(3)

        # Last chunk
        result = self.simulator.step(3)
        env_result = self.fixed_env.get_video_chunk(3)
        self.assertTrue(result.end_of_video)
        self._compare_results(result, env_result, "Last chunk: ")

        # First chunk of next video
        result = self.simulator.step(0)
        env_result = self.fixed_env.get_video_chunk(0)
        self._compare_results(result, env_result, "Next video first chunk: ")

    def test_large_chunk_download(self):
        """Test downloading large chunks (highest quality) matches fixed_env."""
        for i in range(5):
            with self.subTest(step=i):
                result = self.simulator.step(5)
                env_result = self.fixed_env.get_video_chunk(5)
                self._compare_results(result, env_result, f"Step {i}: ")

    def test_rapid_quality_switching(self):
        """Test rapid quality switching matches fixed_env."""
        qualities = [0, 5, 0, 5, 2, 4, 1, 3, 5, 0]
        for i, quality in enumerate(qualities):
            with self.subTest(step=i, quality=quality):
                result = self.simulator.step(quality)
                env_result = self.fixed_env.get_video_chunk(quality)
                self._compare_results(result, env_result, f"Step {i}: ")

    def test_trace_wrap_around(self):
        """Test trace wrap-around behavior matches fixed_env."""
        for i in range(100):
            quality = i % BITRATE_LEVELS
            with self.subTest(step=i):
                result = self.simulator.step(quality)
                env_result = self.fixed_env.get_video_chunk(quality)
                self._compare_results(result, env_result, f"Step {i}: ")


# ==============================================================================
# TEST CLASS 2: TraceSimulator initial state matches fixed_env
# ==============================================================================

class TestTraceSimulatorMatchesOriginal(unittest.TestCase):
    """Test TraceSimulator initial state and cycling matches fixed_env.py."""

    @classmethod
    def setUpClass(cls):
        """Load trace data."""
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

    @classmethod
    def tearDownClass(cls):
        """Restore original working directory."""
        os.chdir(cls.original_cwd)

    def test_initial_state_matches_fixed_env(self):
        """Test initial state matches fixed_env.py defaults."""
        trace_sim = TraceSimulator(self.trace_data)
        fixed_env = src_fixed_env.Environment(self.all_cooked_time, self.all_cooked_bw)

        self.assertEqual(trace_sim.trace_idx, fixed_env.trace_idx)
        self.assertEqual(trace_sim.mahimahi_ptr, fixed_env.mahimahi_ptr)
        self.assertEqual(trace_sim.buffer_size, fixed_env.buffer_size)

    def test_trace_cycling_matches_fixed_env(self):
        """Test that trace cycling logic matches fixed_env.py."""
        trace_sim = TraceSimulator(self.trace_data)
        num_traces = len(self.all_cooked_time)

        for i in range(num_traces * 2):
            trace_sim.on_video_finished()
            expected_trace_idx = (i + 1) % num_traces
            with self.subTest(video_finish=i + 1):
                self.assertEqual(
                    trace_sim.trace_idx,
                    expected_trace_idx,
                    msg=f"Trace index mismatch after {i + 1} video finishes"
                )


# ==============================================================================
# TEST CLASS 3: Download simulation exact match
# ==============================================================================

class TestDownloadSimulationMatchesFixedEnv(unittest.TestCase):
    """Test that download simulation exactly matches fixed_env."""

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

    def test_download_simulation_exact_match(self):
        """Test delay, buffer, and rebuffer exactly match fixed_env."""
        video_player = VideoPlayer(self.video_data)
        trace_sim = TraceSimulator(self.trace_data)
        simulator = Simulator(video_player, trace_sim)

        fixed_env = src_fixed_env.Environment(
            self.all_cooked_time,
            self.all_cooked_bw,
            random_seed=RANDOM_SEED
        )

        for i in range(20):
            quality = i % BITRATE_LEVELS
            result = simulator.step(quality)
            env_result = fixed_env.get_video_chunk(quality)
            delay, _, buffer_size, rebuf = env_result[:4]

            with self.subTest(step=i):
                self.assertAlmostEqual(result.delay, delay, places=6,
                                       msg=f"Step {i}: Delay mismatch")
                self.assertAlmostEqual(result.buffer_size, buffer_size, places=6,
                                       msg=f"Step {i}: Buffer size mismatch")
                self.assertAlmostEqual(result.rebuffer, rebuf, places=6,
                                       msg=f"Step {i}: Rebuffer mismatch")


if __name__ == '__main__':
    unittest.main(verbosity=2)
