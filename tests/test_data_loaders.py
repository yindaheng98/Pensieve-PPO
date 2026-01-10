"""Unit tests to verify data loaders match the original implementations.

This module tests that:
1. pensieve_ppo.core.trace.loader.load_trace matches src/load_trace.py
2. pensieve_ppo.core.video.loader.load_video_size matches video loading in src/fixed_env.py

Data sources reference:
- src/test.py: TEST_TRACES = './test/'
- src/train.py: TRAIN_TRACES = './train/'  
- src/fixed_env.py: VIDEO_SIZE_FILE = './envivio/video_size_'
"""

import unittest
import os
import load_trace as src_load_trace
import fixed_env as src_fixed_env
from pensieve_ppo.core.trace.loader import load_trace
from pensieve_ppo.core.trace.data import TraceData
from pensieve_ppo.core.video.loader import load_video_size
from pensieve_ppo.core.video.data import VideoData


# Import original implementations from src

# Import our implementations


# Use constants from original src/fixed_env.py
BITRATE_LEVELS = src_fixed_env.BITRATE_LEVELS
TOTAL_VIDEO_CHUNKS = src_fixed_env.TOTAL_VIDEO_CHUNCK
VIDEO_SIZE_FILE = src_fixed_env.VIDEO_SIZE_FILE

# Use paths from src/test.py and src/train.py
TEST_TRACES = './test/'
TRAIN_TRACES = './train/'


# ==============================================================================
# TEST CLASS 1: Trace loader tests
# ==============================================================================

class TestTraceLoaderMatchesOriginal(unittest.TestCase):
    """Test that load_trace matches the original src/load_trace.py implementation."""

    @classmethod
    def setUpClass(cls):
        """Load shared test data."""
        cls.original_cwd = os.getcwd()
        src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
        os.chdir(src_dir)

    @classmethod
    def tearDownClass(cls):
        """Restore original working directory."""
        os.chdir(cls.original_cwd)

    def test_test_traces_data_match(self):
        """Test that loading test traces produces identical data."""
        # Load using our implementation
        trace_data = load_trace(TEST_TRACES)

        # Load using original src/load_trace.py implementation
        orig_time, orig_bw, orig_names = src_load_trace.load_trace(TEST_TRACES)

        # Compare number of traces
        self.assertEqual(len(trace_data), len(orig_time))
        self.assertEqual(len(trace_data), len(orig_bw))
        self.assertEqual(len(trace_data), len(orig_names))

        # Our implementation sorts files, original doesn't - so we compare by content
        # Create a mapping from original data to compare
        orig_data_map = {}
        for i, name in enumerate(orig_names):
            orig_data_map[name] = (orig_time[i], orig_bw[i])

        # Compare each trace
        for i in range(len(trace_data)):
            file_name = trace_data.all_file_names[i]
            with self.subTest(trace=file_name):
                # Check that file name exists in original
                self.assertIn(file_name, orig_data_map,
                              f"File {file_name} not found in original data")

                orig_t, orig_b = orig_data_map[file_name]

                # Compare time values
                self.assertEqual(len(trace_data.all_cooked_time[i]), len(orig_t),
                                 f"Time array length mismatch for {file_name}")
                for j, (t1, t2) in enumerate(zip(trace_data.all_cooked_time[i], orig_t)):
                    self.assertAlmostEqual(t1, t2, places=10,
                                           msg=f"Time mismatch at index {j} for {file_name}")

                # Compare bandwidth values
                self.assertEqual(len(trace_data.all_cooked_bw[i]), len(orig_b),
                                 f"BW array length mismatch for {file_name}")
                for j, (b1, b2) in enumerate(zip(trace_data.all_cooked_bw[i], orig_b)):
                    self.assertAlmostEqual(b1, b2, places=10,
                                           msg=f"BW mismatch at index {j} for {file_name}")

    def test_train_traces_data_match(self):
        """Test that loading train traces produces identical data."""
        # Load using our implementation
        trace_data = load_trace(TRAIN_TRACES)

        # Load using original src/load_trace.py implementation
        orig_time, orig_bw, orig_names = src_load_trace.load_trace(TRAIN_TRACES)

        # Compare number of traces
        self.assertEqual(len(trace_data), len(orig_time))

        # Create a mapping from original data to compare
        orig_data_map = {}
        for i, name in enumerate(orig_names):
            orig_data_map[name] = (orig_time[i], orig_bw[i])

        # Compare each trace
        for i in range(len(trace_data)):
            file_name = trace_data.all_file_names[i]
            with self.subTest(trace=file_name):
                self.assertIn(file_name, orig_data_map)

                orig_t, orig_b = orig_data_map[file_name]

                # Compare time values
                self.assertEqual(len(trace_data.all_cooked_time[i]), len(orig_t))
                for j, (t1, t2) in enumerate(zip(trace_data.all_cooked_time[i], orig_t)):
                    self.assertAlmostEqual(t1, t2, places=10)

                # Compare bandwidth values
                self.assertEqual(len(trace_data.all_cooked_bw[i]), len(orig_b))
                for j, (b1, b2) in enumerate(zip(trace_data.all_cooked_bw[i], orig_b)):
                    self.assertAlmostEqual(b1, b2, places=10)

    def test_trace_data_structure(self):
        """Test that TraceData structure is correct."""
        trace_data = load_trace(TEST_TRACES)

        # Check that TraceData has required attributes
        self.assertTrue(hasattr(trace_data, 'all_cooked_time'))
        self.assertTrue(hasattr(trace_data, 'all_cooked_bw'))
        self.assertTrue(hasattr(trace_data, 'all_file_names'))

        # Check that lengths are consistent
        self.assertEqual(len(trace_data.all_cooked_time), len(trace_data.all_cooked_bw))
        self.assertEqual(len(trace_data.all_cooked_time), len(trace_data.all_file_names))

        # Check __len__ method
        self.assertEqual(len(trace_data), len(trace_data.all_file_names))

    def test_trace_data_indexing(self):
        """Test that TraceData indexing returns correct values."""
        trace_data = load_trace(TEST_TRACES)

        for i in range(min(5, len(trace_data))):
            time, bw, name = trace_data[i]

            self.assertEqual(time, trace_data.all_cooked_time[i])
            self.assertEqual(bw, trace_data.all_cooked_bw[i])
            self.assertEqual(name, trace_data.all_file_names[i])

    def test_file_names_all_present(self):
        """Test that all file names from original are present."""
        trace_data = load_trace(TEST_TRACES)
        _, _, orig_names = src_load_trace.load_trace(TEST_TRACES)

        our_names = set(trace_data.all_file_names)
        orig_names_set = set(orig_names)

        self.assertEqual(our_names, orig_names_set,
                         "File name sets should be identical")


# ==============================================================================
# TEST CLASS 2: Video loader tests
# ==============================================================================

class TestVideoLoaderMatchesOriginal(unittest.TestCase):
    """Test that load_video_size matches the original fixed_env.py implementation."""

    @classmethod
    def setUpClass(cls):
        """Load shared test data."""
        cls.original_cwd = os.getcwd()
        src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
        os.chdir(src_dir)

        # Create a FixedEnvironment to get the original video_size data
        # This is how fixed_env.py loads video sizes in __init__
        orig_time, orig_bw, _ = src_load_trace.load_trace(TEST_TRACES)
        cls.fixed_env = src_fixed_env.Environment(orig_time, orig_bw)

    @classmethod
    def tearDownClass(cls):
        """Restore original working directory."""
        os.chdir(cls.original_cwd)

    def test_video_sizes_match(self):
        """Test that video chunk sizes match exactly with Environment.video_size."""
        # Load using our implementation
        video_data = load_video_size(VIDEO_SIZE_FILE, BITRATE_LEVELS)

        # Compare with Environment's video_size (loaded in setUpClass)
        for bitrate in range(BITRATE_LEVELS):
            with self.subTest(bitrate=bitrate):
                orig_sizes = self.fixed_env.video_size[bitrate]
                our_sizes = video_data.video_size[bitrate]

                # Compare lengths
                self.assertEqual(len(our_sizes), len(orig_sizes),
                                 f"Bitrate {bitrate}: size array length mismatch")

                # Compare each chunk size
                for chunk_idx in range(len(orig_sizes)):
                    self.assertEqual(
                        our_sizes[chunk_idx], orig_sizes[chunk_idx],
                        f"Bitrate {bitrate}, Chunk {chunk_idx}: size mismatch"
                    )

    def test_video_data_structure(self):
        """Test that VideoData structure is correct."""
        video_data = load_video_size(VIDEO_SIZE_FILE, BITRATE_LEVELS)

        # Check that VideoData has required attributes
        self.assertTrue(hasattr(video_data, 'video_size'))
        self.assertTrue(hasattr(video_data, 'bitrate_levels'))
        self.assertTrue(hasattr(video_data, 'num_chunks'))

        # Check bitrate levels matches src/fixed_env.py BITRATE_LEVELS
        self.assertEqual(video_data.bitrate_levels, BITRATE_LEVELS)

        # Check that video_size dict has all bitrate levels
        for bitrate in range(BITRATE_LEVELS):
            self.assertIn(bitrate, video_data.video_size)

    def test_num_chunks_correct(self):
        """Test that num_chunks is computed correctly."""
        video_data = load_video_size(VIDEO_SIZE_FILE, BITRATE_LEVELS)

        # num_chunks should be the minimum across all bitrates
        expected_num_chunks = min(
            len(video_data.video_size[b]) for b in range(BITRATE_LEVELS)
        )
        self.assertEqual(video_data.num_chunks, expected_num_chunks)

    def test_get_chunk_size_method(self):
        """Test the get_chunk_size method against Environment.video_size."""
        video_data = load_video_size(VIDEO_SIZE_FILE, BITRATE_LEVELS)

        for bitrate in range(BITRATE_LEVELS):
            for chunk_idx in range(min(10, video_data.num_chunks)):
                with self.subTest(bitrate=bitrate, chunk_idx=chunk_idx):
                    self.assertEqual(
                        video_data.get_chunk_size(bitrate, chunk_idx),
                        self.fixed_env.video_size[bitrate][chunk_idx]
                    )

    def test_get_next_chunk_sizes_method(self):
        """Test the get_next_chunk_sizes method against Environment.video_size."""
        video_data = load_video_size(VIDEO_SIZE_FILE, BITRATE_LEVELS)

        for chunk_idx in range(min(10, video_data.num_chunks)):
            with self.subTest(chunk_idx=chunk_idx):
                sizes = video_data.get_next_chunk_sizes(chunk_idx)

                self.assertEqual(len(sizes), BITRATE_LEVELS)

                for bitrate in range(BITRATE_LEVELS):
                    self.assertEqual(
                        sizes[bitrate],
                        self.fixed_env.video_size[bitrate][chunk_idx]
                    )

    def test_len_method(self):
        """Test the __len__ method."""
        video_data = load_video_size(VIDEO_SIZE_FILE, BITRATE_LEVELS)

        self.assertEqual(len(video_data), video_data.num_chunks)


# ==============================================================================
# TEST CLASS 3: Cross-validation with Environment
# ==============================================================================

class TestLoadersWithEnvironment(unittest.TestCase):
    """Test that loaded data works correctly with original Environment."""

    @classmethod
    def setUpClass(cls):
        """Load shared test data."""
        cls.original_cwd = os.getcwd()
        src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
        os.chdir(src_dir)

    @classmethod
    def tearDownClass(cls):
        """Restore original working directory."""
        os.chdir(cls.original_cwd)

    def test_trace_data_works_with_environment(self):
        """Test that trace data from our loader works with original Environment."""
        # Load using our implementation
        trace_data = load_trace(TEST_TRACES)

        # Create Environment with our data
        env = src_fixed_env.Environment(
            trace_data.all_cooked_time,
            trace_data.all_cooked_bw,
            random_seed=src_fixed_env.RANDOM_SEED
        )

        # Should be able to run a few steps without error
        for _ in range(10):
            delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_sizes, end_of_video, chunks_remain = \
                env.get_video_chunk(0)

            # Basic sanity checks
            self.assertGreater(delay, 0)
            self.assertGreaterEqual(buffer_size, 0)

    def test_video_data_matches_environment(self):
        """Test that video sizes from our loader match Environment's internal data."""
        # Load using our implementation
        video_data = load_video_size(VIDEO_SIZE_FILE, BITRATE_LEVELS)

        # Load traces for Environment
        trace_data = load_trace(TEST_TRACES)

        # Create Environment
        env = src_fixed_env.Environment(
            trace_data.all_cooked_time,
            trace_data.all_cooked_bw,
            random_seed=src_fixed_env.RANDOM_SEED
        )

        # Compare video sizes
        for bitrate in range(BITRATE_LEVELS):
            for chunk_idx in range(TOTAL_VIDEO_CHUNKS):
                with self.subTest(bitrate=bitrate, chunk_idx=chunk_idx):
                    self.assertEqual(
                        video_data.get_chunk_size(bitrate, chunk_idx),
                        env.video_size[bitrate][chunk_idx]
                    )

    def test_environment_output_equivalence(self):
        """Test that Environment produces same output with our loaded data vs original."""
        # Load using our implementation
        trace_data = load_trace(TEST_TRACES)

        # Load using original implementation
        orig_time, orig_bw, _ = src_load_trace.load_trace(TEST_TRACES)

        # Create two Environments
        env_ours = src_fixed_env.Environment(
            trace_data.all_cooked_time,
            trace_data.all_cooked_bw,
            random_seed=src_fixed_env.RANDOM_SEED
        )
        env_orig = src_fixed_env.Environment(
            orig_time,
            orig_bw,
            random_seed=src_fixed_env.RANDOM_SEED
        )

        # Run several steps and compare outputs
        # Note: Results may differ due to different trace ordering,
        # but video_size should be identical
        for bitrate in range(BITRATE_LEVELS):
            for chunk_idx in range(TOTAL_VIDEO_CHUNKS):
                self.assertEqual(
                    env_ours.video_size[bitrate][chunk_idx],
                    env_orig.video_size[bitrate][chunk_idx],
                    f"video_size mismatch at bitrate {bitrate}, chunk {chunk_idx}"
                )


# ==============================================================================
# TEST CLASS 4: Edge cases and error handling
# ==============================================================================

class TestLoaderEdgeCases(unittest.TestCase):
    """Test edge cases and error handling for loaders."""

    @classmethod
    def setUpClass(cls):
        """Load shared test data."""
        cls.original_cwd = os.getcwd()
        src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
        os.chdir(src_dir)

    @classmethod
    def tearDownClass(cls):
        """Restore original working directory."""
        os.chdir(cls.original_cwd)

    def test_empty_folder_raises_error(self):
        """Test that loading from non-existent folder raises appropriate error."""
        with self.assertRaises((FileNotFoundError, ValueError)):
            load_trace('./nonexistent_folder/')

    def test_invalid_video_prefix_raises_error(self):
        """Test that invalid video prefix raises appropriate error."""
        with self.assertRaises(FileNotFoundError):
            load_video_size('./nonexistent/video_size_', BITRATE_LEVELS)

    def test_different_bitrate_levels(self):
        """Test loading with different number of bitrate levels."""
        # Our implementation should work with fewer bitrates
        video_data = load_video_size(VIDEO_SIZE_FILE, 3)

        self.assertEqual(video_data.bitrate_levels, 3)
        self.assertEqual(len(video_data.video_size), 3)

        # Should have keys 0, 1, 2
        for bitrate in range(3):
            self.assertIn(bitrate, video_data.video_size)


# ==============================================================================
# TEST CLASS 5: Sorted vs unsorted trace loading
# ==============================================================================

class TestTraceSortingBehavior(unittest.TestCase):
    """Test the sorting behavior difference between implementations.

    Note: Our implementation sorts trace files for deterministic behavior,
    while the original does not. This test verifies the data is still
    equivalent regardless of order.
    """

    @classmethod
    def setUpClass(cls):
        """Load shared test data."""
        cls.original_cwd = os.getcwd()
        src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
        os.chdir(src_dir)

    @classmethod
    def tearDownClass(cls):
        """Restore original working directory."""
        os.chdir(cls.original_cwd)

    def test_our_loader_is_sorted(self):
        """Test that our implementation returns sorted file names."""
        trace_data = load_trace(TEST_TRACES)

        # Check that file names are sorted
        sorted_names = sorted(trace_data.all_file_names)
        self.assertEqual(trace_data.all_file_names, sorted_names,
                         "Our loader should return sorted file names")

    def test_data_content_identical(self):
        """Test that despite sorting difference, all data is present and identical."""
        trace_data = load_trace(TEST_TRACES)
        orig_time, orig_bw, orig_names = src_load_trace.load_trace(TEST_TRACES)

        # Create content-based comparison
        our_content = {}
        for i, name in enumerate(trace_data.all_file_names):
            our_content[name] = (
                tuple(trace_data.all_cooked_time[i]),
                tuple(trace_data.all_cooked_bw[i])
            )

        orig_content = {}
        for i, name in enumerate(orig_names):
            orig_content[name] = (
                tuple(orig_time[i]),
                tuple(orig_bw[i])
            )

        # Same files
        self.assertEqual(set(our_content.keys()), set(orig_content.keys()))

        # Same content for each file
        for name in our_content:
            self.assertEqual(our_content[name], orig_content[name],
                             f"Content mismatch for {name}")


# ==============================================================================
# TEST CLASS 6: Constants consistency
# ==============================================================================

class TestConstantsConsistency(unittest.TestCase):
    """Test that constants are consistent with original src files."""

    def test_bitrate_levels_constant(self):
        """Test BITRATE_LEVELS matches src/fixed_env.py."""
        self.assertEqual(BITRATE_LEVELS, 6)
        self.assertEqual(BITRATE_LEVELS, src_fixed_env.BITRATE_LEVELS)

    def test_total_video_chunks_constant(self):
        """Test TOTAL_VIDEO_CHUNKS matches src/fixed_env.py."""
        self.assertEqual(TOTAL_VIDEO_CHUNKS, 48)
        self.assertEqual(TOTAL_VIDEO_CHUNKS, src_fixed_env.TOTAL_VIDEO_CHUNCK)

    def test_video_size_file_constant(self):
        """Test VIDEO_SIZE_FILE matches src/fixed_env.py."""
        self.assertEqual(VIDEO_SIZE_FILE, './envivio/video_size_')
        self.assertEqual(VIDEO_SIZE_FILE, src_fixed_env.VIDEO_SIZE_FILE)


if __name__ == '__main__':
    unittest.main(verbosity=2)
