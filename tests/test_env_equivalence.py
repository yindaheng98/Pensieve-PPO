"""Unit tests to verify ABREnv behavior matches original src/env.py ABREnv.

This module tests equivalence between:
- pensieve_ppo.gym.env.ABREnv (Gymnasium-based)
- src/env.py ABREnv (original implementation)

Both environments should produce identical:
- States (observations)
- Rewards
- Episode termination signals
- Internal state tracking

Testing strategy:
- Create each environment separately with the same seed
- Run them independently and compare results
- This accounts for potential differences in random number consumption during init

Note on constants:
- Constants are imported from pensieve_ppo.gym.defaults and pensieve_ppo.gym.env
  to verify equivalence with the original src_env.
- TestConstantsMatch class explicitly verifies these values match between
  our implementation and src_env to ensure consistency with the original
  Pensieve-PPO implementation.
"""

import os
import sys
import unittest

import numpy as np

# Add src directory to path for importing original implementation
SRC_DIR = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, SRC_DIR)

# Import original implementation from src
import env as src_env

# Import our gymnasium implementation and constants.
from pensieve_ppo.defaults import VIDEO_BIT_RATE, TOTAL_VIDEO_CHUNKS, create_env_with_default
from pensieve_ppo.gym.env import S_INFO, S_LEN

# Test-specific constants (must match src_env for equivalence verification)
A_DIM = len(VIDEO_BIT_RATE)  # Number of bitrate levels
RANDOM_SEED = 42  # Default random seed for reproducible tests


class TestABREnvEquivalenceBase(unittest.TestCase):
    """Base class for ABREnv equivalence tests."""

    @classmethod
    def setUpClass(cls):
        """Change to src directory for file path resolution."""
        cls.original_cwd = os.getcwd()
        os.chdir(SRC_DIR)

    @classmethod
    def tearDownClass(cls):
        """Restore original working directory."""
        os.chdir(cls.original_cwd)

    def _run_src_trajectory(self, seed: int, actions: list):
        """Run src env and return trajectory (states, rewards, dones)."""
        np.random.seed(seed)
        env = src_env.ABREnv(random_seed=seed)
        initial_state = env.reset()

        states = [initial_state]
        rewards = []
        dones = []

        for action in actions:
            state, reward, done, info = env.step(action)
            states.append(state)
            rewards.append(reward)
            dones.append(done)

        return states, rewards, dones, env

    def _run_gym_trajectory(self, seed: int, actions: list):
        """Run gym env and return trajectory (states, rewards, dones)."""
        # Set global random seed to match src/env.py behavior
        np.random.seed(seed)
        env = create_env_with_default(train=True)
        initial_state, _ = env.reset()

        states = [initial_state]
        rewards = []
        dones = []

        for action in actions:
            state, reward, terminated, truncated, info = env.step(action)
            states.append(state)
            rewards.append(reward)
            dones.append(terminated)

        return states, rewards, dones, env

    def _compare_trajectories(self, src_traj, gym_traj, msg_prefix: str = ""):
        """Compare two trajectories."""
        src_states, src_rewards, src_dones, _ = src_traj
        gym_states, gym_rewards, gym_dones, _ = gym_traj

        # Compare states
        self.assertEqual(len(src_states), len(gym_states), f"{msg_prefix}State count mismatch")
        for i, (src_s, gym_s) in enumerate(zip(src_states, gym_states)):
            np.testing.assert_array_almost_equal(
                src_s, gym_s, decimal=6,
                err_msg=f"{msg_prefix}State {i} mismatch"
            )

        # Compare rewards
        self.assertEqual(len(src_rewards), len(gym_rewards), f"{msg_prefix}Reward count mismatch")
        for i, (src_r, gym_r) in enumerate(zip(src_rewards, gym_rewards)):
            self.assertAlmostEqual(src_r, gym_r, places=6,
                                   msg=f"{msg_prefix}Reward {i} mismatch")

        # Compare dones
        self.assertEqual(src_dones, gym_dones, f"{msg_prefix}Done signals mismatch")


class TestConstantsMatch(TestABREnvEquivalenceBase):
    """Test that constants match between implementations."""

    def test_video_bitrate_values(self):
        """Test VIDEO_BIT_RATE values match."""
        expected = np.array([300., 750., 1200., 1850., 2850., 4300.])
        np.testing.assert_array_equal(src_env.VIDEO_BIT_RATE, expected)
        np.testing.assert_array_equal(VIDEO_BIT_RATE, expected)

    def test_normalization_constants(self):
        """Test normalization constants match."""
        from pensieve_ppo.gym.env import BUFFER_NORM_FACTOR, M_IN_K
        self.assertEqual(src_env.BUFFER_NORM_FACTOR, BUFFER_NORM_FACTOR)
        self.assertEqual(src_env.M_IN_K, M_IN_K)
        # CHUNK_TIL_VIDEO_END_CAP is now derived from simulator.total_chunks

    def test_reward_constants(self):
        """Test reward penalty constants match."""
        from pensieve_ppo.gym.env import REBUF_PENALTY, SMOOTH_PENALTY
        self.assertEqual(src_env.REBUF_PENALTY, REBUF_PENALTY)
        self.assertEqual(src_env.SMOOTH_PENALTY, SMOOTH_PENALTY)

    def test_dimension_constants(self):
        """Test dimension constants match."""
        from pensieve_ppo.gym.env import S_INFO as gym_S_INFO, S_LEN as gym_S_LEN
        self.assertEqual(src_env.S_INFO, gym_S_INFO)
        self.assertEqual(src_env.S_LEN, gym_S_LEN)
        # A_DIM is now defined by caller via video_bit_rate length
        self.assertEqual(src_env.A_DIM, A_DIM)


class TestResetEquivalence(TestABREnvEquivalenceBase):
    """Test that reset() produces identical results."""

    def test_reset_state_shape(self):
        """Test reset returns same state shape."""
        src_traj = self._run_src_trajectory(RANDOM_SEED, [])
        gym_traj = self._run_gym_trajectory(RANDOM_SEED, [])

        self.assertEqual(src_traj[0][0].shape, (S_INFO, S_LEN))
        self.assertEqual(gym_traj[0][0].shape, (S_INFO, S_LEN))

    def test_reset_state_values(self):
        """Test reset returns identical state values."""
        src_traj = self._run_src_trajectory(RANDOM_SEED, [])
        gym_traj = self._run_gym_trajectory(RANDOM_SEED, [])

        np.testing.assert_array_almost_equal(
            src_traj[0][0], gym_traj[0][0], decimal=6,
            err_msg="Reset state mismatch"
        )

    def test_reset_with_different_seeds(self):
        """Test reset produces different states with different seeds."""
        src_traj1 = self._run_src_trajectory(111, [])
        src_traj2 = self._run_src_trajectory(222, [])

        self.assertFalse(np.allclose(src_traj1[0][0], src_traj2[0][0]),
                         "Different seeds should produce different states")


class TestStepEquivalence(TestABREnvEquivalenceBase):
    """Test that step() produces identical results."""

    def test_single_step_all_actions(self):
        """Test single step with all possible actions."""
        for action in range(A_DIM):
            with self.subTest(action=action):
                src_traj = self._run_src_trajectory(RANDOM_SEED, [action])
                gym_traj = self._run_gym_trajectory(RANDOM_SEED, [action])
                self._compare_trajectories(src_traj, gym_traj, f"Action {action}: ")

    def test_multiple_sequential_steps(self):
        """Test multiple sequential steps with varying actions."""
        actions = [0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0, 3, 3, 3]

        src_traj = self._run_src_trajectory(RANDOM_SEED, actions)
        gym_traj = self._run_gym_trajectory(RANDOM_SEED, actions)
        self._compare_trajectories(src_traj, gym_traj)

    def test_rapid_quality_switching(self):
        """Test rapid quality switching."""
        actions = [0, 5, 0, 5, 2, 4, 1, 3, 5, 0]

        src_traj = self._run_src_trajectory(RANDOM_SEED, actions)
        gym_traj = self._run_gym_trajectory(RANDOM_SEED, actions)
        self._compare_trajectories(src_traj, gym_traj)


class TestEpisodeEquivalence(TestABREnvEquivalenceBase):
    """Test complete episode behavior matches."""

    def test_complete_episode(self):
        """Test complete video playback (47 steps after reset)."""
        actions = [i % A_DIM for i in range(TOTAL_VIDEO_CHUNKS - 1)]

        src_traj = self._run_src_trajectory(RANDOM_SEED, actions)
        gym_traj = self._run_gym_trajectory(RANDOM_SEED, actions)

        self._compare_trajectories(src_traj, gym_traj)

        # Last step should end video
        self.assertTrue(src_traj[2][-1], "src should end after 47 steps")
        self.assertTrue(gym_traj[2][-1], "gym should end after 47 steps")

    def test_multiple_episodes_via_reset(self):
        """Test multiple episodes by checking state consistency."""
        # First episode
        actions1 = [3] * (TOTAL_VIDEO_CHUNKS - 1)
        src_traj1 = self._run_src_trajectory(RANDOM_SEED, actions1)
        gym_traj1 = self._run_gym_trajectory(RANDOM_SEED, actions1)
        self._compare_trajectories(src_traj1, gym_traj1, "Episode 1: ")

        # After episode ends, reset and play again with different seed
        actions2 = [2] * (TOTAL_VIDEO_CHUNKS - 1)
        src_traj2 = self._run_src_trajectory(RANDOM_SEED + 100, actions2)
        gym_traj2 = self._run_gym_trajectory(RANDOM_SEED + 100, actions2)
        self._compare_trajectories(src_traj2, gym_traj2, "Episode 2: ")


class TestRewardCalculation(TestABREnvEquivalenceBase):
    """Test reward calculation equivalence."""

    def test_reward_with_constant_quality(self):
        """Test rewards with constant quality."""
        for quality in range(A_DIM):
            with self.subTest(quality=quality):
                actions = [quality] * 10
                src_traj = self._run_src_trajectory(RANDOM_SEED, actions)
                gym_traj = self._run_gym_trajectory(RANDOM_SEED, actions)

                for i, (src_r, gym_r) in enumerate(zip(src_traj[1], gym_traj[1])):
                    self.assertAlmostEqual(src_r, gym_r, places=6,
                                           msg=f"Quality {quality}, Step {i}: Reward mismatch")

    def test_reward_with_quality_changes(self):
        """Test rewards with quality switching."""
        actions = [0, 5, 0, 5, 2, 4, 1, 3]

        src_traj = self._run_src_trajectory(RANDOM_SEED, actions)
        gym_traj = self._run_gym_trajectory(RANDOM_SEED, actions)

        for i, (src_r, gym_r) in enumerate(zip(src_traj[1], gym_traj[1])):
            self.assertAlmostEqual(src_r, gym_r, places=6,
                                   msg=f"Step {i}: Reward mismatch")


class TestInternalStateTracking(TestABREnvEquivalenceBase):
    """Test internal state tracking consistency."""

    def test_last_bitrate_tracking(self):
        """Test last_bit_rate is tracked identically."""
        actions = [0, 3, 5, 2, 1, 4]

        _, _, _, src_env_obj = self._run_src_trajectory(RANDOM_SEED, actions)
        _, _, _, gym_env_obj = self._run_gym_trajectory(RANDOM_SEED, actions)

        self.assertEqual(src_env_obj.last_bit_rate, gym_env_obj.last_bit_rate)
        self.assertEqual(src_env_obj.last_bit_rate, actions[-1])

    def test_buffer_size_tracking(self):
        """Test buffer_size is tracked identically."""
        actions = [3] * 10

        _, _, _, src_env_obj = self._run_src_trajectory(RANDOM_SEED, actions)
        _, _, _, gym_env_obj = self._run_gym_trajectory(RANDOM_SEED, actions)

        self.assertAlmostEqual(src_env_obj.buffer_size, gym_env_obj.buffer_size, places=6)


class TestEdgeCases(TestABREnvEquivalenceBase):
    """Test edge cases and boundary conditions."""

    def test_constant_low_quality(self):
        """Test continuous low quality streaming."""
        actions = [0] * 20

        src_traj = self._run_src_trajectory(RANDOM_SEED, actions)
        gym_traj = self._run_gym_trajectory(RANDOM_SEED, actions)
        self._compare_trajectories(src_traj, gym_traj)

    def test_constant_high_quality(self):
        """Test continuous high quality streaming."""
        actions = [A_DIM - 1] * 20

        src_traj = self._run_src_trajectory(RANDOM_SEED, actions)
        gym_traj = self._run_gym_trajectory(RANDOM_SEED, actions)
        self._compare_trajectories(src_traj, gym_traj)

    def test_video_boundary_transition(self):
        """Test behavior at video end boundary."""
        actions = [3] * (TOTAL_VIDEO_CHUNKS - 1)

        src_traj = self._run_src_trajectory(RANDOM_SEED, actions)
        gym_traj = self._run_gym_trajectory(RANDOM_SEED, actions)

        # Last step should signal end
        self.assertTrue(src_traj[2][-1], "src should end video")
        self.assertTrue(gym_traj[2][-1], "gym should end video")

        # States and rewards should match
        self._compare_trajectories(src_traj, gym_traj)


class TestInterfaceCompatibility(TestABREnvEquivalenceBase):
    """Test interface compatibility."""

    def _create_gym_env(self, seed: int):
        """Helper to create gym env with given seed."""
        np.random.seed(seed)
        return create_env_with_default(train=True)

    def test_reset_return_format(self):
        """Test reset return format differences."""
        np.random.seed(RANDOM_SEED)
        src_abr = src_env.ABREnv(random_seed=RANDOM_SEED)
        gym_abr = self._create_gym_env(RANDOM_SEED)

        # src returns just state
        src_result = src_abr.reset()
        self.assertIsInstance(src_result, np.ndarray)

        # gym returns (state, info) tuple
        gym_result = gym_abr.reset()
        self.assertIsInstance(gym_result, tuple)
        self.assertEqual(len(gym_result), 2)
        self.assertIsInstance(gym_result[0], np.ndarray)
        self.assertIsInstance(gym_result[1], dict)

    def test_step_return_format(self):
        """Test step return format differences."""
        np.random.seed(RANDOM_SEED)
        src_abr = src_env.ABREnv(random_seed=RANDOM_SEED)
        src_abr.reset()
        gym_abr = self._create_gym_env(RANDOM_SEED)
        gym_abr.reset()

        # src returns (state, reward, done, info)
        src_result = src_abr.step(2)
        self.assertEqual(len(src_result), 4)

        # gym returns (state, reward, terminated, truncated, info)
        gym_result = gym_abr.step(2)
        self.assertEqual(len(gym_result), 5)

    def test_info_dict_common_fields(self):
        """Test info dict contains common fields."""
        np.random.seed(RANDOM_SEED)
        src_abr = src_env.ABREnv(random_seed=RANDOM_SEED)
        src_abr.reset()
        gym_abr = self._create_gym_env(RANDOM_SEED)
        gym_abr.reset()

        _, _, _, src_info = src_abr.step(2)
        _, _, _, _, gym_info = gym_abr.step(2)

        # src uses 'bitrate', gym uses 'quality' (more generic name)
        self.assertIn('bitrate', src_info)
        self.assertIn('rebuffer', src_info)
        self.assertIn('quality', gym_info)
        self.assertIn('rebuffer', gym_info)

        # Values should be equal
        self.assertEqual(src_info['bitrate'], gym_info['quality'])


class TestDeterminism(TestABREnvEquivalenceBase):
    """Test determinism with same seed."""

    def test_identical_trajectory_with_same_seed(self):
        """Test that both envs produce identical trajectories with same seed."""
        actions = [i % A_DIM for i in range(20)]

        src_traj = self._run_src_trajectory(RANDOM_SEED, actions)
        gym_traj = self._run_gym_trajectory(RANDOM_SEED, actions)

        self._compare_trajectories(src_traj, gym_traj)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        actions = [3] * 5

        src_traj1 = self._run_src_trajectory(111, actions)
        src_traj2 = self._run_src_trajectory(222, actions)

        # States should differ
        self.assertFalse(np.allclose(src_traj1[0][-1], src_traj2[0][-1]),
                         "Different seeds should produce different states")

    def test_reproducibility(self):
        """Test that same seed produces same results across runs."""
        actions = [i % A_DIM for i in range(10)]

        # Run twice
        traj1 = self._run_src_trajectory(RANDOM_SEED, actions)
        traj2 = self._run_src_trajectory(RANDOM_SEED, actions)

        for i, (s1, s2) in enumerate(zip(traj1[0], traj2[0])):
            np.testing.assert_array_equal(s1, s2,
                                          err_msg=f"State {i} not reproducible")


class TestLongTrajectory(TestABREnvEquivalenceBase):
    """Test long trajectory equivalence."""

    def test_100_steps(self):
        """Test equivalence over 100 steps."""
        actions = [i % A_DIM for i in range(100)]

        src_traj = self._run_src_trajectory(RANDOM_SEED, actions)
        gym_traj = self._run_gym_trajectory(RANDOM_SEED, actions)

        self._compare_trajectories(src_traj, gym_traj)

    def test_random_actions(self):
        """Test equivalence with random action sequence."""
        np.random.seed(12345)
        actions = [np.random.randint(0, A_DIM) for _ in range(50)]

        src_traj = self._run_src_trajectory(RANDOM_SEED, actions)
        gym_traj = self._run_gym_trajectory(RANDOM_SEED, actions)

        self._compare_trajectories(src_traj, gym_traj)


if __name__ == '__main__':
    unittest.main(verbosity=2)
