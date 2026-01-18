"""Unit tests to verify PPOAgent behavior matches src/ppo2.py Network.

This module tests equivalence between:
- pensieve_ppo.agent.rl.ppo.PPOAgent vs src/ppo2.py Network

Both implementations are loaded with the same pretrained model weights
and tested with identical inputs to ensure behavioral equivalence.
"""

import os
import unittest

import numpy as np
import torch

# Import original implementation from src
import ppo2 as src_ppo2

# Import our implementation
from pensieve_ppo.agent.rl.ppo import PPOAgent
from pensieve_ppo.agent.rl.ppo.model import Actor, Critic

# src directory path for pretrained model
SRC_DIR = os.path.join(os.path.dirname(__file__), '..', 'src')

# Constants matching src/test.py and src/train.py
# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L11-L13
S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8   # take how many frames in the past
A_DIM = 6   # number of bitrate levels

# State dimension as tuple
STATE_DIM = [S_INFO, S_LEN]
ACTION_DIM = A_DIM

# Pretrained model path
PRETRAINED_MODEL = os.path.join(SRC_DIR, 'pretrain', 'nn_model_ep_155400.pth')

# Learning rate for training tests
LEARNING_RATE = 1e-4

# Random seed for reproducibility
RANDOM_SEED = 42


class TestPPOModelArchitecture(unittest.TestCase):
    """Test that Actor and Critic model architectures match original."""

    def test_actor_parameter_count_matches(self):
        """Test Actor network has same number of parameters."""
        our_actor = Actor(STATE_DIM, ACTION_DIM)
        original_actor = src_ppo2.Actor(STATE_DIM, ACTION_DIM)

        our_params = sum(p.numel() for p in our_actor.parameters())
        original_params = sum(p.numel() for p in original_actor.parameters())

        self.assertEqual(our_params, original_params,
                         f"Actor parameter count mismatch: ours={our_params}, original={original_params}")

    def test_critic_parameter_count_matches(self):
        """Test Critic network has same number of parameters."""
        our_critic = Critic(STATE_DIM, ACTION_DIM)
        original_critic = src_ppo2.Critic(STATE_DIM, ACTION_DIM)

        our_params = sum(p.numel() for p in our_critic.parameters())
        original_params = sum(p.numel() for p in original_critic.parameters())

        self.assertEqual(our_params, original_params,
                         f"Critic parameter count mismatch: ours={our_params}, original={original_params}")

    def test_actor_layer_shapes_match(self):
        """Test Actor layer shapes match original."""
        our_actor = Actor(STATE_DIM, ACTION_DIM)
        original_actor = src_ppo2.Actor(STATE_DIM, ACTION_DIM)

        our_state_dict = our_actor.state_dict()
        original_state_dict = original_actor.state_dict()

        self.assertEqual(set(our_state_dict.keys()), set(original_state_dict.keys()),
                         "Actor layer names mismatch")

        for key in our_state_dict.keys():
            self.assertEqual(our_state_dict[key].shape, original_state_dict[key].shape,
                             f"Actor layer '{key}' shape mismatch")

    def test_critic_layer_shapes_match(self):
        """Test Critic layer shapes match original."""
        our_critic = Critic(STATE_DIM, ACTION_DIM)
        original_critic = src_ppo2.Critic(STATE_DIM, ACTION_DIM)

        our_state_dict = our_critic.state_dict()
        original_state_dict = original_critic.state_dict()

        self.assertEqual(set(our_state_dict.keys()), set(original_state_dict.keys()),
                         "Critic layer names mismatch")

        for key in our_state_dict.keys():
            self.assertEqual(our_state_dict[key].shape, original_state_dict[key].shape,
                             f"Critic layer '{key}' shape mismatch")


class TestPPOAgentWithPretrainedModel(unittest.TestCase):
    """Test PPOAgent matches original Network when using pretrained model."""

    @classmethod
    def setUpClass(cls):
        """Load pretrained model and create both agent instances."""
        if not os.path.exists(PRETRAINED_MODEL):
            raise unittest.SkipTest(f"Pretrained model not found: {PRETRAINED_MODEL}")

        # Create our PPOAgent
        cls.our_agent = PPOAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            learning_rate=LEARNING_RATE,
        )
        cls.our_agent.load(PRETRAINED_MODEL)

        # Create original Network
        cls.original_network = src_ppo2.Network(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            learning_rate=LEARNING_RATE,
        )
        cls.original_network.load_model(PRETRAINED_MODEL)

        # Set random seeds for reproducibility
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

    def _generate_random_state(self, batch_size: int = 1) -> np.ndarray:
        """Generate random state for testing.

        Returns state with shape (batch_size, S_INFO, S_LEN).
        Values are normalized similar to actual usage.
        """
        state = np.random.randn(batch_size, S_INFO, S_LEN).astype(np.float32)
        # Normalize to realistic ranges
        state = np.clip(state, -5, 5)
        return state

    def _generate_realistic_state(self) -> np.ndarray:
        """Generate realistic state mimicking actual environment states."""
        state = np.zeros((1, S_INFO, S_LEN), dtype=np.float32)
        # bit_rate: normalized (0-1)
        state[0, 0, -1] = 0.5
        # buffer_size: normalized
        state[0, 1, -1] = 0.3
        # bandwidth history: normalized throughput
        state[0, 2, :] = np.linspace(0.1, 0.5, S_LEN)
        # download time history
        state[0, 3, :] = np.linspace(0.1, 0.4, S_LEN)
        # next chunk sizes (only first A_DIM values matter)
        state[0, 4, :A_DIM] = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        # chunks remaining: normalized
        state[0, 5, -1] = 0.8
        return state

    def test_predict_single_state_matches(self):
        """Test predict() produces identical outputs for single state."""
        state = self._generate_realistic_state()

        our_result = self.our_agent.predict(state)
        original_result = self.original_network.predict(state)

        np.testing.assert_allclose(
            our_result, original_result, rtol=1e-6, atol=1e-6,
            err_msg="Predict outputs differ for single state"
        )

    def test_predict_random_states_match(self):
        """Test predict() produces identical outputs for random states."""
        np.random.seed(RANDOM_SEED)

        for i in range(10):
            with self.subTest(iteration=i):
                state = self._generate_random_state(batch_size=1)

                our_result = self.our_agent.predict(state)
                original_result = self.original_network.predict(state)

                np.testing.assert_allclose(
                    our_result, original_result, rtol=1e-6, atol=1e-6,
                    err_msg=f"Predict outputs differ at iteration {i}"
                )

    def test_predict_probabilities_approximately_sum_to_one(self):
        """Test that predicted probabilities approximately sum to 1.

        Note: Due to ACTION_EPS clamping (1e-4), probabilities may not sum exactly
        to 1.0. This is expected behavior from the original implementation.
        """
        state = self._generate_realistic_state()

        our_result = self.our_agent.predict(state)
        original_result = self.original_network.predict(state)

        # Allow for small deviation due to ACTION_EPS clamping
        # With 6 actions and ACTION_EPS=1e-4, max deviation is about 6 * 1e-4 = 6e-4
        self.assertAlmostEqual(np.sum(our_result), 1.0, places=3,
                               msg="Our agent probabilities don't approximately sum to 1")
        self.assertAlmostEqual(np.sum(original_result), 1.0, places=3,
                               msg="Original network probabilities don't approximately sum to 1")

        # More importantly, both implementations should sum to the same value
        self.assertAlmostEqual(np.sum(our_result), np.sum(original_result), places=6,
                               msg="Probability sums differ between implementations")

    def test_actor_forward_matches(self):
        """Test Actor forward pass produces identical outputs."""
        state = self._generate_realistic_state()
        state_tensor = torch.from_numpy(state).to(torch.float32)

        with torch.no_grad():
            our_output = self.our_agent.actor.forward(state_tensor)
            original_output = self.original_network.actor.forward(state_tensor)

        np.testing.assert_allclose(
            our_output.numpy(), original_output.numpy(), rtol=1e-6, atol=1e-6,
            err_msg="Actor forward outputs differ"
        )

    def test_critic_forward_matches(self):
        """Test Critic forward pass produces identical outputs."""
        state = self._generate_realistic_state()
        state_tensor = torch.from_numpy(state).to(torch.float32)

        with torch.no_grad():
            our_output = self.our_agent.critic.forward(state_tensor)
            original_output = self.original_network.critic.forward(state_tensor)

        np.testing.assert_allclose(
            our_output.numpy(), original_output.numpy(), rtol=1e-6, atol=1e-6,
            err_msg="Critic forward outputs differ"
        )

    def test_actor_batch_forward_matches(self):
        """Test Actor forward with batch input."""
        np.random.seed(RANDOM_SEED)
        batch_state = self._generate_random_state(batch_size=16)
        state_tensor = torch.from_numpy(batch_state).to(torch.float32)

        with torch.no_grad():
            our_output = self.our_agent.actor.forward(state_tensor)
            original_output = self.original_network.actor.forward(state_tensor)

        np.testing.assert_allclose(
            our_output.numpy(), original_output.numpy(), rtol=1e-6, atol=1e-6,
            err_msg="Actor batch forward outputs differ"
        )

    def test_critic_batch_forward_matches(self):
        """Test Critic forward with batch input."""
        np.random.seed(RANDOM_SEED)
        batch_state = self._generate_random_state(batch_size=16)
        state_tensor = torch.from_numpy(batch_state).to(torch.float32)

        with torch.no_grad():
            our_output = self.our_agent.critic.forward(state_tensor)
            original_output = self.original_network.critic.forward(state_tensor)

        np.testing.assert_allclose(
            our_output.numpy(), original_output.numpy(), rtol=1e-6, atol=1e-6,
            err_msg="Critic batch forward outputs differ"
        )


class TestPPOAgentComputeV(unittest.TestCase):
    """Test compute_v method matches original implementation."""

    @classmethod
    def setUpClass(cls):
        """Load pretrained model and create both agent instances."""
        if not os.path.exists(PRETRAINED_MODEL):
            raise unittest.SkipTest(f"Pretrained model not found: {PRETRAINED_MODEL}")

        # Create our PPOAgent
        cls.our_agent = PPOAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            learning_rate=LEARNING_RATE,
        )
        cls.our_agent.load(PRETRAINED_MODEL)

        # Create original Network
        cls.original_network = src_ppo2.Network(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            learning_rate=LEARNING_RATE,
        )
        cls.original_network.load_model(PRETRAINED_MODEL)

        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

    def _generate_trajectory(self, length: int = 10):
        """Generate a trajectory of states, actions, and rewards."""
        np.random.seed(RANDOM_SEED)

        s_batch = []
        a_batch = []
        r_batch = []

        for _ in range(length):
            state = np.random.randn(S_INFO, S_LEN).astype(np.float32) * 0.5
            action = np.zeros(ACTION_DIM, dtype=np.float32)
            action[np.random.randint(ACTION_DIM)] = 1
            reward = np.random.randn() * 0.5

            s_batch.append(state)
            a_batch.append(action)
            r_batch.append(reward)

        return s_batch, a_batch, r_batch

    def test_compute_v_terminal_matches(self):
        """Test compute_v with terminal=True produces identical results."""
        s_batch, a_batch, r_batch = self._generate_trajectory(length=10)

        our_result = self.our_agent.compute_v(s_batch, a_batch, r_batch, terminal=True)
        original_result = self.original_network.compute_v(
            torch.from_numpy(np.array(s_batch)).to(torch.float32),
            a_batch, r_batch, terminal=True
        )

        np.testing.assert_allclose(
            our_result, original_result, rtol=1e-6, atol=1e-6,
            err_msg="compute_v with terminal=True differs"
        )

    def test_compute_v_non_terminal_matches(self):
        """Test compute_v with terminal=False produces identical results."""
        s_batch, a_batch, r_batch = self._generate_trajectory(length=10)

        our_result = self.our_agent.compute_v(s_batch, a_batch, r_batch, terminal=False)
        original_result = self.original_network.compute_v(
            torch.from_numpy(np.array(s_batch)).to(torch.float32),
            a_batch, r_batch, terminal=False
        )

        np.testing.assert_allclose(
            our_result, original_result, rtol=1e-6, atol=1e-6,
            err_msg="compute_v with terminal=False differs"
        )

    def test_compute_v_short_trajectory_matches(self):
        """Test compute_v with short trajectory."""
        s_batch, a_batch, r_batch = self._generate_trajectory(length=3)

        our_result = self.our_agent.compute_v(s_batch, a_batch, r_batch, terminal=True)
        original_result = self.original_network.compute_v(
            torch.from_numpy(np.array(s_batch)).to(torch.float32),
            a_batch, r_batch, terminal=True
        )

        np.testing.assert_allclose(
            our_result, original_result, rtol=1e-6, atol=1e-6,
            err_msg="compute_v with short trajectory differs"
        )

    def test_compute_v_long_trajectory_matches(self):
        """Test compute_v with long trajectory."""
        s_batch, a_batch, r_batch = self._generate_trajectory(length=50)

        our_result = self.our_agent.compute_v(s_batch, a_batch, r_batch, terminal=True)
        original_result = self.original_network.compute_v(
            torch.from_numpy(np.array(s_batch)).to(torch.float32),
            a_batch, r_batch, terminal=True
        )

        np.testing.assert_allclose(
            our_result, original_result, rtol=1e-6, atol=1e-6,
            err_msg="compute_v with long trajectory differs"
        )


class TestPPOAgentTraining(unittest.TestCase):
    """Test training behavior matches original implementation."""

    def setUp(self):
        """Create fresh agent instances for each test."""
        if not os.path.exists(PRETRAINED_MODEL):
            self.skipTest(f"Pretrained model not found: {PRETRAINED_MODEL}")

        # Set seeds for reproducibility
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

        # Create our PPOAgent
        self.our_agent = PPOAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            learning_rate=LEARNING_RATE,
        )
        self.our_agent.load(PRETRAINED_MODEL)

        # Reset seeds before creating original network
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

        # Create original Network
        self.original_network = src_ppo2.Network(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            learning_rate=LEARNING_RATE,
        )
        self.original_network.load_model(PRETRAINED_MODEL)

    def _generate_training_batch(self, batch_size: int = 16):
        """Generate training batch data."""
        np.random.seed(RANDOM_SEED)

        s_batch = np.random.randn(batch_size, S_INFO, S_LEN).astype(np.float32) * 0.5
        a_batch = np.zeros((batch_size, ACTION_DIM), dtype=np.float32)
        for i in range(batch_size):
            a_batch[i, np.random.randint(ACTION_DIM)] = 1
        p_batch = np.random.dirichlet(np.ones(ACTION_DIM), size=batch_size).astype(np.float32)
        v_batch = np.random.randn(batch_size, 1).astype(np.float32) * 0.5

        return s_batch, a_batch, p_batch, v_batch

    def test_importance_ratio_matches(self):
        """Test importance sampling ratio calculation matches."""
        np.random.seed(RANDOM_SEED)

        pi_new = torch.tensor([[0.1, 0.2, 0.3, 0.2, 0.1, 0.1]], dtype=torch.float32)
        pi_old = torch.tensor([[0.15, 0.25, 0.25, 0.15, 0.1, 0.1]], dtype=torch.float32)
        acts = torch.tensor([[0, 0, 1, 0, 0, 0]], dtype=torch.float32)

        our_ratio = self.our_agent._r(pi_new, pi_old, acts)
        original_ratio = self.original_network.r(pi_new, pi_old, acts)

        np.testing.assert_allclose(
            our_ratio.numpy(), original_ratio.numpy(), rtol=1e-6, atol=1e-6,
            err_msg="Importance sampling ratio differs"
        )

    def test_weights_after_training_match(self):
        """Test that weights after training step match."""
        s_batch, a_batch, p_batch, v_batch = self._generate_training_batch(batch_size=16)

        # Synchronize weights before training
        self.original_network.actor.load_state_dict(self.our_agent.actor.state_dict())
        self.original_network.critic.load_state_dict(self.our_agent.critic.state_dict())

        # Also synchronize optimizer states
        self.original_network.optimizer = torch.optim.Adam(
            list(self.original_network.actor.parameters()) +
            list(self.original_network.critic.parameters()),
            lr=LEARNING_RATE
        )
        self.our_agent.optimizer = torch.optim.Adam(
            list(self.our_agent.actor.parameters()) +
            list(self.our_agent.critic.parameters()),
            lr=LEARNING_RATE
        )

        # Ensure same entropy weight
        self.original_network._entropy_weight = self.our_agent._entropy_weight

        # Set seeds before training
        torch.manual_seed(RANDOM_SEED)
        self.our_agent.train(s_batch, a_batch, p_batch, v_batch, epoch=1)

        torch.manual_seed(RANDOM_SEED)
        self.original_network.train(s_batch, a_batch, p_batch, v_batch, epoch=1)

        # Compare actor weights
        for key in self.our_agent.actor.state_dict().keys():
            np.testing.assert_allclose(
                self.our_agent.actor.state_dict()[key].numpy(),
                self.original_network.actor.state_dict()[key].numpy(),
                rtol=1e-5, atol=1e-5,
                err_msg=f"Actor weight '{key}' differs after training"
            )

        # Compare critic weights
        for key in self.our_agent.critic.state_dict().keys():
            np.testing.assert_allclose(
                self.our_agent.critic.state_dict()[key].numpy(),
                self.original_network.critic.state_dict()[key].numpy(),
                rtol=1e-5, atol=1e-5,
                err_msg=f"Critic weight '{key}' differs after training"
            )


class TestPPOAgentModelSaveLoad(unittest.TestCase):
    """Test model save/load functionality matches original."""

    @classmethod
    def setUpClass(cls):
        """Set up test directory."""
        cls.test_dir = os.path.join(os.path.dirname(__file__), 'test_models')
        os.makedirs(cls.test_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """Clean up test directory."""
        import shutil
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_load_pretrained_model(self):
        """Test that loading pretrained model produces same weights."""
        if not os.path.exists(PRETRAINED_MODEL):
            self.skipTest(f"Pretrained model not found: {PRETRAINED_MODEL}")

        our_agent = PPOAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            learning_rate=LEARNING_RATE,
        )
        our_agent.load(PRETRAINED_MODEL)

        original_network = src_ppo2.Network(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            learning_rate=LEARNING_RATE,
        )
        original_network.load_model(PRETRAINED_MODEL)

        # Compare actor weights
        for key in our_agent.actor.state_dict().keys():
            np.testing.assert_allclose(
                our_agent.actor.state_dict()[key].numpy(),
                original_network.actor.state_dict()[key].numpy(),
                rtol=1e-7, atol=1e-7,
                err_msg=f"Loaded actor weight '{key}' differs"
            )

        # Compare critic weights
        for key in our_agent.critic.state_dict().keys():
            np.testing.assert_allclose(
                our_agent.critic.state_dict()[key].numpy(),
                original_network.critic.state_dict()[key].numpy(),
                rtol=1e-7, atol=1e-7,
                err_msg=f"Loaded critic weight '{key}' differs"
            )

    def test_save_and_reload_matches(self):
        """Test that our saved model can be loaded by original and vice versa."""
        if not os.path.exists(PRETRAINED_MODEL):
            self.skipTest(f"Pretrained model not found: {PRETRAINED_MODEL}")

        our_save_path = os.path.join(self.test_dir, 'our_model.pth')
        original_save_path = os.path.join(self.test_dir, 'original_model.pth')

        # Create and save using our agent
        our_agent = PPOAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            learning_rate=LEARNING_RATE,
        )
        our_agent.load(PRETRAINED_MODEL)
        our_agent.save(our_save_path)

        # Create and save using original network
        original_network = src_ppo2.Network(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            learning_rate=LEARNING_RATE,
        )
        original_network.load_model(PRETRAINED_MODEL)
        original_network.save_model(original_save_path)

        # Load our saved model with original network
        original_loaded_ours = src_ppo2.Network(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            learning_rate=LEARNING_RATE,
        )
        original_loaded_ours.load_model(our_save_path)

        # Load original saved model with our agent
        our_loaded_original = PPOAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            learning_rate=LEARNING_RATE,
        )
        our_loaded_original.load(original_save_path)

        # Test state for comparison
        np.random.seed(RANDOM_SEED)
        test_state = np.random.randn(1, S_INFO, S_LEN).astype(np.float32) * 0.5

        # Verify cross-loaded models produce same output
        original_output = original_loaded_ours.predict(test_state)
        our_output = our_loaded_original.predict(test_state)

        np.testing.assert_allclose(
            original_output, our_output, rtol=1e-6, atol=1e-6,
            err_msg="Cross-loaded models produce different outputs"
        )


class TestPPOAgentGetSetParams(unittest.TestCase):
    """Test get/set network parameters functionality."""

    def test_get_network_params_format_matches(self):
        """Test get_network_params returns same format."""
        our_agent = PPOAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            learning_rate=LEARNING_RATE,
        )

        original_network = src_ppo2.Network(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            learning_rate=LEARNING_RATE,
        )

        our_params = our_agent.get_params()
        original_params = original_network.get_network_params()

        # Check structure
        self.assertEqual(len(our_params), len(original_params),
                         "get_network_params returns different number of elements")
        self.assertEqual(len(our_params), 2,
                         "get_network_params should return [actor_state_dict, critic_state_dict]")

    def test_set_network_params_from_original(self):
        """Test setting our agent params from original network."""
        if not os.path.exists(PRETRAINED_MODEL):
            self.skipTest(f"Pretrained model not found: {PRETRAINED_MODEL}")

        original_network = src_ppo2.Network(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            learning_rate=LEARNING_RATE,
        )
        original_network.load_model(PRETRAINED_MODEL)

        our_agent = PPOAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            learning_rate=LEARNING_RATE,
        )

        # Get params from original and set to ours
        params = original_network.get_network_params()
        our_agent.set_params(params)

        # Verify by predicting
        np.random.seed(RANDOM_SEED)
        test_state = np.random.randn(1, S_INFO, S_LEN).astype(np.float32) * 0.5

        our_output = our_agent.predict(test_state)
        original_output = original_network.predict(test_state)

        np.testing.assert_allclose(
            our_output, original_output, rtol=1e-6, atol=1e-6,
            err_msg="Predictions differ after set_network_params"
        )


class TestPPOAgentDeterminism(unittest.TestCase):
    """Test deterministic behavior with same inputs."""

    @classmethod
    def setUpClass(cls):
        """Load pretrained model."""
        if not os.path.exists(PRETRAINED_MODEL):
            raise unittest.SkipTest(f"Pretrained model not found: {PRETRAINED_MODEL}")

        cls.our_agent = PPOAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            learning_rate=LEARNING_RATE,
        )
        cls.our_agent.load(PRETRAINED_MODEL)

        cls.original_network = src_ppo2.Network(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            learning_rate=LEARNING_RATE,
        )
        cls.original_network.load_model(PRETRAINED_MODEL)

    def test_multiple_predict_calls_same_result(self):
        """Test that multiple predict calls with same input produce same result."""
        np.random.seed(RANDOM_SEED)
        state = np.random.randn(1, S_INFO, S_LEN).astype(np.float32) * 0.5

        results = []
        for _ in range(5):
            result = self.our_agent.predict(state.copy())
            results.append(result)

        for i in range(1, len(results)):
            np.testing.assert_array_equal(
                results[0], results[i],
                err_msg=f"Predict result {i} differs from result 0"
            )

    def test_batch_vs_individual_forward_matches(self):
        """Test batch forward produces same results as individual forwards."""
        np.random.seed(RANDOM_SEED)
        batch_size = 8
        states = np.random.randn(batch_size, S_INFO, S_LEN).astype(np.float32) * 0.5

        # Batch forward
        state_tensor = torch.from_numpy(states).to(torch.float32)
        with torch.no_grad():
            batch_output = self.our_agent.actor.forward(state_tensor).numpy()

        # Individual forwards
        individual_outputs = []
        for i in range(batch_size):
            single_state = states[i:i+1]
            single_tensor = torch.from_numpy(single_state).to(torch.float32)
            with torch.no_grad():
                output = self.our_agent.actor.forward(single_tensor).numpy()
            individual_outputs.append(output[0])

        individual_outputs = np.array(individual_outputs)

        np.testing.assert_allclose(
            batch_output, individual_outputs, rtol=1e-6, atol=1e-6,
            err_msg="Batch forward differs from individual forwards"
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
