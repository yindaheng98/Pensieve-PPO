# Pensieve PPO

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.28+-green.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A user-friendly PyTorch implementation of **Pensieve** [1], a neural adaptive video streaming system. This implementation uses **Proximal Policy Optimization (PPO)** instead of the original A3C algorithm, achieving improved training stability and comparable performance.

## Features

- **Modern PyTorch Implementation**: Clean, modular codebase using PyTorch 2.0+
- **Gymnasium Environment**: Standard RL environment interface for easy integration
- **Parallel Training**: Multi-worker distributed training support
- **TensorBoard Integration**: Real-time training visualization
- **Extensible Architecture**: Easy to add new agents and environments

## Installation

### From Source

```bash
git clone https://github.com/yindaheng98/Pensieve-PPO.git
cd Pensieve-PPO
pip install -e . # or
pip install --target . --upgrade . --no-deps
```

### Dependencies

```bash
pip install torch numpy gymnasium tensorboard
```

## Quick Start

### Using the Gymnasium Environment

```python
from pensieve_ppo.gym import ABREnv
from pensieve_ppo.defaults import create_env_with_default

# Create environment with default Pensieve parameters
env = create_env_with_default(train=True)

# Standard Gymnasium API
obs, info = env.reset()
while True:
    action = env.action_space.sample()  # or use your agent
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Training a PPO Agent

```bash
# Train with default settings
python -m pensieve_ppo.train

# Train with custom parameters
python -m pensieve_ppo.train \
    --parallel-workers 16 \
    --train-epochs 500000 \
    --model-save-interval 300 \
    --output-dir ./ppo
```

### Testing a Trained Model

```bash
# Test with a trained model
python -m pensieve_ppo.test --model-path ./ppo/model_ep300.pt

# Test with custom trace folder
python -m pensieve_ppo.test \
    --model-path ./ppo/model_ep300.pt \
    --test-trace-folder ./src/test/
```

## Package Structure

```
pensieve_ppo/
├── core/                  # Core simulation components
│   ├── trace/             # Network trace handling (bandwidth, latency)
│   ├── video/             # Video processing (chunk sizes, bitrates, playback)
│   └── simulator/         # Combines trace & video into ABR simulator
├── gym/                   # Gymnasium environment wrapper
│   └── env.py             # ABREnv - standard RL environment interface
├── agent/                 # RL agent implementations
│   ├── abc.py             # Abstract base classes
│   ├── registry.py        # Agent factory and registration
│   ├── trainer.py         # Distributed training framework
│   └── rl/
│       ├── observer.py    # State observation and reward calculation
│       └── ppo/           # PPO agent implementation
├── defaults.py            # Default parameters and factory functions
├── train.py               # Training script
└── test.py                # Testing script
```

## Architecture

### Agent Class Hierarchy

The agent system follows a hierarchical inheritance structure:

```
AbstractAgent
    ├── select_action(state) -> (action, action_prob)
    │
    └── AbstractTrainableAgent
        ├── select_action_for_training(state) -> (action, action_prob)
        ├── produce_training_batch(trajectory, done) -> TrainingBatch
        ├── train_batch(training_batches, epoch) -> metrics
        ├── get_params() -> params
        ├── set_params(params) -> None
        ├── save(path) -> None
        └── load(path) -> None
            │
            └── AbstractRLAgent
                ├── train(s_batch, a_batch, p_batch, v_batch, epoch) -> metrics
                ├── compute_v(s_batch, a_batch, r_batch, terminal) -> v_batch
                ├── produce_training_batch(trajectory, done) -> RLTrainingBatch  # Implemented
                └── train_batch(training_batches, epoch) -> metrics  # Implemented
```

**AbstractAgent** (`pensieve_ppo.agent.abc.AbstractAgent`):
- Base class for all agents
- Defines the minimal interface: `select_action(state)` method
- Used for inference-only agents that don't require training

**AbstractTrainableAgent** (`pensieve_ppo.agent.trainable.AbstractTrainableAgent`):
- Extends `AbstractAgent` with training infrastructure
- Adds methods for:
  - Training-time action selection (`select_action_for_training`)
  - Converting trajectories to training batches (`produce_training_batch`)
  - Training on batches (`train_batch`)
  - Model persistence (`save`, `load`, `get_params`, `set_params`)
- Abstract methods must be implemented by subclasses

**AbstractRLAgent** (`pensieve_ppo.agent.rl.abc.AbstractRLAgent`):
- Extends `AbstractTrainableAgent` with RL-specific functionality
- Implements `produce_training_batch` and `train_batch` using RL methods
- Requires subclasses to implement:
  - `train()`: Core training logic (e.g., PPO, A3C, DQN)
  - `compute_v()`: Value target computation (returns/advantages)
- Concrete implementations: `PPOAgent`, `A3CAgent`, `DQNAgent`, etc.

> **Note on RL Agent Implementations**: The agents in `pensieve_ppo/agent/rl/` (PPO, A3C, DQN) are **reinforcement learning algorithms**, not imitation learning algorithms. While the framework technically allows running them with `pensieve_ppo.imitate`, this is not recommended as these algorithms are designed to learn from rewards through environment interaction, not from teacher demonstrations.
>
> **Note on A3C Implementation**: The `A3CAgent` implementation is based on the A3C (Asynchronous Advantage Actor-Critic) algorithm, but the actual `Trainer` performs **synchronous updates** rather than asynchronous updates. This means all workers synchronize before each parameter update, which differs from the original A3C paper's asynchronous design.

### Agent, Observer, and Trainer Relationships

**AbstractABRStateObserver** (`pensieve_ppo.gym.env.AbstractABRStateObserver`):
- Abstract interface for state observation and reward calculation
- Decouples state representation from environment dynamics
- Methods:
  - `reset(env, initial_bit_rate) -> (state, info)`: Initialize state
  - `observe(env, bit_rate, result) -> (state, reward, info)`: Update state and compute reward
- Implementations:
  - `RLABRStateObserver`: For RL agents (returns `np.ndarray` states)
  - `MPCABRStateObserver`: For MPC-based agents (returns `MPCState` dataclass)
  - `NetLLMABRStateObserver`: For NetLLM agents (returns `NetLLMState`)

**ABREnv** (`pensieve_ppo.gym.env.ABREnv`):
- Gymnasium-compatible environment wrapper
- Uses an `AbstractABRStateObserver` instance to:
  - Observe states from simulator results
  - Compute rewards based on actions and results
- The observer is injected via constructor, allowing different state representations for different agent types

**Trainer** (`pensieve_ppo.agent.trainer.Trainer`):
- Coordinates distributed training with multiple parallel workers
- Architecture:
  - **Central Agent**: Aggregates experiences from workers, updates model, distributes parameters
  - **Worker Agents**: Collect experiences by interacting with environments
- Uses `AbstractTrainableAgent` interface:
  - Workers call `select_action_for_training()` for exploration
  - Workers call `produce_training_batch()` to convert trajectories
  - Central agent calls `train_batch()` to update the model
  - Parameters synchronized via `get_params()` and `set_params()`

**Relationship Flow**:
```
Trainer
  ├── Creates multiple (env, agent) pairs via factories
  ├── Workers: env.step(action) -> observer.observe() -> state, reward
  ├── Workers: agent.select_action_for_training(state) -> action
  ├── Workers: agent.produce_training_batch(trajectory) -> TrainingBatch
  └── Central: agent.train_batch(batches) -> updates model
```

### State, Step, and TrainingBatch

**State** (`pensieve_ppo.gym.env.State`):
- Type alias: `State = Any`
- Represents environment observations
- Concrete type depends on the observer:
  - `RLState` (`np.ndarray`) for RL agents: shape `(S_INFO, S_LEN)` = `(6, 8)`
  - `MPCState` (dataclass) for MPC agents
  - `NetLLMState` (dataclass) for NetLLM agents
- Used in:
  - `AbstractAgent.select_action(state)`
  - `AbstractTrainableAgent.select_action_for_training(state)`
  - `Step.state` field

**Step** (`pensieve_ppo.agent.trainable.Step`):
- Dataclass representing a single environment step
- Fields:
  - `state: State`: Observation at this step
  - `action: List[int]`: One-hot encoded action (e.g., `[0, 0, 1, 0, 0, 0]`)
  - `action_prob: List[float]`: Action probability distribution from agent
  - `reward: float`: Reward received
  - `step: int`: Step index within trajectory
  - `done: bool`: Whether episode terminated/truncated
- Usage:
  - Collected during environment rollout in `Trainer._agent_worker()`
  - Stored in `trajectory: List[Step]`
  - Converted to `TrainingBatch` via `produce_training_batch()`

**TrainingBatch** (`pensieve_ppo.agent.trainable.TrainingBatch`):
- Abstract base class for training data containers
- Subclasses define algorithm-specific fields:
  - **RLTrainingBatch** (`pensieve_ppo.agent.rl.abc.RLTrainingBatch`):
    - `s_batch: List[RLState]`: States
    - `a_batch: List[List[int]]`: One-hot actions
    - `p_batch: List[List[float]]`: Action probabilities
    - `v_batch: List[float]`: Computed value targets (returns)
  - **NetLLMTrainingBatch** (`pensieve_ppo.agent.netllm.abc.NetLLMTrainingBatch`):
    - `states: List[torch.Tensor]`: State tensors
    - `actions: List[int]`: Action indices
    - `returns: List[float]`: Return-to-go values
    - `timesteps: List[int]`: Timestep indices
    - `labels: List[int]`: Target labels
- Usage:
  - Created by `produce_training_batch(trajectory)` from `List[Step]`
  - Multiple batches aggregated in `train_batch(List[TrainingBatch])`
  - Converted to numpy arrays/tensors for actual training

**Data Flow**:
```
Environment Step
  → Step(state, action, action_prob, reward, step, done)
  → Collected in trajectory: List[Step]
  → produce_training_batch(trajectory)
  → TrainingBatch (e.g., RLTrainingBatch)
  → train_batch([TrainingBatch, ...])
  → Model update
```

### Imitation Learning

**ImitationObserver** (`pensieve_ppo.gym.imitate.ImitationObserver`):
- Combines two observers (student and teacher) in the same environment
- Both observers observe the same environment state and teacher's actions
- Returns `ImitationState` containing:
  - `student_state`: State for training the student agent (neural network)
  - `teacher_state`: State for teacher agent's decision-making

**How It Works**:
1. **Two Observers in Same Environment**: The `ImitationObserver` wraps both a `student_observer` and a `teacher_observer`, both observing the same `ABREnv` instance
2. **Teacher Makes Decisions**: The teacher agent uses `teacher_state` to select actions
3. **Teacher Actions Are Executed**: The teacher's selected actions are executed in the environment
4. **Student Learns from Teacher**: The student agent receives `student_state` and learns to imitate the teacher's decisions through behavioral cloning

**ImitationState** (`pensieve_ppo.gym.imitate.ImitationState`):
- Dataclass containing both `student_state` and `teacher_state`
- Both states are generated from the same environment step and teacher action
- `student_state` is used for training (e.g., RL policy updates)
- `teacher_state` is used by the teacher agent for action selection

**ImitationTrainer** (`pensieve_ppo.agent.imitate.ImitationTrainer`):
- Extends `Trainer` for distributed imitation learning
- Architecture:
  - **Central Agent (Student)**: Neural network that learns to imitate teacher decisions
  - **Worker Agents (Teacher)**: Expert agents (e.g., BBA, MPC, LLM-based) that generate trajectories
- Workflow:
  - Workers use teacher agent with `teacher_state` to select actions
  - Teacher actions are executed in the environment
  - Both observers update their states from the same environment result
  - Student receives `student_state` and teacher's action for training
  - No parameter synchronization between student and teacher (different agent types)

**Example Usage**:
```python
from pensieve_ppo.gym.imitate import ImitationObserver
from pensieve_ppo.agent.rl.observer import RLABRStateObserver
from pensieve_ppo.agent.bba.observer import BBAStateObserver

# Create observers
student_observer = RLABRStateObserver(levels_quality=VIDEO_BIT_RATE)
teacher_observer = BBAStateObserver(levels_quality=VIDEO_BIT_RATE)

# Combine for imitation learning
imitation_observer = ImitationObserver(
    student_observer=student_observer,
    teacher_observer=teacher_observer,
)

# Use in environment
env = ABREnv(simulator=simulator, observer=imitation_observer)
state, info = env.reset()
# state.student_state: RLState for training RL agent
# state.teacher_state: BBAState for BBA agent's decision
```

**Training with Imitation Learning**:
```bash
# Train student agent (PPO) to imitate teacher agent (BBA)
python -m pensieve_ppo.imitate \
    --agent-name ppo \
    --teacher-agent-name bba \
    --parallel-workers 16 \
    --train-epochs 500000
```

> **Warning**: The RL agents (`ppo`, `a3c`, `dqn` in `pensieve_ppo/agent/rl/`) are **reinforcement learning algorithms** designed to learn from reward signals, not from teacher demonstrations. Although the framework allows running them with `pensieve_ppo.imitate`, this is **not recommended** for production use. For proper imitation learning, consider using agents specifically designed for behavioral cloning or other imitation learning methods (e.g., `netllm`).

## API Reference

### Creating Environment and Agent

```python
from pensieve_ppo.defaults import (
    create_env_with_default,
    create_env_agent_with_default,
    create_env_agent_factory_with_default,
)

# Create just the environment
env = create_env_with_default(
    levels_quality=[300., 750., 1200., 1850., 2850., 4300.],  # Kbps
    trace_folder='./src/train/',
    train=True,
)

# Create compatible env and agent pair
env, agent = create_env_agent_with_default(
    agent_name='ppo',
    model_path='./ppo/model.pt',  # Optional: load pretrained weights
    device='cuda',
)

# Create factories for distributed training
env_factory, agent_factory = create_env_agent_factory_with_default(
    agent_name='ppo',
    train=True,
)
```

### Using the Agent

```python
from pensieve_ppo.agent import create_agent

# Create agent directly
agent = create_agent(
    name='ppo',
    state_dim=(6, 8),  # (S_INFO, S_LEN)
    action_dim=6,       # Number of bitrate levels
    device='cuda',
    learning_rate=1e-4,
    gamma=0.99,
)

# Predict action
state = env.reset()[0]
action, action_prob = agent.select_action(state)

# Train on batch
metrics = agent.train(s_batch, a_batch, p_batch, v_batch, epoch)
```

### Registering Custom Agents

The `register` function allows you to register custom agent implementations with the Pensieve-PPO framework. Once registered, your custom agent can be used with all factory functions (`create_agent`, `create_env`, etc.) and command-line tools.

**Function Signature**:
```python
from pensieve_ppo.agent import register
from pensieve_ppo.agent.abc import AbstractAgent
from pensieve_ppo.gym.env import AbstractABRStateObserver

register(
    name: str,
    agent_cls: Type[AbstractAgent],
    observer_cls: Type[AbstractABRStateObserver],
    trainable_agent_cls: Optional[Type[AbstractTrainableAgent]] = None,
) -> None
```

**Parameters**:
- `name`: Name to register the agent under (case-sensitive). This name will be used in `create_agent()`, `create_env()`, and command-line arguments.
- `agent_cls`: The agent class to register. Must be a subclass of `AbstractAgent`.
- `observer_cls`: The observer class associated with this agent. Must be a subclass of `AbstractABRStateObserver`. The observer handles state observation and reward calculation for the agent.
- `trainable_agent_cls`: Optional trainable agent class. If not provided, will be automatically set to `agent_cls` if it's a subclass of `AbstractTrainableAgent`.

**Example: Registering a Custom Agent**:
```python
from pensieve_ppo.agent import register
from pensieve_ppo.agent.abc import AbstractAgent
from pensieve_ppo.gym.env import AbstractABRStateObserver

# Define your custom agent
class MyCustomAgent(AbstractAgent):
    def select_action(self, state):
        # Your implementation
        pass

# Define your custom observer
class MyCustomObserver(AbstractABRStateObserver):
    def reset(self, env, initial_bit_rate):
        # Your implementation
        pass
    
    def observe(self, env, bit_rate, result):
        # Your implementation
        pass

# Register the agent
register("my-custom-agent", MyCustomAgent, MyCustomObserver)

# Now you can use it with factory functions
from pensieve_ppo.agent import create_agent, create_env

agent = create_agent(name="my-custom-agent", ...)
env = create_env(name="my-custom-agent", ...)
```

**Example: Registering a Trainable Agent**:
```python
from pensieve_ppo.agent import register
from pensieve_ppo.agent.trainable import AbstractTrainableAgent

class MyTrainableAgent(AbstractTrainableAgent):
    # Implement all required methods
    pass

# Register with trainable agent class
register("my-trainable", MyTrainableAgent, MyCustomObserver)

# Can be used for training
from pensieve_ppo.defaults import create_env_agent_with_default
env, agent = create_env_agent_with_default(agent_name="my-trainable")
```

**Checking Available Agents**:
```python
from pensieve_ppo.agent import get_available_agents, get_available_trainable_agents

# Get all registered agents
all_agents = get_available_agents()
print(all_agents)  # ['ppo', 'bba', 'mpc', 'dqn', 'a3c', ...]

# Get only trainable agents
trainable_agents = get_available_trainable_agents()
print(trainable_agents)  # ['ppo', 'dqn', 'a3c', ...]
```

**Note**: Agent registration typically happens in the agent module's `__init__.py` file. When you import the module, the registration is automatically executed. For example, importing `pensieve_ppo.agent.rl` automatically registers all RL agents (ppo, dqn, a3c).

### Environment Details

**Observation Space** (`Box(6, 8)`):
| Index | Description |
|-------|-------------|
| 0 | Last quality normalized by max quality |
| 1 | Buffer size normalized by buffer_norm_factor |
| 2 | Throughput (chunk_size / delay) in Mbps |
| 3 | Delay normalized by buffer_norm_factor |
| 4 | Next chunk sizes at each bitrate level (MB) |
| 5 | Remaining chunks normalized by total |

**Action Space** (`Discrete(6)`): Select bitrate level (0-5)

**Reward**: `quality - 4.3 * rebuffer - 1.0 * |quality_change|`

## Command Line Options

### Training (`pensieve_ppo.train`)

```
--train-trace-folder    Training trace folder (default: ./src/train/)
--output-dir            Output directory (default: ./ppo)
--parallel-workers      Number of parallel workers (default: 16)
--steps-per-epoch       Steps per epoch per worker (default: 1000)
--train-epochs          Total training epochs (default: 500000)
--model-save-interval   Model checkpoint interval (default: 300)
--pretrained-model-path Resume from pretrained model
```

### Testing (`pensieve_ppo.test`)

```
--test-trace-folder     Test trace folder (default: ./src/test/)
--model-path            Path to trained model weights
--test-log-file-prefix  Prefix for test log files
```

### Common Options

```
--agent-name            RL algorithm (default: ppo)
--device                PyTorch device (cuda/cpu)
--levels-quality        Bitrate levels in Kbps
--state-history-len     State history length (default: 8)
--random-seed           Random seed (default: 42)
-o, --agent-options     Extra agent kwargs (e.g., learning_rate=1e-4)
-e, --env-options       Extra env kwargs (e.g., rebuf_penalty=4.3)
```

## TensorBoard Monitoring

Monitor training in real-time:

```bash
tensorboard --logdir=./ppo
```

---

# Original README

### Updates

**Jan. 18, 2025:** We removed the rate-based method and added NetLLM [4].

**May. 4, 2024:** We removed the Elastic, revised  BOLA, and add new baseline Comyco [3] and Genet [2].

**Jan. 26, 2024:** We are excited to announce significant updates to Pensieve-PPO! We have replaced TensorFlow with PyTorch, and we have achieved a similar training speed while training models that rival in performance.

*For the TensorFlow version, please check [Pensieve-PPO TF Branch](https://github.com/godka/Pensieve-PPO/tree/master).*

**Dec. 28, 2021:** In a previous update, we enhanced Pensieve-PPO with several state-of-the-art technologies, including Dual-Clip PPO and adaptive entropy decay.

## About Pensieve-PPO

Pensieve-PPO is a user-friendly PyTorch implementation of Pensieve [1], a neural adaptive video streaming system. Unlike A3C, we utilize the Proximal Policy Optimization (PPO) algorithm for training.

This stable version of Pensieve-PPO includes both the training and test datasets.

You can run the repository by executing the following command:

```
python train.py
```

The results will be evaluated on the test set (from HSDPA) every 300 epochs.

## Tensorboard Integration

To monitor the training process in real time, you can leverage Tensorboard. Simply run the following command:

```
tensorboard --logdir=./
```

## Pretrained Model

We have also added a pretrained model, which can be found at [this link](https://github.com/godka/Pensieve-PPO/tree/torch/src/pretrain). This model demonstrates a substantial improvement of 7.03% (from 0.924 to 0.989) in average Quality of Experience (QoE) compared to the original Pensieve model [1]. For a more detailed performance analysis, refer to the figures below:

<p align="center">
    <img src="src/baselines-br.png" width="50%"><img src="src/baselines-bs.png" width="50%">
</p>
<p align="center">
    <img src="src/baselines-qoe.png" width="100%">
</p>
If you have any questions or require further assistance, please don't hesitate to reach out.

## Additional Reinforcement Learning Algorithms

For more implementations of reinforcement learning algorithms, please visit the following branches:

- DQN: [Pensieve-PPO DQN Branch](https://github.com/godka/Pensieve-PPO/tree/dqn)
- SAC: [Pensieve-PPO SAC Branch](https://github.com/godka/Pensieve-PPO/tree/SAC) or [Pensieve-SAC Repository](https://github.com/godka/Pensieve-SAC)

[1] Mao H, Netravali R, Alizadeh M. Neural adaptive video streaming with Pensieve[C]//Proceedings of the Conference of the ACM Special Interest Group on Data Communication. ACM, 2017: 197-210.

[2] Xia, Zhengxu, et al. "Genet: automatic curriculum generation for learning adaptation in networking." Proceedings of the ACM SIGCOMM 2022 Conference. 2022.

[3] Huang, Tianchi, et al. "Comyco: Quality-aware adaptive video streaming via imitation learning." Proceedings of the 27th ACM international conference on multimedia. 2019.

[4] Wu, Duo, et al. "Netllm: Adapting large language models for networking." Proceedings of the ACM SIGCOMM 2024 Conference. 2024.

* We use the following command to test the *entire traces* in the dataset.

```
python run_plm.py --test --plm-type llama --plm-size base --rank 128 --device cuda:0 --trace-num -1 --model-dir  data/ft_plms/try_llama2_7b
```
