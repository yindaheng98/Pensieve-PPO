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
pip install torch numpy gymnasium tensorboard tqdm transformers peft
```

## Quick Start

### Using the Gymnasium Environment

```python
from pensieve_ppo.agent import create_env
from pensieve_ppo.quality_ladder import QualityLadderRequest
import pensieve_ppo.quality_ladder.rl  # registers built-in RL agents

# Create environment with default Pensieve parameters
env = create_env(name='ppo', trace_folder='./src/train/', train=True)

# Initialize the episode, then explicitly download the first chunk.
initial_request = QualityLadderRequest(1)
_, info = env.reset()
obs, _, terminated, truncated, info = env.step(initial_request)
while True:
    action = QualityLadderRequest(1)  # or use your agent
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Training a PPO Agent

```bash
# Train with default settings
python -m pensieve_ppo train

# Train with custom parameters
python -m pensieve_ppo train \
    --parallel-workers 16 \
    --train-epochs 500000 \
    --model-save-interval 300 \
    --output-dir ./ppo
```

### Testing a Trained Model

```bash
# Test with a trained model
python -m pensieve_ppo test --model-path ./ppo/nn_model_ep_300.pth

# Test with custom trace folder
python -m pensieve_ppo test \
    --model-path ./ppo/nn_model_ep_300.pth \
    --test-trace-folder ./src/test/
```

> **Data path note**: The default training traces, test traces, and video chunk
> size files are currently read from the legacy `src/` tree:
> `./src/train/`, `./src/test/`, and `./src/envivio/video_size_`.
> Keep those folders present when using the default configuration, or pass
> explicit `--train-trace-folder`, `--test-trace-folder`, and
> `video_size_file_prefix` through `--player-options`.

## Package Structure

```
pensieve_ppo/
├── core/                  # Core simulation components
│   ├── trace/             # Network trace handling (bandwidth, latency)
│   ├── video/             # Video processing (chunk sizes, bitrates, playback)
│   └── simulator/         # Combines trace & video into ABR simulator
├── gym/                   # Gymnasium environment wrapper
│   ├── env.py             # ABREnv - standard RL environment interface
│   └── imitate.py         # Imitation observer wrapper
├── agent/                 # Agent implementations and training loops
│   ├── abc.py             # Abstract base classes
│   ├── registry.py        # Agent factory and registration
│   ├── trainer.py         # Distributed training framework
│   └── imitate.py         # Imitation trainer
├── quality_ladder/        # Quality-ladder request, player, and agents
│   ├── bba/               # Buffer-based baseline
│   ├── mpc/               # MPC and oracle MPC baselines
│   ├── netllm/            # NetLLM-style Decision Transformer agents
│   └── rl/                # PPO, A3C, and DQN agents
├── exp_pool/              # Experience pool data and offline trainer
├── defaults.py            # Default parameters and factory functions
├── train.py               # Training script
└── test.py                # Testing script
```

## Architecture

### Agent Class Hierarchy

The agent system follows a hierarchical inheritance structure:

```
AbstractAgent
    ├── select_action(state) -> ActionDecision
    │
    └── AbstractTrainableAgent
        ├── select_action_for_training(state) -> ActionDecision
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

**AbstractRLAgent** (`pensieve_ppo.quality_ladder.rl.abc.AbstractRLAgent`):
- Extends `AbstractTrainableAgent` with RL-specific functionality
- Implements `produce_training_batch` and `train_batch` using RL methods
- Requires subclasses to implement:
  - `train()`: Core training logic (e.g., PPO, A3C, DQN)
  - `compute_v()`: Value target computation (returns/advantages)
- Concrete implementations: `PPOAgent`, `A3CAgent`, `DQNAgent`, etc.

> **Note on RL Agent Implementations**: The agents in `pensieve_ppo/quality_ladder/rl/` (PPO, A3C, DQN) are **reinforcement learning algorithms**, not imitation learning algorithms. While the framework technically allows running them with `python -m pensieve_ppo imitate`, this is not recommended as these algorithms are designed to learn from rewards through environment interaction, not from teacher demonstrations.
>
> **Note on A3C Implementation**: The `A3CAgent` implementation is based on the A3C (Asynchronous Advantage Actor-Critic) algorithm, but the actual `Trainer` performs **synchronous updates** rather than asynchronous updates. This means all workers synchronize before each parameter update, which differs from the original A3C paper's asynchronous design.

### Agent Statefulness

**Design Principle**: Logically, an Agent should be **stateless**. All historical information needed by the Agent should be collected by the Observer and passed through the State object. However, in some special cases, an Agent may need to maintain its own "internal state".

**Stateless Agents**: The agents in `pensieve_ppo/quality_ladder/rl/` (PPO, A3C, DQN), `pensieve_ppo/quality_ladder/mpc/`, and `pensieve_ppo/quality_ladder/bba/` are all **stateless** - they do not maintain any "internal state" between `select_action` calls. Each action is computed purely from the current input state.

**"Stateful" Agents**: In certain cases, agents need to maintain "internal state". For example, in **NetLLM** (`pensieve_ppo/quality_ladder/netllm/`), the large language model needs to cache embeddings of historical states to avoid redundant computation. Since the embedding model is a trainable part of the policy, it cannot be moved into the Observer. In such cases, the Agent must maintain its own "internal state" (again, not the actual environment state), i.e., some special internal data structures that accelerate computation.

> **Reference**: The NetLLM implementation follows the architecture from [NetLLM's OfflineRLPolicy](https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/models/rl_policy.py), which uses deques (`states_dq`, `returns_dq`, `actions_dq`) to cache embeddings for autoregressive inference.

**Technical Details**: The "internal state" management in NetLLM is essentially maintaining an **embedding cache** rather than managing the actual environment state; it just reuses a "state-style" management approach to maintain internal acceleration data structures. Theoretically, this should support out-of-order `select_action` calls by querying pre-computed embeddings based on the input state. However, since the current codebase does not have out-of-order `select_action` calls, the implementation assumes sequential calls only.

**Tradeoffs** of using "state management" to handle embedding caches:
- **Pros**: Eliminates embedding cache lookup steps; allows precise control of cache size since we know exactly which embeddings are needed; better performance optimization.
- **Cons**: If the same state appears at distant timesteps, the embedding must be recomputed rather than retrieved from cache.

**Reset Method**: The `AbstractAgent.reset()` method should be called at the beginning of each episode to clear any "internal state" (e.g., embedding caches). For stateless agents, this is a no-op. For "stateful" agents like NetLLM, this clears the embedding caches via `clear_dq()`.

### Agent, Observer, and Trainer Relationships

**AbstractABRStateObserver** (`pensieve_ppo.gym.env.AbstractABRStateObserver`):
- Abstract interface for state observation and reward calculation
- Decouples state representation from environment dynamics
- Methods:
  - `reset(env) -> None`: Reset observer-owned history/state
  - `observe(env, chunk_request, result) -> (state, reward, info)`: Update state and compute reward
- Implementations:
  - `RLABRStateObserver`: For RL agents (returns `RLState` wrapping an `np.ndarray`)
  - `MPCABRStateObserver`: For MPC-based agents (returns `MPCState` dataclass)
  - `NetLLMABRStateObserver`: For NetLLM agents (returns `NetLLMState`)

**ABREnv** (`pensieve_ppo.gym.env.ABREnv`):
- Gymnasium-compatible environment wrapper
- `reset()` only initializes environment and observer state; it returns `None`
  plus reset metadata. The first usable observation is produced by an explicit
  `step(initial_request)` call.
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
- Base dataclass for environment observations
- Concrete type depends on the observer:
  - `RLState` for RL agents wraps `state_matrix` with shape `(S_INFO, S_LEN)` = `(6, 8)`
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
  - `action: ActionDecision`: Agent decision used to step the environment
  - `reward: float`: Reward received
  - `step: int`: Step index within trajectory
  - `done: bool`: Whether episode terminated/truncated
- Usage:
  - Collected during environment rollout in `Trainer.agent_worker()`
  - Stored in `trajectory: List[Step]`
  - Converted to `TrainingBatch` via `produce_training_batch()`

**TrainingBatch** (`pensieve_ppo.agent.trainable.TrainingBatch`):
- Abstract base class for training data containers
- Subclasses define algorithm-specific fields:
  - **RLTrainingBatch** (`pensieve_ppo.quality_ladder.rl.abc.RLTrainingBatch`):
    - `s_batch: List[RLState]`: States
    - `a_batch: List[List[int]]`: One-hot actions
    - `p_batch: List[List[float]]`: Action probabilities
    - `v_batch: List[float]`: Computed value targets (returns)
  - **NetLLMTrainingBatch** (`pensieve_ppo.quality_ladder.netllm.abc.NetLLMTrainingBatch`):
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
  → Step(state, action_decision, reward, step, done)
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
from pensieve_ppo.quality_ladder.rl.observer import RLABRStateObserver
from pensieve_ppo.quality_ladder.bba.observer import BBAStateObserver

# Create observers
student_observer = RLABRStateObserver()
teacher_observer = BBAStateObserver()

# Combine for imitation learning
imitation_observer = ImitationObserver(
    student_observer=student_observer,
    teacher_observer=teacher_observer,
)

# Use in environment
from pensieve_ppo.quality_ladder import QualityLadderRequest

env = ABREnv(simulator=simulator, observer=imitation_observer)
_, info = env.reset()
state, _, _, _, info = env.step(QualityLadderRequest(1))
# state.student_state: RLState for training RL agent
# state.teacher_state: BBAState for BBA agent's decision
```

**Training with Imitation Learning**:
```bash
# Train student agent (PPO) to imitate teacher agent (BBA)
python -m pensieve_ppo imitate \
    --agent-name ppo \
    --teacher-agent-name bba \
    --parallel-workers 16 \
    --train-epochs 500000
```

> **Warning**: The RL agents (`ppo`, `a3c`, `dqn` in `pensieve_ppo/quality_ladder/rl/`) are **reinforcement learning algorithms** designed to learn from reward signals, not from teacher demonstrations. Although the framework allows running them with `python -m pensieve_ppo imitate`, this is **not recommended** for production use. For proper imitation learning, consider using agents specifically designed for behavioral cloning or other imitation learning methods (e.g., `netllm`).

## API Reference

### Creating Environment and Agent

```python
from pensieve_ppo.agent import create_env
from pensieve_ppo.defaults import (
    create_env_agent,
    create_env_agent_factory,
)
import pensieve_ppo.quality_ladder.rl  # registers built-in RL agents

# Create just the environment
env = create_env(
    name='ppo',
    trace_folder='./src/train/',
    train=True,
    player_kwargs={
        'video_size_file_prefix': './src/envivio/video_size_',
        'quality': [300., 750., 1200., 1850., 2850., 4300.],  # Kbps
    },
)

# Create compatible env and agent pair
env, agent = create_env_agent(
    name='ppo',
    model_path='./ppo/nn_model_ep_300.pth',  # Optional: load pretrained weights
    agent_kwargs={'device': 'cuda'},
)

# Create factories for distributed training
env_factory, agent_factory = create_env_agent_factory(
    name='ppo',
    train=True,
)
```

### Using the Agent

```python
from pensieve_ppo.agent import create_agent
import pensieve_ppo.quality_ladder.rl  # registers "ppo"
from pensieve_ppo.quality_ladder import QualityLadderRequest

# Create agent directly
agent = create_agent(
    name='ppo',
    agent_kwargs={
        'state_dim': (6, 8),  # (S_INFO, S_LEN)
        'action_dim': 6,      # Number of bitrate levels
        'device': 'cuda',
        'learning_rate': 1e-4,
        'gamma': 0.99,
    },
)

# Predict action
initial_request = agent.reset(QualityLadderRequest(1))
_, _ = env.reset()
state, _, _, _, _ = env.step(initial_request)
decision = agent.select_action(state)
action = decision.action_index
action_prob = decision.action_prob

# Train on batch
metrics = agent.train(s_batch, a_batch, p_batch, v_batch, epoch)
```

### Extending Without Editing Package Code

Pensieve-PPO discovers algorithms through a runtime registry. A registry entry
binds three compatible classes together:

- `agent_cls`: an `AbstractAgent` that chooses the next `VideoChunkRequest`
- `observer_cls`: an `AbstractABRStateObserver` that converts simulator results
  into the state type expected by the agent and computes rewards
- `player_cls`: a `VideoPlayer` that resolves requests into chunk sizes and
  quality values

Put your extension in an importable Python package, call `register()` at import
time, and load that package with `--registry-package` / `--import-package`.
No files under `pensieve_ppo/` need to be modified.

**Registration API**:
```python
from typing import Optional, Type

from pensieve_ppo.agent import register
from pensieve_ppo.agent.abc import AbstractAgent
from pensieve_ppo.agent.trainable import AbstractTrainableAgent
from pensieve_ppo.core.video import VideoPlayer
from pensieve_ppo.gym.env import AbstractABRStateObserver

register(
    name: str,
    agent_cls: Type[AbstractAgent],
    observer_cls: Type[AbstractABRStateObserver],
    player_cls: Type[VideoPlayer],
    trainable_agent_cls: Optional[Type[AbstractTrainableAgent]] = None,
) -> None
```

If `agent_cls` is already a subclass of `AbstractTrainableAgent`,
`trainable_agent_cls` is detected automatically. Only trainable agents appear in
training commands such as `train` and `imitate-exp-pool`; inference-only agents
can still be used by `test`, as imitation teachers, or as experience-pool actors.

**External package layout**:
```text
my_pensieve_ext/
├── __init__.py
└── registry.py
```

`my_pensieve_ext/__init__.py` can be empty. Put the registration side effect in
`my_pensieve_ext/registry.py`:

```python
from dataclasses import dataclass
from typing import Optional

from pensieve_ppo.agent import register
from pensieve_ppo.agent.abc import AbstractAgent
from pensieve_ppo.core.simulator import StepResult
from pensieve_ppo.core.video import VideoChunkRequest
from pensieve_ppo.gym import ABREnv, AbstractABRStateObserver, State
from pensieve_ppo.quality_ladder import (
    QualityLadderActionDecision,
    QualityLadderResolvedChunk,
    QualityLadderRequest,
    QualityLadderVideoPlayer,
)


@dataclass
class MyState(State):
    buffer_size: float
    last_quality: float


class MyObserver(AbstractABRStateObserver):
    def __init__(self, rebuf_penalty: float = 4.3):
        self.rebuf_penalty = rebuf_penalty

    def reset(self, env: ABREnv) -> None:
        """Reset observer-owned history. No usable state is returned here."""
        pass

    def observe(
        self,
        env: ABREnv,
        chunk_request: VideoChunkRequest,
        result: StepResult,
    ) -> tuple[MyState, float, dict]:
        if not isinstance(result.resolved_chunk, QualityLadderResolvedChunk):
            raise TypeError(f"expected QualityLadderResolvedChunk, got {type(result.resolved_chunk).__name__}")
        quality = result.resolved_chunk.quality
        state = MyState(
            buffer_size=result.buffer_size,
            last_quality=quality,
        )
        reward = quality - self.rebuf_penalty * result.rebuffer
        info = {
            "quality": quality,
            "buffer_size": result.buffer_size,
            "rebuffer": result.rebuffer,
            "video_chunk_size": result.video_chunk_size,
            "delay": result.delay,
        }
        return state, reward, info


class MyPlayer(QualityLadderVideoPlayer):
    """Use the built-in quality-ladder data loader with a custom registered name."""


class MyAgent(AbstractAgent):
    def __init__(self, low_buffer_level: float = 5.0, initial_level: int = 0):
        self.low_buffer_level = low_buffer_level
        self.initial_level = initial_level

    def reset(
        self,
        initial_chunk_request: Optional[VideoChunkRequest] = None,
    ) -> VideoChunkRequest:
        return initial_chunk_request or QualityLadderRequest(self.initial_level)

    def select_action(self, state: MyState) -> QualityLadderActionDecision:
        level = 0 if state.buffer_size < self.low_buffer_level else 1
        return QualityLadderActionDecision(QualityLadderRequest(level))


register("my-agent", MyAgent, MyObserver, MyPlayer)
```

This example is an inference-only agent. To use the existing distributed
training loop, implement `AbstractTrainableAgent` or a subclass such as
`AbstractRLAgent`, and make sure the decisions and training batches carry the
metadata your trainer needs, such as action indices and probabilities.

Use the same registered name from Python:

```python
import my_pensieve_ext.registry  # registers "my-agent"

from pensieve_ppo.defaults import create_env_agent

env, agent = create_env_agent(
    name="my-agent",
    agent_kwargs={"low_buffer_level": 8.0},
    observer_kwargs={"rebuf_penalty": 5.0},
    player_kwargs={"name": "envivio"},
)
```

Or load the extension directly from the command line:

```bash
python -m pensieve_ppo \
    --registry-package my_pensieve_ext.registry \
    test \
    --agent-name my-agent \
    --agent-options low_buffer_level=8.0 initial_level=1 \
    --observer-options rebuf_penalty=5.0 \
    --player-options name='envivio'
```

The module entry point is also valid when you prefer invoking one command
module directly:

```bash
python -m pensieve_ppo.test \
    --registry-package my_pensieve_ext.registry \
    --agent-name my-agent
```

`--registry-package` can be repeated when you need multiple extension packages.
Unqualified names such as `quality_ladder` are resolved under `pensieve_ppo`
for built-ins. For external code, prefer a fully qualified dotted module path
such as `my_pensieve_ext.registry`. If the package is not installed, make its
parent directory visible through `PYTHONPATH` or install it with
`pip install -e /path/to/my-extension`.

To check what has been registered:

```python
from pensieve_ppo.agent import get_available_agents, get_available_trainable_agents

print(get_available_agents())
print(get_available_trainable_agents())
```

### Default Quality-Ladder Environment Details

The default `ppo`, `a3c`, and `dqn` registrations use
`QualityLadderVideoPlayer` with `RLABRStateObserver`. That observer returns an
`RLState` whose `state_matrix` has shape `(6, state_history_len)`, with
`state_history_len=8` by default:

| Index | Description |
|-------|-------------|
| 0 | Last quality normalized by max quality |
| 1 | Buffer size normalized by buffer_norm_factor |
| 2 | Throughput (chunk_size / delay) in Mbps |
| 3 | Delay normalized by buffer_norm_factor |
| 4 | Next chunk sizes at each bitrate level (MB) |
| 5 | Remaining chunks normalized by total |

Actions are `QualityLadderRequest(level)` values. With the built-in Envivio
ladder there are six levels, so valid levels are `0` through `5`.

**Reward**: `quality - 4.3 * rebuffer - 1.0 * |quality_change|`

### NetLLM Agents

NetLLM-style agents are registered under names such as `netllm-gpt2`,
`netllm-llama`, and `netllm-gpt2-lora`. They use the `NetLLMABRStateObserver`
and the model wrappers in `pensieve_ppo.quality_ladder.netllm`.

When creating a NetLLM agent, provide the reward normalization range required
by the offline RL data processing:

```bash
python -m pensieve_ppo imitate-exp-pool \
    --agent-name netllm-gpt2 \
    -o state_dim=(6,6) min_reward=-10.0 max_reward=10.0 plm_size='small'
```

NetLLM currently expects a state history length of `6`, while the default PPO
observer uses `8`. Pass `state_dim=(6,6)` through `--agent-options` for NetLLM
runs unless you are using a custom NetLLM-compatible state encoder.

## Command Line Options

You can run commands either through the package-level entry point
(`python -m pensieve_ppo <command> ...`) or the module entry point
(`python -m pensieve_ppo.<module> ...`). The package-level commands are:
`train`, `test`, `imitate`, `generate-exp-pool`, and `imitate-exp-pool`.
After installation, the same package-level CLI is also available as
`pensieve-ppo <command> ...`.

### Training (`python -m pensieve_ppo train`)

```
--train-trace-folder    Training trace folder (default: ./src/train/)
--output-dir            Output directory (default: ./ppo)
--parallel-workers      Number of parallel workers (default: 16)
--max-steps-per-epoch   Maximum steps per epoch per worker (default: 1000)
--train-epochs          Total training epochs (default: 500000)
--model-save-interval   Model checkpoint interval (default: 300)
--model-path            Resume from pretrained model
```

### Testing (`python -m pensieve_ppo test`)

```
--test-trace-folder     Test trace folder (default: ./src/test/)
--model-path            Path to trained model weights
--test-log-file-prefix  Prefix for test log files
```

### Common Options

```
--registry-package      Import package/module before building choices; repeatable
--import-package        Alias of --registry-package
--agent-name            Registered agent name (default: ppo)
--random-seed           Random seed (default: 42)
-o, --agent-options     Extra agent kwargs (e.g., learning_rate=1e-4 device='cuda')
--observer-options      Extra observer kwargs (e.g., state_history_len=6)
--player-options        Extra player kwargs (e.g., name='envivio')
```

Values passed through `--agent-options`, `--observer-options`, and
`--player-options` are parsed as Python expressions. Quote string values, for
example `plm_size='small'`. If your shell strips inner quotes, quote the whole
assignment, for example `"device='cuda'"`.

## TensorBoard Monitoring

Monitor training in real-time:

```bash
tensorboard --logdir=./ppo
```

---

# Legacy Notes From Original README

### Updates

**Jan. 18, 2025:** We removed the rate-based method and added NetLLM [4].

**May. 4, 2024:** We removed the Elastic, revised  BOLA, and add new baseline Comyco [3] and Genet [2].

**Jan. 26, 2024:** We are excited to announce significant updates to Pensieve-PPO! We have replaced TensorFlow with PyTorch, and we have achieved a similar training speed while training models that rival in performance.

*For the TensorFlow version, please check [Pensieve-PPO TF Branch](https://github.com/godka/Pensieve-PPO/tree/master).*

**Dec. 28, 2021:** In a previous update, we enhanced Pensieve-PPO with several state-of-the-art technologies, including Dual-Clip PPO and adaptive entropy decay.

## About Pensieve-PPO

Pensieve-PPO is a user-friendly PyTorch implementation of Pensieve [1], a neural adaptive video streaming system. Unlike A3C, we utilize the Proximal Policy Optimization (PPO) algorithm for training.

This stable version of Pensieve-PPO includes both the training and test datasets.

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
