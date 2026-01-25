"""Experience pool training script for Pensieve PPO.

This module implements training from a pre-collected experience pool (ExperiencePool).
Unlike the distributed training (train.py) or imitation learning (imitate.py), this
module reads trajectories from a saved experience pool file and trains an agent
without requiring live environment interaction or multiprocessing.

The key concepts:
- ExperiencePool: Pre-collected trajectories saved to disk (e.g., by generate_exp_pool.py).
- ExpPoolTrainer: Single-process trainer that iterates over the experience pool.
- ExpPoolDataset: PyTorch Dataset wrapper for ExperiencePool.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py
"""

import argparse
import os
from typing import Callable

from .agent import AbstractTrainableAgent, EpochEndCallback, SaveModelCallback, get_available_trainable_agents
from .defaults import create_agent_factory_with_default, VIDEO_BIT_RATE, S_LEN
from .args import add_env_agent_arguments, parse_env_agent_args
from .test import add_testing_arguments
from .train import TestingCallback, MODEL_SAVE_INTERVAL, SUMMARY_DIR
from .exp_pool import ExperiencePool, ExpPoolTrainer
from .generate_exp_pool import add_exp_pool_arguments, EXP_POOL_PATH

# Default constants for experience pool training
EXP_POOL_TRAIN_EPOCHS = 100
EXP_POOL_BATCH_SIZE = 16


def prepare_exp_pool_training(
    exp_pool_path: str = EXP_POOL_PATH,
    # Agent parameters
    name: str = 'ppo',
    model_path: str = None,
    device: str = None,
    agent_options: dict = {},
    # Compatibility parameters (shared between env and agent)
    levels_quality: list = VIDEO_BIT_RATE,
    state_history_len: int = S_LEN,
    # Training parameters
    output_dir: str = SUMMARY_DIR,
    batch_size: int = EXP_POOL_BATCH_SIZE,
    train_epochs: int = EXP_POOL_TRAIN_EPOCHS,
    model_save_interval: int = MODEL_SAVE_INTERVAL,
    shuffle: bool = True,
    num_workers: int = 0,
    on_epoch_end: Callable[[int, AbstractTrainableAgent, dict], None] = EpochEndCallback(),
    on_save_model: Callable[[int, str, AbstractTrainableAgent], None] = SaveModelCallback(),
) -> ExpPoolTrainer:
    """Prepare trainer for experience pool training.

    This function creates an ExpPoolTrainer configured for training from
    a pre-collected experience pool file. No multiprocessing or live environment
    interaction is required.

    Reference:
        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L77-L127
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py

    Args:
        exp_pool_path: Path to the experience pool file (.pkl).
        name: Agent algorithm name (e.g., 'ppo').
        model_path: Path to load pre-trained model weights.
        device: PyTorch device for computation.
        agent_options: Additional kwargs for agent.
        levels_quality: Quality metric list for each bitrate level.
        state_history_len: Number of past observations in state.
        output_dir: Directory for saving logs and model checkpoints.
        batch_size: Number of trajectories per training batch.
        train_epochs: Total number of training epochs.
        model_save_interval: Interval for saving model checkpoints.
        shuffle: Whether to shuffle the dataset each epoch.
        num_workers: Number of DataLoader workers (0 for main process only).
        on_epoch_end: Callback function invoked at the end of each epoch.
                     Signature: (epoch: int, agent: AbstractTrainableAgent, info: dict) -> None
        on_save_model: Callback function invoked when model is saved.
                     Signature: (epoch: int, model_path: str, agent: AbstractTrainableAgent) -> None

    Returns:
        Configured ExpPoolTrainer instance ready for training.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load experience pool
    exp_pool = ExperiencePool.load(exp_pool_path)
    print(f"Loaded experience pool: {exp_pool}")

    # Create agent factory
    agent_factory = create_agent_factory_with_default(
        model_path=model_path,
        name=name,
        device=device,
        levels_quality=levels_quality,
        state_history_len=state_history_len,
        agent_options=agent_options,
    )

    # Create ExpPoolTrainer
    trainer = ExpPoolTrainer(
        exp_pool=exp_pool,
        agent_factory=agent_factory,
        batch_size=batch_size,
        train_epochs=train_epochs,
        model_save_interval=model_save_interval,
        output_dir=output_dir,
        shuffle=shuffle,
        num_workers=num_workers,
        on_epoch_end=on_epoch_end,
        on_save_model=on_save_model,
    )

    return trainer


def add_exp_pool_training_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for experience pool training configuration.

    Note: --exp-pool-path is added by add_exp_pool_arguments() from generate_exp_pool.py.

    Args:
        parser: ArgumentParser to add arguments to.
    """
    parser.add_argument('--output-dir', type=str, default=SUMMARY_DIR,
                        help=f"Directory for saving logs and model checkpoints "
                             f"(default: '{SUMMARY_DIR}')")
    parser.add_argument('--batch-size', type=int, default=EXP_POOL_BATCH_SIZE,
                        help=f"Number of trajectories per training batch (default: {EXP_POOL_BATCH_SIZE})")
    parser.add_argument('--train-epochs', type=int, default=EXP_POOL_TRAIN_EPOCHS,
                        help=f"Total number of training epochs (default: {EXP_POOL_TRAIN_EPOCHS})")
    parser.add_argument('--model-save-interval', type=int, default=MODEL_SAVE_INTERVAL,
                        help=f"Interval for saving model checkpoints (default: {MODEL_SAVE_INTERVAL})")
    parser.add_argument('--shuffle', action='store_true', default=True,
                        help="Shuffle the dataset each epoch (default: True)")
    parser.add_argument('--no-shuffle', action='store_false', dest='shuffle',
                        help="Disable shuffling the dataset")
    parser.add_argument('--num-workers', type=int, default=0,
                        help="Number of DataLoader workers, 0 for main process only (default: 0)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Pensieve agent from experience pool')
    add_env_agent_arguments(parser, available_agents=get_available_trainable_agents())
    add_testing_arguments(parser)
    add_exp_pool_arguments(parser)
    add_exp_pool_training_arguments(parser)
    args = parser.parse_args()

    # Post-process arguments (parse options, set seed)
    parse_env_agent_args(args)

    # Create on_save_model callback (reuse TestingCallback from train.py)
    on_save_model = TestingCallback(args=args, output_dir=args.output_dir)

    # Prepare trainer for experience pool training
    trainer = prepare_exp_pool_training(
        exp_pool_path=args.exp_pool_path,
        # Agent parameters
        name=args.agent_name,
        model_path=args.model_path,
        device=args.device,
        agent_options=args.agent_options,
        # Compatibility parameters
        levels_quality=args.levels_quality,
        state_history_len=args.state_history_len,
        # Training parameters
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        train_epochs=args.train_epochs,
        model_save_interval=args.model_save_interval,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        on_save_model=on_save_model,
    )

    # Start training from experience pool
    print(f"Starting training from experience pool...")
    print(f"Experience pool path: {args.exp_pool_path}")
    print(f"Agent: {args.agent_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total epochs: {args.train_epochs}")
    trainer.train()

    print("\nTraining completed!")
