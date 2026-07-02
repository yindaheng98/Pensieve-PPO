"""Training script for Pensieve PPO.

This module implements the complete training pipeline using distributed
parallel agents, with periodic testing and TensorBoard logging.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py
"""

import argparse
import copy
import os
import shutil
from typing import Callable, Optional

from torch.utils.tensorboard import SummaryWriter

from .agent import AbstractTrainableAgent, Trainer, SaveModelCallback, EpochEndCallback, get_available_trainable_agents
from .defaults import create_env_agent_factory, TRAIN_TRACES
from .args import add_env_agent_arguments, parse_env_agent_args, prepare_registry_package
from .test import run_evaluation, calculate_test_statistics, add_testing_arguments

# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L14-L23
NUM_AGENTS = 16
TRAIN_SEQ_LEN = 1000  # take as a train batch
TRAIN_EPOCH = 500000
MODEL_SAVE_INTERVAL = 300
SUMMARY_DIR = './ppo'


class TestingCallback(SaveModelCallback):
    """Picklable callback for testing and logging during training."""

    def __init__(self, args: argparse.Namespace, output_dir: str):
        # Create output directory and necessary objects
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.writer = None
        log_file_path = os.path.join(output_dir, 'log_test.txt')
        self.log_file_path = log_file_path
        with open(self.log_file_path, 'w'):
            pass
        self.args = args

    def __call__(self, epoch: int, model_path: str, agent: AbstractTrainableAgent) -> None:
        """Callback invoked when model is saved.

        This function runs test.py's evaluation helper and logs the results
        to both the test log file and TensorBoard.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L31-L75

        Args:
            epoch: Current training epoch.
            model_path: Path to saved model.
            agent: Trained agent (used for logging entropy weight if available).
        """
        self.writer = self.writer or SummaryWriter(log_dir=self.output_dir)
        args, writer, log_file_path = self.args, self.writer, self.log_file_path
        # Extract test log folder from test_log_file_prefix and clean up
        test_log_folder = os.path.dirname(args.test_log_file_prefix)
        if os.path.exists(test_log_folder):
            shutil.rmtree(test_log_folder)
        os.makedirs(test_log_folder, exist_ok=True)

        # Create a copy of args and update model_path and test_log_file_prefix
        test_args = copy.deepcopy(args)
        test_args.model_path = model_path

        # Run testing using test.py's evaluation helper
        log_file_prefix = run_evaluation(test_args)

        # Calculate statistics from test results
        stats = calculate_test_statistics(log_file_prefix)

        rewards_min = stats['rewards_min']
        rewards_5per = stats['rewards_5per']
        rewards_mean = stats['rewards_mean']
        rewards_median = stats['rewards_median']
        rewards_95per = stats['rewards_95per']
        rewards_max = stats['rewards_max']
        with open(log_file_path, 'a') as log_file:
            log_file.write(str(epoch) + '\t' +
                           str(rewards_min) + '\t' +
                           str(rewards_5per) + '\t' +
                           str(rewards_mean) + '\t' +
                           str(rewards_median) + '\t' +
                           str(rewards_95per) + '\t' +
                           str(rewards_max) + '\n')

        # Log to TensorBoard
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L124-L127
        agent.tensorboard_logging(writer, epoch)
        writer.add_scalar('Reward', rewards_mean, epoch)
        writer.flush()

        print(f'Epoch {epoch}: avg_reward={rewards_mean:.4f}')


def prepare_training(
    name: str = 'ppo',
    trace_folder: Optional[str] = None,
    random_seed: Optional[int] = None,
    observer_kwargs: dict = {},
    player_kwargs: dict = {},
    model_path: Optional[str] = None,
    agent_kwargs: dict = {},
    output_dir: str = SUMMARY_DIR,
    parallel_workers: int = NUM_AGENTS,
    max_steps_per_epoch: int = TRAIN_SEQ_LEN,
    train_epochs: int = TRAIN_EPOCH,
    model_save_interval: int = MODEL_SAVE_INTERVAL,
    on_epoch_end: Callable[[int, AbstractTrainableAgent, dict], None] = EpochEndCallback(),
    on_save_model: Callable[[int, str, AbstractTrainableAgent], None] = SaveModelCallback(),
) -> Trainer:
    """Prepare trainer for distributed training.

    Wrapper for create_env_agent_factory with train=True. Environment and
    agent parameters are listed in the same order as create_env_agent,
    excluding train.

    Reference:
        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L77-L127

    Args:
        name: Agent name.
        trace_folder: Folder containing training traces.
        random_seed: Random seed.
        observer_kwargs: Keyword arguments for the observer.
        player_kwargs: Keyword arguments for the video player.
        model_path: Optional path to load a model from.
        agent_kwargs: Keyword arguments for the agent.
        output_dir: Directory for saving logs and model checkpoints.
        parallel_workers: Number of parallel worker agents.
        max_steps_per_epoch: Maximum number of environment steps per epoch per worker.
            The actual number may be less if the episode terminates or truncates early.
        train_epochs: Total number of training epochs.
        model_save_interval: Interval for saving model checkpoints.
        on_epoch_end: Callback function invoked at the end of each epoch.
                     Signature: (epoch: int, agent: AbstractTrainableAgent, info: dict) -> None
        on_save_model: Callback function invoked when model is saved.
                     Signature: (epoch: int, model_path: str, agent: AbstractTrainableAgent) -> None

    Returns:
        Configured Trainer instance ready for training.
    """
    # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L26-L27
    os.makedirs(output_dir, exist_ok=True)

    # Create environment and agent factories
    # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L83-L85
    # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L131-L133
    env_factory, agent_factory = create_env_agent_factory(
        name=name,
        trace_folder=trace_folder,
        train=True,
        random_seed=random_seed,
        observer_kwargs=observer_kwargs,
        player_kwargs=player_kwargs,
        model_path=model_path,
        agent_kwargs=agent_kwargs,
    )

    # Create trainer
    # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L77-L176
    trainer = Trainer(
        env_factory=env_factory,
        agent_factory=agent_factory,
        parallel_workers=parallel_workers,
        max_steps_per_epoch=max_steps_per_epoch,
        train_epochs=train_epochs,
        model_save_interval=model_save_interval,
        output_dir=output_dir,
        on_epoch_end=on_epoch_end,
        on_save_model=on_save_model,
    )

    return trainer


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """Add training-specific arguments to parser."""
    parser.add_argument('--train-trace-folder', type=str, default=TRAIN_TRACES,
                        help=f"Folder containing network bandwidth trace files for training "
                             f"(default: '{TRAIN_TRACES}')")
    parser.add_argument('--output-dir', type=str, default=SUMMARY_DIR,
                        help=f"Directory for saving logs and model checkpoints "
                             f"(default: '{SUMMARY_DIR}')")
    parser.add_argument('--parallel-workers', type=int, default=NUM_AGENTS,
                        help=f"Number of parallel worker agents (default: {NUM_AGENTS})")
    parser.add_argument('--max-steps-per-epoch', type=int, default=TRAIN_SEQ_LEN,
                        dest='max_steps_per_epoch',
                        help=f"Maximum number of environment steps per epoch per worker. "
                             f"Actual steps may be less if episode ends early "
                             f"(default: {TRAIN_SEQ_LEN})")
    parser.add_argument('--train-epochs', type=int, default=TRAIN_EPOCH,
                        help=f"Total number of training epochs (default: {TRAIN_EPOCH})")
    parser.add_argument('--model-save-interval', type=int, default=MODEL_SAVE_INTERVAL,
                        help=f"Interval for saving model checkpoints (default: {MODEL_SAVE_INTERVAL})")


DESCRIPTION = 'Train Pensieve agent'


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add command-line arguments for training."""
    prepare_registry_package(parser)
    add_env_agent_arguments(parser, available_agents=get_available_trainable_agents())
    add_testing_arguments(parser)
    add_training_arguments(parser)


def main(args: argparse.Namespace) -> None:
    """Run distributed training from parsed command-line arguments."""
    # Post-process arguments (parse options, set seed)
    parse_env_agent_args(args)

    # Create on_save_model callback
    on_save_model = TestingCallback(args=args, output_dir=args.output_dir)

    # Prepare trainer
    trainer = prepare_training(
        name=args.agent_name,
        trace_folder=args.train_trace_folder,
        random_seed=args.random_seed,
        observer_kwargs=args.observer_options,
        player_kwargs=args.player_options,
        model_path=args.model_path,
        agent_kwargs=args.agent_options,
        output_dir=args.output_dir,
        parallel_workers=args.parallel_workers,
        max_steps_per_epoch=args.max_steps_per_epoch,
        train_epochs=args.train_epochs,
        model_save_interval=args.model_save_interval,
        on_save_model=on_save_model,
    )

    # Start training
    # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L197-L198
    print(f"Starting training with {args.parallel_workers} parallel workers...")
    print(f"Output directory: {args.output_dir}")
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    add_arguments(parser)
    main(parser.parse_args())
