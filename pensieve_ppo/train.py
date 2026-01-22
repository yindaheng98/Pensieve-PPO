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
from typing import Callable

from torch.utils.tensorboard import SummaryWriter

from .agent import AbstractTrainableAgent, Trainer, SaveModelCallback, get_available_trainable_agents
from .defaults import create_env_agent_factory_with_default, TRAIN_TRACES
from .args import add_env_agent_arguments, parse_env_agent_args
from .test import main as test_main, calculate_test_statistics, add_testing_arguments

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

        This function runs testing using test.py's main function and logs the results
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

        # Run testing using test.py's main function
        log_file_prefix = test_main(test_args)

        # Calculate statistics from test results
        stats = calculate_test_statistics(log_file_prefix)

        rewards_min = stats['rewards_min']
        rewards_5per = stats['rewards_5per']
        rewards_mean = stats['rewards_mean']
        rewards_median = stats['rewards_median']
        rewards_95per = stats['rewards_95per']
        rewards_max = stats['rewards_max']
        avg_entropy = stats['avg_entropy']

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
        writer.add_scalar('Entropy', avg_entropy, epoch)
        writer.flush()

        print(f'Epoch {epoch}: avg_reward={rewards_mean:.4f}, avg_entropy={avg_entropy:.4f}')


def prepare_training(
    *args,
    output_dir: str = SUMMARY_DIR,
    parallel_workers: int = NUM_AGENTS,
    steps_per_epoch: int = TRAIN_SEQ_LEN,
    train_epochs: int = TRAIN_EPOCH,
    model_save_interval: int = MODEL_SAVE_INTERVAL,
    pretrained_model_path: str = None,
    on_save_model: Callable[[int, str, AbstractTrainableAgent], None] = None,
    **kwargs,
) -> Trainer:
    """Prepare trainer for distributed training.

    Wrapper for create_env_agent_factory_with_default with train=True.
    See create_env_agent_factory_with_default for available parameters.

    Reference:
        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L77-L127

    Args:
        *args: Positional arguments passed to create_env_agent_factory_with_default.
        output_dir: Directory for saving logs and model checkpoints.
        parallel_workers: Number of parallel worker agents.
        steps_per_epoch: Number of environment steps per epoch per worker.
        train_epochs: Total number of training epochs.
        model_save_interval: Interval for saving model checkpoints.
        pretrained_model_path: Path to pre-trained model to resume from.
        on_save_model: Callback function invoked when model is saved.
                     Signature: (epoch: int, model_path: str, agent: AbstractTrainableAgent) -> None
        **kwargs: Keyword arguments passed to create_env_agent_factory_with_default.

    Returns:
        Configured Trainer instance ready for training.
    """
    # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L26-L27
    os.makedirs(output_dir, exist_ok=True)

    # Create environment and agent factories
    # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L83-L85
    # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L131-L133
    env_factory, agent_factory = create_env_agent_factory_with_default(
        *args, train=True, **kwargs,
    )

    # Create trainer
    # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L77-L176
    trainer = Trainer(
        env_factory=env_factory,
        agent_factory=agent_factory,
        parallel_workers=parallel_workers,
        steps_per_epoch=steps_per_epoch,
        train_epochs=train_epochs,
        model_save_interval=model_save_interval,
        output_dir=output_dir,
        pretrained_model_path=pretrained_model_path,
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
    parser.add_argument('--steps-per-epoch', type=int, default=TRAIN_SEQ_LEN,
                        help=f"Number of environment steps per epoch per worker "
                             f"(default: {TRAIN_SEQ_LEN})")
    parser.add_argument('--train-epochs', type=int, default=TRAIN_EPOCH,
                        help=f"Total number of training epochs (default: {TRAIN_EPOCH})")
    parser.add_argument('--model-save-interval', type=int, default=MODEL_SAVE_INTERVAL,
                        help=f"Interval for saving model checkpoints (default: {MODEL_SAVE_INTERVAL})")
    parser.add_argument('--pretrained-model-path', type=str, default=None,
                        help="Path to pre-trained model to resume training from (default: None)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Pensieve agent')
    add_env_agent_arguments(parser, available_agents=get_available_trainable_agents())
    add_testing_arguments(parser)
    add_training_arguments(parser)
    args = parser.parse_args()

    # Post-process arguments (parse options, set seed)
    parse_env_agent_args(args)

    # Create on_save_model callback
    on_save_model = TestingCallback(args=args, output_dir=args.output_dir)

    # Prepare trainer
    trainer = prepare_training(
        trace_folder=args.train_trace_folder,
        # Agent parameters
        name=args.agent_name,
        model_path=args.model_path,
        device=args.device,
        agent_options=args.agent_options,
        # Shared parameters
        levels_quality=args.levels_quality,
        state_history_len=args.state_history_len,
        initial_level=args.initial_level,
        env_options=args.env_options,
        # Training parameters
        output_dir=args.output_dir,
        parallel_workers=args.parallel_workers,
        steps_per_epoch=args.steps_per_epoch,
        train_epochs=args.train_epochs,
        model_save_interval=args.model_save_interval,
        pretrained_model_path=args.pretrained_model_path,
        on_save_model=on_save_model,
    )

    # Start training
    # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L197-L198
    print(f"Starting training with {args.parallel_workers} parallel workers...")
    print(f"Output directory: {args.output_dir}")
    trainer.train()
