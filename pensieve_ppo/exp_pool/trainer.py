"""ExpPoolTrainer for training agents from ExperiencePool data.

This module implements the ExpPoolTrainer class that trains agents using
pre-collected experience data from an ExperiencePool, without requiring
multiprocessing.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py
"""

from typing import Callable

from torch.utils.data import DataLoader
from tqdm import tqdm

from ..agent.trainable import AbstractTrainableAgent, TrainBatchInfo
from ..agent.trainer import EpochEndCallback, SaveModelCallback
from .dataset import ExpPoolDataset
from .pool import ExperiencePool


def _identity_collate_fn(batch):
    """Identity collate function that returns batch as-is.

    This is defined at module level (not as a lambda) to allow pickling
    when DataLoader uses num_workers > 0.
    """
    return batch


class ExpPoolTrainer:
    """Trainer for agents using pre-collected ExperiencePool data.

    This class implements a single-process training loop that:
    1. Loads trajectories from an ExperiencePool (List[List[Step]])
    2. Converts trajectories to TrainingBatch using agent.produce_training_batch
    3. Trains the agent using agent.train_batch

    Unlike the distributed Trainer class, this does not use multiprocessing.
    The training data comes from a pre-collected ExperiencePool instead of
    live environment interactions.

    Example:
        >>> pool = ExperiencePool.load("exp_pool.pkl")
        >>> agent = MyTrainableAgent()
        >>> trainer = ExpPoolTrainer(
        ...     exp_pool=pool,
        ...     agent_factory=lambda: MyTrainableAgent(),
        ...     batch_size=16,
        ...     train_epochs=100,
        ... )
        >>> trainer.train()

    Reference:
        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py
    """

    def __init__(
        self,
        exp_pool: ExperiencePool,
        agent_factory: Callable[[], AbstractTrainableAgent],
        batch_size: int = 1,
        train_epochs: int = 100,
        model_save_interval: int = 10,
        output_dir: str = './exp_pool_ppo',
        # pretrained_model_path: Optional[str] = None,  # Model loading is handled in create_agent
        shuffle: bool = True,
        num_workers: int = 0,
        on_epoch_end: Callable[[int, AbstractTrainableAgent, TrainBatchInfo], None] = EpochEndCallback(),
        on_save_model: Callable[[int, str, AbstractTrainableAgent], None] = SaveModelCallback(),
    ):
        """Initialize the ExpPoolTrainer.

        Args:
            exp_pool: ExperiencePool containing the training data.
            agent_factory: Factory function () -> AbstractTrainableAgent.
            batch_size: Number of trajectories per training batch.
            train_epochs: Total number of training epochs.
            model_save_interval: Interval for saving model checkpoints.
            output_dir: Directory for saving logs and model checkpoints.
            shuffle: Whether to shuffle the dataset each epoch.
            num_workers: Number of DataLoader workers (0 for main process only).
            on_epoch_end: Callback invoked at the end of each epoch.
            on_save_model: Callback invoked when model is saved.
        """
        self.exp_pool = exp_pool
        self.agent_factory = agent_factory
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.model_save_interval = model_save_interval
        self.summary_dir = output_dir
        # self.nn_model = pretrained_model_path  # Model loading is handled in create_agent
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.on_epoch_end = on_epoch_end
        self.on_save_model = on_save_model

    def train(self) -> AbstractTrainableAgent:
        """Run the training loop.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L77-L127

        Returns:
            The trained agent.
        """
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L83-L85
        actor = self.agent_factory()

        # Model loading is now handled in create_agent via model_path parameter
        # # restore neural net parameters
        # # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L90-L100
        # if self.nn_model is not None:
        #     actor.load(self.nn_model)
        #     print('Model restored.')

        # Create dataset and dataloader
        # Note: We don't pass the agent to the dataset to avoid pickle issues
        # with models that have hooks (e.g., from enable_input_require_grads).
        # The conversion to TrainingBatch is done in the training loop below.
        dataset = ExpPoolDataset(self.exp_pool)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=_identity_collate_fn,  # Return list of (trajectory, done) as-is
        )

        print(f'ExpPoolTrainer: {dataset.num_trajectories} trajectories, '
              f'{dataset.total_steps} total steps')

        # while True:  # assemble training batches from agents, compute the gradients
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L96
        for epoch in range(1, self.train_epochs + 1):
            batch_count = 0
            train_info: TrainBatchInfo = None  # type: ignore[assignment]

            # Create progress bar for batches
            pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{self.train_epochs}', leave=False)
            for trajectory_batch in pbar:
                # trajectory_batch is List[(trajectory, done)]
                # Convert to TrainingBatch here to avoid pickle issues with agent
                training_batches = [
                    actor.produce_training_batch(trajectory, done)
                    for trajectory, done in trajectory_batch
                ]
                # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L114
                train_info = actor.train_batch(training_batches, epoch)

                # Update progress bar with train_info
                postfix = {'loss': f'{train_info.loss:.4f}'}
                pbar.set_postfix(postfix)

                batch_count += 1

            pbar.close()
            # Add batch_count to the train_info's extra dict
            train_info.extra['batch_count'] = batch_count

            # Callback for epoch end
            self.on_epoch_end(epoch, actor, train_info)  # TODO: epoch here is different with epoch in ..agent.trainer

            # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L116-L127
            if epoch % self.model_save_interval == 0:
                # Save the neural net parameters to disk.
                model_path = f'{self.summary_dir}/nn_model_ep_{epoch}.pth'
                actor.save(model_path)

                # Callback for model saving (e.g., testing and logging)
                self.on_save_model(epoch, model_path, actor)

        return actor
