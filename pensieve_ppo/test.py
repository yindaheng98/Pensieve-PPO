"""Testing script for Pensieve PPO.

This module implements the complete testing pipeline, evaluating trained models
on test traces and logging detailed results.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .agent import AbstractAgent, get_available_agents
from .defaults import create_env_agent
from .gym.env import ABREnv
from .args import add_env_agent_arguments, parse_env_agent_args, prepare_registry_package

# Conversion constant (milliseconds to seconds)
M_IN_K = 1000.0

# Reference: https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L22
TEST_LOG_FOLDER = './test_results/'

# Default log file prefix (agent name will be appended)
# Reference: https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L25
LOG_FILE_PREFIX = os.path.join(TEST_LOG_FOLDER, 'log_sim_')


def prepare_testing(
    name: str = 'ppo',
    trace_folder: Optional[str] = None,
    random_seed: Optional[int] = None,
    observer_kwargs: dict = {},
    player_kwargs: dict = {},
    model_path: Optional[str] = None,
    agent_kwargs: dict = {},
) -> Tuple[ABREnv, AbstractAgent]:
    """Prepare env and agent for testing.

    Wrapper for create_env_agent with train=False. Parameters are listed in
    the same order as create_env_agent, excluding train.
    """
    return create_env_agent(
        name=name,
        trace_folder=trace_folder,
        train=False,
        random_seed=random_seed,
        observer_kwargs=observer_kwargs,
        player_kwargs=player_kwargs,
        model_path=model_path,
        agent_kwargs=agent_kwargs,
    )


def testing(
    env: ABREnv,
    agent: AbstractAgent,
    log_file_prefix: str,
) -> None:
    """Run testing on all test traces.

    Reference:
        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L30-L151

    The loop structure matches src/test.py exactly:
        while True:
            1. step(action) - download chunk
            2. write log
            3. predict next chunk request
            4. if end_of_video: reset and open new log

    Args:
        env: The ABR environment.
        agent: The trained agent.
        log_file_prefix: Prefix for log file paths.
    """
    # Create log directory if needed
    log_dir = os.path.dirname(log_file_prefix)
    os.makedirs(log_dir, exist_ok=True)

    # s_info, s_len = agent.s_dim
    # a_dim = agent.a_dim # no where to use a_dim

    # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L36
    trace_progress = env.simulator.trace_simulator.get_trace_progress()
    all_file_names = trace_progress.all_trace_names

    # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L41-L42
    log_path = log_file_prefix + '_' + all_file_names[trace_progress.trace_index]
    jsonl_path = f"{log_path}.jsonl"
    # First time open with 'w' to clear the file
    with open(log_path, 'w'):
        pass
    with open(jsonl_path, 'w'):
        pass

    # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L53
    initial_chunk_request = agent.reset()  # Reset agent's "internal state" (e.g., embedding caches) for new episode
    env.reset()

    # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L55-L66
    # last_bit_rate = initial_level # no where to use last_bit_rate
    chunk_request = initial_chunk_request

    # action_vec = np.zeros(a_dim) # no where to use action_vec
    # action_vec[bit_rate] = 1

    # s_batch = [np.zeros((s_info, s_len))] # no where to use s_batch
    # a_batch = [action_vec] # no where to use a_batch
    # r_batch = [] # no where to use r_batch
    video_count = 0

    # Progress bar for testing
    total_chunks = env.simulator.video_player.total_chunks
    interactive = sys.stderr.isatty()
    pbar = tqdm(total=len(all_file_names), desc="Testing", unit="trace", position=0)
    pbar_step = tqdm(total=total_chunks, desc="Steps", unit="chunk", position=1, leave=False, disable=not interactive)

    while True:  # serve video forever
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L69-L85
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L100-L115
        # the action is from the last decision
        # this is to make the framework similar to the real
        state, reward, end_of_video, truncated, info = env.step(chunk_request)
        pbar_step.update(1)

        # r_batch.append(reward)

        # last_bit_rate = bit_rate

        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L89-L98
        # log time_stamp, bit_rate, buffer_size, reward
        # Open file: first time with 'w' to clear, then with 'a' to append
        with open(log_path, 'a') as log_file:
            log_file.write(str(info['time_stamp'] / M_IN_K) + '\t' +
                           str(info['quality']) + '\t' +
                           str(info['buffer_size']) + '\t' +
                           str(info['rebuffer']) + '\t' +
                           str(info['video_chunk_size']) + '\t' +
                           str(info['delay']) + '\t' +
                           str(reward) + '\n')
        with open(jsonl_path, 'a') as jsonl_file:
            jsonl_file.write(json.dumps({'reward': reward, 'info': info}) + '\n')

        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L117-L123
        decision = agent.select_action(state)
        chunk_request = decision.action

        # s_batch.append(state)

        if end_of_video:
            # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L126-L147
            # Append newline and close file
            with open(log_path, 'a') as log_file:
                log_file.write('\n')

            # last_bit_rate = initial_level

            # del s_batch[:]
            # del a_batch[:]
            # del r_batch[:]

            # action_vec = np.zeros(a_dim)
            # action_vec[bit_rate] = 1

            # s_batch.append(np.zeros((s_info, s_len)))
            # a_batch.append(action_vec)

            video_count += 1
            pbar.update(1)
            pbar_step.reset()

            if video_count >= len(all_file_names):
                pbar_step.close()
                pbar.close()
                break

            # Reset for next trace - only initializes state
            initial_chunk_request = agent.reset()
            chunk_request = initial_chunk_request
            env.reset(options={'reset_time_stamp': False})

            # Open new log file
            # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L149-L150
            trace_progress = env.simulator.trace_simulator.get_trace_progress()
            log_path = log_file_prefix + '_' + all_file_names[trace_progress.trace_index]
            jsonl_path = f"{log_path}.jsonl"
            with open(log_path, 'w'):
                pass
            with open(jsonl_path, 'w'):
                pass


def calculate_test_statistics(log_file_prefix: str) -> Dict[str, float]:
    """Calculate statistics from test log files.

    Reference:
        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L42-L55

    Args:
        log_file_prefix: Prefix for log file paths (same as used in testing()).

    Returns:
        Dictionary with statistics: min, 5th percentile, mean, median,
        95th percentile, and max rewards.
    """
    rewards = []
    test_log_files = [
        test_log_file
        for test_log_file in glob.glob(log_file_prefix + "*")
        if not test_log_file.endswith(".jsonl")
    ]
    for test_log_file in test_log_files:
        reward = []
        with open(test_log_file, 'rb') as f:
            for line in f:
                parse = line.split()
                try:
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(np.mean(reward[1:]))

    rewards = np.array(rewards)

    return {
        'rewards_min': np.min(rewards),
        'rewards_5per': np.percentile(rewards, 5),
        'rewards_mean': np.mean(rewards),
        'rewards_median': np.percentile(rewards, 50),
        'rewards_95per': np.percentile(rewards, 95),
        'rewards_max': np.max(rewards),
    }


def add_testing_arguments(parser: argparse.ArgumentParser) -> None:
    """Add testing-specific arguments to parser.

    This function adds the --test-log-file-prefix argument needed for testing.
    It should be called along with add_env_agent_arguments to get all
    arguments required by the main() function.
    """
    parser.add_argument('--test-log-file-prefix', type=str, default=LOG_FILE_PREFIX,
                        dest='test_log_file_prefix',
                        help=f"Prefix for test log files (default: {LOG_FILE_PREFIX}). "
                             f"Actual log path: <prefix><agent_name>_<trace_name>")


def run_evaluation(args: argparse.Namespace) -> str:
    """Run evaluation from parsed arguments and return the test log prefix."""
    # Append agent name to log file prefix
    log_file_prefix = args.test_log_file_prefix + args.agent_name

    # Prepare env and agent
    env, agent = prepare_testing(
        name=args.agent_name,
        trace_folder=args.test_trace_folder,
        random_seed=args.random_seed,
        observer_kwargs=args.observer_options,
        player_kwargs=args.player_options,
        model_path=args.model_path,
        agent_kwargs=args.agent_options,
    )

    # Run testing
    testing(
        env=env,
        agent=agent,
        log_file_prefix=log_file_prefix,
    )

    return log_file_prefix


DESCRIPTION = 'Test Pensieve agent'


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add command-line arguments for testing."""
    prepare_registry_package(parser)
    add_env_agent_arguments(parser, available_agents=get_available_agents())
    add_testing_arguments(parser)


def main(args: argparse.Namespace) -> None:
    """Run testing from parsed command-line arguments and print statistics."""
    # Post-process arguments (parse options, set seed)
    parse_env_agent_args(args)

    # Run testing
    log_file_prefix = run_evaluation(args)

    # Calculate and print test statistics
    stats = calculate_test_statistics(log_file_prefix)
    print("\n" + "=" * 50)
    print("Test Statistics")
    print("=" * 50)
    print(f"Reward Min:     {stats['rewards_min']:.4f}")
    print(f"Reward 5%:      {stats['rewards_5per']:.4f}")
    print(f"Reward Mean:    {stats['rewards_mean']:.4f}")
    print(f"Reward Median:  {stats['rewards_median']:.4f}")
    print(f"Reward 95%:     {stats['rewards_95per']:.4f}")
    print(f"Reward Max:     {stats['rewards_max']:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    add_arguments(parser)
    main(parser.parse_args())
