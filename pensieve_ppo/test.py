"""Testing script for Pensieve PPO.

This module implements the complete testing pipeline, evaluating trained models
on test traces and logging detailed results.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py
"""

import argparse
import os
from typing import Dict, Tuple

import numpy as np

from .agent.abc import AbstractAgent
from .defaults import create_env_agent_with_default
from .gym.env import ABREnv, M_IN_K
from .args import add_env_agent_arguments, parse_env_agent_args

# Reference: https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L22
TEST_LOG_FOLDER = './test_results/'

# Default log file prefix (agent name will be appended)
# Reference: https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L25
LOG_FILE_PREFIX = os.path.join(TEST_LOG_FOLDER, 'log_sim_')


def prepare_testing(*args, **kwargs) -> Tuple[ABREnv, AbstractAgent]:
    """Prepare env and agent for testing.

    Wrapper for create_env_agent_with_default with train=False.
    See create_env_agent_with_default for available parameters.
    """
    return create_env_agent_with_default(*args, train=False, **kwargs)


def testing(
    env: ABREnv,
    agent: AbstractAgent,
    log_file_prefix: str,
    initial_level: int,
) -> None:
    """Run testing on all test traces.

    Reference:
        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L30-L151

    The loop structure matches src/test.py exactly:
        while True:
            1. step(action) - download chunk
            2. write log (using previous entropy)
            3. predict (update entropy for next log)
            4. if end_of_video: reset and open new log

    Args:
        env: The ABR environment.
        agent: The trained agent.
        log_file_prefix: Prefix for log file paths.
        initial_level: Initial quality level index on reset.
    """
    # Create log directory if needed
    log_dir = os.path.dirname(log_file_prefix)
    os.makedirs(log_dir, exist_ok=True)

    s_info, s_len = agent.s_dim
    a_dim = agent.a_dim

    # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L36
    trace_progress = env.simulator.trace_simulator.get_trace_progress()
    all_file_names = trace_progress.all_trace_names

    # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L41-L42
    log_path = log_file_prefix + '_' + all_file_names[trace_progress.trace_index]
    log_file = open(log_path, 'w')

    # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L53
    env.reset(options={'initial_level': initial_level})

    # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L55-L66
    last_bit_rate = initial_level
    bit_rate = initial_level

    action_vec = np.zeros(a_dim)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((s_info, s_len))]
    a_batch = [action_vec]
    r_batch = []
    entropy_record = []
    entropy_ = 0.5
    video_count = 0

    while True:  # serve video forever
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L69-L85
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L100-L115
        # the action is from the last decision
        # this is to make the framework similar to the real
        state, reward, end_of_video, truncated, info = env.step(bit_rate)

        r_batch.append(reward)

        last_bit_rate = bit_rate

        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L89-L98
        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write(str(info['time_stamp'] / M_IN_K) + '\t' +
                       str(info['quality']) + '\t' +
                       str(info['buffer_size']) + '\t' +
                       str(info['rebuffer']) + '\t' +
                       str(info['video_chunk_size']) + '\t' +
                       str(info['delay']) + '\t' +
                       str(entropy_) + '\t' +
                       str(reward) + '\n')
        log_file.flush()

        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L117-L123
        action_prob = agent.predict(np.reshape(state, (1, s_info, s_len)))
        noise = np.random.gumbel(size=len(action_prob))
        bit_rate = np.argmax(np.log(action_prob) + noise)

        s_batch.append(state)
        entropy_ = -np.dot(action_prob, np.log(action_prob))
        entropy_record.append(entropy_)

        if end_of_video:
            # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L126-L147
            log_file.write('\n')
            log_file.close()

            last_bit_rate = initial_level
            bit_rate = initial_level  # use the default action here

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            action_vec = np.zeros(a_dim)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((s_info, s_len)))
            a_batch.append(action_vec)
            # print(np.mean(entropy_record))
            entropy_record = []

            video_count += 1

            if video_count >= len(all_file_names):
                break

            # Reset for next trace - only initializes state
            env.reset(options={'reset_time_stamp': False, 'initial_level': initial_level})

            # Open new log file
            # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L149-L150
            trace_progress = env.simulator.trace_simulator.get_trace_progress()
            log_path = log_file_prefix + '_' + all_file_names[trace_progress.trace_index]
            log_file = open(log_path, 'w')


def calculate_test_statistics(log_file_prefix: str) -> Dict[str, float]:
    """Calculate statistics from test log files.

    Reference:
        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L42-L55

    Args:
        log_file_prefix: Prefix for log file paths (same as used in testing()).

    Returns:
        Dictionary with statistics: min, 5th percentile, mean, median, 
        95th percentile, max rewards, and avg entropy.
    """
    log_folder = os.path.dirname(log_file_prefix)
    rewards, entropies = [], []
    test_log_files = os.listdir(log_folder)
    for test_log_file in test_log_files:
        reward, entropy = [], []
        with open(os.path.join(log_folder, test_log_file), 'rb') as f:
            for line in f:
                parse = line.split()
                try:
                    entropy.append(float(parse[-2]))
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(np.mean(reward[1:]))
        entropies.append(np.mean(entropy[1:]))

    rewards = np.array(rewards)

    return {
        'rewards_min': np.min(rewards),
        'rewards_5per': np.percentile(rewards, 5),
        'rewards_mean': np.mean(rewards),
        'rewards_median': np.percentile(rewards, 50),
        'rewards_95per': np.percentile(rewards, 95),
        'rewards_max': np.max(rewards),
        'avg_entropy': np.mean(entropies),
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


def main(args):
    # Append agent name to log file prefix
    log_file_prefix = args.test_log_file_prefix + args.agent_name

    # Prepare env and agent
    env, agent = prepare_testing(
        trace_folder=args.test_trace_folder,
        model_path=args.model_path,
        agent_name=args.agent_name,
        device=args.device,
        levels_quality=args.levels_quality,
        state_history_len=args.state_history_len,
        agent_options=args.agent_options,
        env_options=args.env_options,
    )

    # Run testing
    testing(
        env=env,
        agent=agent,
        log_file_prefix=log_file_prefix,
        initial_level=args.initial_level,
    )

    return log_file_prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Pensieve PPO agent')
    add_env_agent_arguments(parser)
    add_testing_arguments(parser)
    args = parser.parse_args()

    # Post-process arguments (parse options, set seed)
    parse_env_agent_args(args)

    # Run testing
    log_file_prefix = main(args)

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
    print(f"Avg Entropy:    {stats['avg_entropy']:.4f}")
    print("=" * 50)
