"""Convenience functions for creating combined TraceSimulator configurations."""

from typing import Optional

from ..simulator import TraceSimulator
from ..abc import TraceData
from .noise import NoiseTraceSimulator, NOISE_LOW, NOISE_HIGH
from .random import RandomTraceSimulator


def create_train_simulator(
    trace_data: TraceData,
    noise_low: float = NOISE_LOW,
    noise_high: float = NOISE_HIGH,
    random_seed: Optional[int] = None,
) -> TraceSimulator:
    """Create a TraceSimulator configured for training.

    Applies both noise and random trace selection:
    - Multiplicative noise on download delay
    - Random trace selection on reset and video end
    - Random start point within each trace

    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py

    Args:
        trace_data: Loaded network trace data
        noise_low: Lower bound of noise multiplier (default: 0.9)
        noise_high: Upper bound of noise multiplier (default: 1.1)
        random_seed: Random seed for reproducibility

    Returns:
        TraceSimulator with noise and random trace selection
    """
    simulator = TraceSimulator(trace_data)
    simulator = NoiseTraceSimulator(
        simulator,
        noise_low=noise_low,
        noise_high=noise_high,
        random_seed=random_seed,
    )
    simulator = RandomTraceSimulator(
        simulator,
        random_seed=random_seed,
    )
    return simulator


def create_test_simulator(
    trace_data: TraceData,
) -> TraceSimulator:
    """Create a TraceSimulator configured for testing.

    No noise, sequential trace iteration.

    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py

    Args:
        trace_data: Loaded network trace data

    Returns:
        Base TraceSimulator without wrappers
    """
    return TraceSimulator(trace_data)
