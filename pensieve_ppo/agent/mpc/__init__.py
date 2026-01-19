"""MPC (Model Predictive Control) algorithm implementation.

This module provides the MPC agent for adaptive bitrate streaming using
future bandwidth prediction.

Reference:
    https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc_future_bandwidth.py
"""

from .agent import MPCAgent
from .observer import MPCABRStateObserver, PredictionState
from .env import create_mpc_env
from ..registry import register_agent, register_env

# Register MPC agent
register_agent("mpc", MPCAgent)

# Register MPC environment
register_env("mpc", create_mpc_env)

__all__ = [
    'MPCAgent',
    'MPCABRStateObserver',
    'PredictionState',
    'create_mpc_env',
]
