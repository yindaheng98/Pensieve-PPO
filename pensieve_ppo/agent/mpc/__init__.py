"""MPC (Model Predictive Control) algorithm implementation.

This module provides the MPC agents for adaptive bitrate streaming:
- MPCAgent: Uses RobustMPC with harmonic mean bandwidth prediction
- OracleMPCAgent: Uses oracle future bandwidth information

Reference:
    https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc.py
    https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc_future_bandwidth.py
"""

from .agent import MPCAgent
from .agent_oracle import OracleMPCAgent
from .observer import MPCABRStateObserver, MPCState
from .observer_oracle import OracleMPCABRStateObserver, OracleMPCState
from .env import create_mpc_env, create_oracle_mpc_env
from ..registry import register_agent, register_env

# Register MPC agents
register_agent("mpc", MPCAgent)
register_agent("mpc-oracle", OracleMPCAgent)

# Register MPC environments
register_env("mpc", create_mpc_env)
register_env("mpc-oracle", create_oracle_mpc_env)

__all__ = [
    'MPCAgent',
    'OracleMPCAgent',
    'MPCABRStateObserver',
    'MPCState',
    'OracleMPCABRStateObserver',
    'OracleMPCState',
    'create_mpc_env',
    'create_oracle_mpc_env',
]
