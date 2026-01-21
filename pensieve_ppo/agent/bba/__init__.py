"""BBA (Buffer-Based Adaptive) algorithm implementation.

This module provides the BBA agent for buffer-based adaptive bitrate streaming.

Reference:
    https://github.com/hongzimao/Genet/blob/main/src/simulator/abr_simulator/bba.py
"""

from .agent import BBAAgent
from .observer import BBAStateObserver, BBAState
from .env import create_bba_env
from ..registry import register_agent, register_env

# Register BBA agent
register_agent("bba", BBAAgent)

# Register BBA environment
register_env("bba", create_bba_env)

__all__ = [
    'BBAAgent',
    'BBAStateObserver',
    'BBAState',
    'create_bba_env',
]
