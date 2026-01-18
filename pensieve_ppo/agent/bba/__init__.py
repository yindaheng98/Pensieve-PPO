"""BBA (Buffer-Based Adaptive) algorithm implementation.

This module provides the BBA agent for buffer-based adaptive bitrate streaming.

Reference:
    https://github.com/hongzimao/Genet/blob/main/src/simulator/abr_simulator/bba.py
"""

from .agent import BBAAgent
from .observer import BBAStateObserver
from ..registry import register_agent

# Register BBA agent
register_agent("bba", BBAAgent)

__all__ = [
    'BBAAgent',
    'BBAStateObserver',
]
