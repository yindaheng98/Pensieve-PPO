"""BBA (Buffer-Based Adaptive) algorithm implementation.

This module provides the BBA agent for buffer-based adaptive bitrate streaming.

Reference:
    https://github.com/hongzimao/Genet/blob/main/src/simulator/abr_simulator/bba.py
"""

from .agent import BBAAgent
from .observer import BBAStateObserver, BBAState
from ..player import QualityLadderVideoPlayer
from ...agent.registry import register

# Register BBA agent
register("bba", BBAAgent, BBAStateObserver, QualityLadderVideoPlayer)

__all__ = [
    'BBAAgent',
    'BBAStateObserver',
    'BBAState',
]
