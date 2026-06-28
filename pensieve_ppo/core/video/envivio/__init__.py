"""Envivio video data loading module."""

from .player import EnvivioVideoChunkRequest, EnvivioVideoPlayer
from ..registry import register


register('envivio', EnvivioVideoPlayer)

__all__ = [
    'EnvivioVideoChunkRequest',
    'EnvivioVideoPlayer',
]
