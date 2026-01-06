"""Video data loading module."""

from .abc import VideoData
from .loader import load_video_size

__all__ = ['load_video_size', 'VideoData']
