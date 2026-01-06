"""Trace data loading module."""

from .abc import TraceData
from .loader import load_trace

__all__ = ['load_trace', 'TraceData']
