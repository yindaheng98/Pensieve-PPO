"""Network trace data classes."""

from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class TraceData:
    """Container for loaded trace data."""
    all_cooked_time: List[List[float]]
    all_cooked_bw: List[List[float]]
    all_file_names: List[str]
