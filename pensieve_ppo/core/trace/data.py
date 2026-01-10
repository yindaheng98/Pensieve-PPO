"""Network trace data classes."""

from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class TraceData:
    """Container for loaded trace data."""
    all_cooked_time: List[List[float]]
    all_cooked_bw: List[List[float]]
    all_file_names: List[str]

    def __len__(self) -> int:
        return len(self.all_file_names)

    def __getitem__(self, idx: int) -> Tuple[List[float], List[float], str]:
        return (
            self.all_cooked_time[idx],
            self.all_cooked_bw[idx],
            self.all_file_names[idx]
        )
