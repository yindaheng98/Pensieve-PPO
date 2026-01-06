"""Network trace data loader."""

import os
from .abc import TraceData


def load_trace(trace_folder: str) -> TraceData:
    """
    Load network bandwidth traces from a folder.

    Each trace file should contain two columns:
    - Column 1: timestamp (seconds)
    - Column 2: bandwidth (Mbps)

    Args:
        trace_folder: Path to folder containing trace files

    Returns:
        TraceData object containing all loaded traces
    """
    cooked_files = sorted(os.listdir(trace_folder))
    if not cooked_files:
        raise ValueError(f"No valid trace files found in {trace_folder}")
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for trace_file in cooked_files:
        file_path = os.path.join(trace_folder, trace_file)
        cooked_time = []
        cooked_bw = []
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(trace_file)
    return TraceData(
        all_cooked_time=all_cooked_time,
        all_cooked_bw=all_cooked_bw,
        all_file_names=all_file_names
    )
