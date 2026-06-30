"""Utility helpers for RL ABR agents."""

from ...core.simulator import StepResult
from ...gym.env import ABREnv
from ...quality_ladder import QualityLadderRequest, QualityLadderVideoPlayer


def get_video_player(env: ABREnv) -> QualityLadderVideoPlayer:
    """Get the bound quality-ladder player or fail fast."""
    video_player = env.simulator.video_player
    if not isinstance(video_player, QualityLadderVideoPlayer):
        raise TypeError(
            "RL observers require QualityLadderVideoPlayer, "
            f"got {type(video_player).__name__}"
        )
    return video_player


def get_bitrate_levels(env: ABREnv) -> int:
    """Get the number of available bitrate levels."""
    return get_video_player(env).bitrate_levels


def get_chunk_idx(env: ABREnv, result: StepResult) -> int:
    """Resolve the chunk index downloaded for a StepResult."""
    video_player = get_video_player(env)
    return video_player.total_chunks - result.video_chunk_remain - 1


def get_last_chunk_idx(env: ABREnv, result: StepResult) -> int:
    """Resolve the chunk index before the downloaded chunk."""
    return max(get_chunk_idx(env, result) - 1, 0)


def get_next_chunk_idx(env: ABREnv, result: StepResult) -> int:
    """Resolve the next chunk index after a StepResult."""
    if result.end_of_video:
        return 0
    video_player = get_video_player(env)
    return video_player.total_chunks - result.video_chunk_remain


def get_initial_chunk_quality(env: ABREnv, bitrate_level: int) -> float:
    """Get one bitrate level's quality for the current initial chunk."""
    video_player = get_video_player(env)
    return video_player.get_chunk_quality(
        QualityLadderRequest(bitrate_level),
        video_player.video_chunk_counter,
    )


def get_initial_chunk_qualities(env: ABREnv) -> list[float]:
    """Get all bitrate levels' qualities for the current initial chunk."""
    video_player = get_video_player(env)
    return video_player.get_chunk_qualities(video_player.video_chunk_counter)


def get_chunk_qualities(env: ABREnv, result: StepResult) -> list[float]:
    """Get all bitrate levels' qualities for a chunk."""
    video_player = get_video_player(env)
    return video_player.get_chunk_qualities(get_chunk_idx(env, result))


def get_last_chunk_qualities(env: ABREnv, result: StepResult) -> list[float]:
    """Get all bitrate levels' qualities for the chunk before the downloaded one."""
    video_player = get_video_player(env)
    return video_player.get_chunk_qualities(get_last_chunk_idx(env, result))


def get_next_chunk_qualities(env: ABREnv, result: StepResult) -> list[float]:
    """Get all bitrate levels' qualities for the next chunk."""
    video_player = get_video_player(env)
    chunk_idx = get_next_chunk_idx(env, result)
    return video_player.get_chunk_qualities(chunk_idx)


def get_next_chunk_sizes(env: ABREnv, result: StepResult) -> list[int]:
    """Get all bitrate levels' sizes for the next chunk."""
    video_player = get_video_player(env)
    chunk_idx = get_next_chunk_idx(env, result)
    return video_player.get_chunk_sizes(chunk_idx)
