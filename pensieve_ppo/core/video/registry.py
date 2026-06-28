"""Video player registry for selecting video format implementations."""

from typing import Type

from .player import VideoPlayer


REGISTRY: dict[str, Type[VideoPlayer]] = {}


def register(name: str, video_player_cls: Type[VideoPlayer]) -> None:
    """Register a video player implementation by name.

    Args:
        name: Name to register the video player under.
        video_player_cls: Video player class to register.

    Raises:
        ValueError: If the class is not a VideoPlayer subclass.
    """
    if not issubclass(video_player_cls, VideoPlayer):
        raise ValueError(
            f"Video player class must be a subclass of VideoPlayer, got {video_player_cls}"
        )
    REGISTRY[name] = video_player_cls


def get_available_video_players() -> list[str]:
    """Get a list of available video player names."""
    return list(REGISTRY.keys())


def create_video_player(name: str, *args, **kwargs) -> VideoPlayer:
    """Create a video player by registered name.

    Args:
        name: Registered video player name.
        *args: Positional arguments passed to the video player constructor.
        **kwargs: Keyword arguments passed to the video player constructor.

    Returns:
        VideoPlayer instance.

    Raises:
        ValueError: If the video player name is not registered.
    """
    if name not in REGISTRY:
        available = ", ".join(get_available_video_players())
        raise ValueError(
            f"Unknown video player: '{name}'. Available video players: {available}"
        )
    return REGISTRY[name](*args, **kwargs)
