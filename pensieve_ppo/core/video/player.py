"""Video player for tracking playback position and chunk information."""

from typing import List, Tuple

from .data import VideoData


class VideoPlayer:
    """Tracks video playback position and provides chunk information.

    This class manages the video chunk counter and provides methods
    to get chunk sizes and advance through the video.
    """

    def __init__(self, video_data: VideoData):
        """Initialize the video player.

        Args:
            video_data: Loaded video chunk size data
        """
        self._video_data = video_data
        self.reset()

    def reset(self) -> None:
        """Reset the video playback position to the beginning.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L138
        """
        self.video_chunk_counter = 0

    def get_chunk_size(self, quality: int) -> int:
        """Get the size of current chunk at given quality level.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L54

        Args:
            quality: Bitrate level (0 to num_bitrates-1)

        Returns:
            Chunk size in bytes
        """
        return self._video_data.get_chunk_size(quality, self.video_chunk_counter)

    def get_next_chunk_sizes(self) -> List[int]:
        """Get sizes of next chunk at all quality levels.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L152-L154

        Returns:
            List of chunk sizes in bytes for each bitrate level
        """
        return self._video_data.get_next_chunk_sizes(self.video_chunk_counter)

    def advance(self) -> Tuple[bool, int]:
        """Advance to the next video chunk.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L131

        Returns:
            Tuple of (end_of_video, remaining_chunks)
        """
        self.video_chunk_counter += 1
        video_chunk_remain = self._video_data.num_chunks - self.video_chunk_counter

        end_of_video = self.video_chunk_counter >= self._video_data.num_chunks

        return end_of_video, video_chunk_remain

    @property
    def num_bitrates(self) -> int:
        """Number of available bitrate levels."""
        return self._video_data.bitrate_levels

    @property
    def num_chunks(self) -> int:
        """Total number of video chunks."""
        return self._video_data.num_chunks
