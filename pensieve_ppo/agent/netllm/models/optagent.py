"""OPT-based NetLLMAgent implementation.

Reference:
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/config.py
"""

from typing import Literal

import torch

from ..container import NetLLMAgent
from .opt import OPTModel


# PLM embed sizes from NetLLM config
# https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/config.py#L35-L64
_OPT_EMBED_SIZES = {
    'large': 5120,
    'base': 4096,
    'small': 2560,
    'xs': 2048,
    'xxs': 512,
}


class OPTNetLLMAgent(NetLLMAgent):
    """NetLLMAgent with OPT as the PLM backbone.

    This class uses fixed default parameters from the NetLLM reference implementation.

    Reference:
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py

    Example:
        >>> agent = OPTNetLLMAgent(
        ...     action_dim=6,
        ...     min_reward=-10.0,
        ...     max_reward=10.0,
        ...     pretrained_path='path/to/opt/base',
        ... )
    """

    def __init__(
        self,
        *args,
        pretrained_path: str,
        rank: int = -1,
        plm_size: Literal['large', 'base', 'small', 'xs', 'xxs'] = 'base',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs,
    ):
        """Initialize the OPTNetLLMAgent.

        Args:
            *args: Positional arguments passed to NetLLMAgent
                (action_dim, min_reward, max_reward).
            pretrained_path: Path to the pretrained OPT model.
            rank: Must be -1 for full fine-tuning (no LoRA).
            plm_size: Size variant of OPT ('large', 'base', 'small', 'xs', 'xxs').
                Used to determine plm_embed_size.
            device: Device to run the model on ('cuda' or 'cpu').
            **kwargs: Additional keyword arguments passed to NetLLMAgent.
        """
        assert rank == -1, f"OPTNetLLMAgent requires rank=-1 (full fine-tuning), got {rank}"

        # Load OPT model
        plm = OPTModel.from_pretrained(pretrained_path).to(device)
        plm_embed_size = _OPT_EMBED_SIZES[plm_size]

        # Initialize parent with rank=-1 (full fine-tuning, no LoRA)
        super().__init__(
            *args,
            plm=plm,
            plm_embed_size=plm_embed_size,
            rank=rank,
            device=device,
            **kwargs,
        )


class OPTLoRANetLLMAgent(NetLLMAgent):
    """NetLLMAgent with OPT as the PLM backbone, using LoRA for parameter-efficient fine-tuning.

    This class wraps the OPT model with LoRA adapters for efficient fine-tuning.

    Reference:
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py#L181-L182
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/models/low_rank.py#L29-L55

    Example:
        >>> agent = OPTLoRANetLLMAgent(
        ...     action_dim=6,
        ...     min_reward=-10.0,
        ...     max_reward=10.0,
        ...     pretrained_path='path/to/opt/base',
        ...     rank=128,
        ... )
    """

    def __init__(
        self,
        *args,
        pretrained_path: str,
        rank: int = 128,
        plm_size: Literal['large', 'base', 'small', 'xs', 'xxs'] = 'base',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs,
    ):
        """Initialize the OPTLoRANetLLMAgent.

        Args:
            *args: Positional arguments passed to NetLLMAgent
                (action_dim, min_reward, max_reward).
            pretrained_path: Path to the pretrained OPT model.
            rank: Rank of LoRA low-rank matrices. Must be > 0. Default is 128.
            plm_size: Size variant of OPT ('large', 'base', 'small', 'xs', 'xxs').
                Used to determine plm_embed_size.
            device: Device to run the model on ('cuda' or 'cpu').
            **kwargs: Additional keyword arguments passed to NetLLMAgent.
        """
        assert rank > 0, f"OPTLoRANetLLMAgent requires rank > 0 (LoRA enabled), got {rank}"

        # Load OPT model (LoRA will be applied by NetLLMAgent based on rank)
        # Reference: https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py#L181-L182
        plm = OPTModel.from_pretrained(pretrained_path).to(device)
        plm_embed_size = _OPT_EMBED_SIZES[plm_size]

        # Initialize parent with rank parameter (NetLLMAgent handles LoRA)
        super().__init__(
            *args,
            plm=plm,
            plm_embed_size=plm_embed_size,
            rank=rank,
            device=device,
            **kwargs,
        )
