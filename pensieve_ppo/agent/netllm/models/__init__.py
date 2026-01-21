"""Pre-trained Language Model implementations for NetLLM.

This module provides customized PLM implementations based on HuggingFace
Transformers models. The key modifications are:
1. Positional embeddings are removed (handled by the policy)
2. Early stopping support for efficient inference

Available models:
- GPT2Model: Based on GPT-2
- LLaMAModel: Based on LLaMA
- MistralModel: Based on Mistral
- OPTModel: Based on OPT
- T5Model: Based on T5

Reference:
    https://github.com/duowuyms/NetLLM/tree/main/adaptive_bitrate_streaming/plm_special/models
"""

from .gpt2 import GPT2Model
from .llama import LlamaModel
from .mistral import MistralModel
from .opt import OPTModel
from .t5 import T5Model

__all__ = [
    'GPT2Model',
    'LlamaModel',
    'MistralModel',
    'OPTModel',
    'T5Model'
]
