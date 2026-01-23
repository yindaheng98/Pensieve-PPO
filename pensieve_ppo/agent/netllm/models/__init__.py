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

Available agents (pre-configured with specific PLM backbones):
- GPT2NetLLMAgent: NetLLMAgent with GPT2 backbone
- LlamaNetLLMAgent: NetLLMAgent with Llama backbone
- MistralNetLLMAgent: NetLLMAgent with Mistral backbone
- OPTNetLLMAgent: NetLLMAgent with OPT backbone
- T5NetLLMAgent: NetLLMAgent with T5 backbone

Reference:
    https://github.com/duowuyms/NetLLM/tree/main/adaptive_bitrate_streaming/plm_special/models
"""

from .gpt2 import GPT2Model
from .llama import LlamaModel
from .mistral import MistralModel
from .opt import OPTModel
from .t5 import T5Model
from .rl_policy import OfflineRLPolicy
from .state_encoder import EncoderNetwork

from .gpt2agent import GPT2NetLLMAgent
from .llamaagent import LlamaNetLLMAgent
from .mistralagent import MistralNetLLMAgent
from .optagent import OPTNetLLMAgent
from .t5agent import T5NetLLMAgent

# Register NetLLM agents
from ...registry import register
from ..observer import NetLLMABRStateObserver

register("netllm-gpt2", GPT2NetLLMAgent, NetLLMABRStateObserver)
register("netllm-llama", LlamaNetLLMAgent, NetLLMABRStateObserver)
register("netllm-mistral", MistralNetLLMAgent, NetLLMABRStateObserver)
register("netllm-opt", OPTNetLLMAgent, NetLLMABRStateObserver)
register("netllm-t5", T5NetLLMAgent, NetLLMABRStateObserver)

__all__ = [
    'GPT2Model',
    'LlamaModel',
    'MistralModel',
    'OPTModel',
    'T5Model',
    'OfflineRLPolicy',
    'EncoderNetwork',
    'GPT2NetLLMAgent',
    'LlamaNetLLMAgent',
    'MistralNetLLMAgent',
    'OPTNetLLMAgent',
    'T5NetLLMAgent',
]
