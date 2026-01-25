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

Available LoRA agents (parameter-efficient fine-tuning with LoRA):
- GPT2LoRANetLLMAgent: NetLLMAgent with GPT2 backbone + LoRA
- LlamaLoRANetLLMAgent: NetLLMAgent with Llama backbone + LoRA
- MistralLoRANetLLMAgent: NetLLMAgent with Mistral backbone + LoRA
- OPTLoRANetLLMAgent: NetLLMAgent with OPT backbone + LoRA
- T5LoRANetLLMAgent: NetLLMAgent with T5 backbone + LoRA

Reference:
    https://github.com/duowuyms/NetLLM/tree/main/adaptive_bitrate_streaming/plm_special/models
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/models/low_rank.py
"""

from .gpt2 import GPT2Model
from .llama import LlamaModel
from .mistral import MistralModel
from .opt import OPTModel
from .t5 import T5Model
from .rl_policy import OfflineRLPolicy
from .state_encoder import EncoderNetwork

from .gpt2agent import GPT2NetLLMAgent, GPT2LoRANetLLMAgent
from .llamaagent import LlamaNetLLMAgent, LlamaLoRANetLLMAgent
from .mistralagent import MistralNetLLMAgent, MistralLoRANetLLMAgent
from .optagent import OPTNetLLMAgent, OPTLoRANetLLMAgent
from .t5agent import T5NetLLMAgent, T5LoRANetLLMAgent

# Register NetLLM agents
from ...registry import register
from ..observer import NetLLMABRStateObserver

register("netllm-gpt2", GPT2NetLLMAgent, NetLLMABRStateObserver)
register("netllm-llama", LlamaNetLLMAgent, NetLLMABRStateObserver)
register("netllm-mistral", MistralNetLLMAgent, NetLLMABRStateObserver)
register("netllm-opt", OPTNetLLMAgent, NetLLMABRStateObserver)
register("netllm-t5", T5NetLLMAgent, NetLLMABRStateObserver)

# Register LoRA NetLLM agents
# Reference: https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py#L181-L182
register("netllm-gpt2-lora", GPT2LoRANetLLMAgent, NetLLMABRStateObserver)
register("netllm-llama-lora", LlamaLoRANetLLMAgent, NetLLMABRStateObserver)
register("netllm-mistral-lora", MistralLoRANetLLMAgent, NetLLMABRStateObserver)
register("netllm-opt-lora", OPTLoRANetLLMAgent, NetLLMABRStateObserver)
register("netllm-t5-lora", T5LoRANetLLMAgent, NetLLMABRStateObserver)

__all__ = [
    'GPT2Model',
    'LlamaModel',
    'MistralModel',
    'OPTModel',
    'T5Model',
    'OfflineRLPolicy',
    'EncoderNetwork',
    # Standard agents
    'GPT2NetLLMAgent',
    'LlamaNetLLMAgent',
    'MistralNetLLMAgent',
    'OPTNetLLMAgent',
    'T5NetLLMAgent',
    # LoRA agents
    # Reference: https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py#L181-L182
    'GPT2LoRANetLLMAgent',
    'LlamaLoRANetLLMAgent',
    'MistralLoRANetLLMAgent',
    'OPTLoRANetLLMAgent',
    'T5LoRANetLLMAgent',
]
