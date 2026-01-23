"""NetLLM agent module.

This module provides classes for NetLLM-based ABR agents that use
pre-trained language models for offline reinforcement learning.

Components:
- NetLLMABRStateObserver: State observer for collecting raw data
- NetLLMState: State container with raw data for training and inference
- AbstractNetLLMAgent: Abstract base class defining agent interface
- NetLLMTrainingBatch: Training batch with processed tensors

Pre-defined agent combinations with specific PLM backbones:
- GPT2NetLLMAgent: NetLLMAgent with GPT2 backbone
- LlamaNetLLMAgent: NetLLMAgent with Llama backbone
- MistralNetLLMAgent: NetLLMAgent with Mistral backbone
- OPTNetLLMAgent: NetLLMAgent with OPT backbone
- T5NetLLMAgent: NetLLMAgent with T5 backbone

The module implements the separation between:
- Raw data collection: NetLLMABRStateObserver (observer.py)
- Data processing: AbstractNetLLMAgent.produce_training_batch() (abc.py)
- Model interface: AbstractNetLLMAgent.forward() and sample() (abc.py)
- Training: AbstractNetLLMAgent.train_batch() (abc.py)
- Inference: AbstractNetLLMAgent.select_action() (abc.py)

Reference:
    https://github.com/duowuyms/NetLLM/tree/main/adaptive_bitrate_streaming/plm_special
"""

from .observer import (
    NetLLMABRStateObserver,
    NetLLMState,
)

from .abc import (
    AbstractNetLLMAgent,
    NetLLMTrainingBatch,
)

from . import models  # noqa: F401

__all__ = [
    'NetLLMABRStateObserver',
    'NetLLMState',
    'AbstractNetLLMAgent',
    'NetLLMTrainingBatch',
    'models',
]
