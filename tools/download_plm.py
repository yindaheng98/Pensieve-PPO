#!/usr/bin/env python
"""Download pre-trained language models for NetLLM agents.

Usage:
    python tools/download_plm.py                          # Download llama base
    python tools/download_plm.py --model gpt2             # Download gpt2 base
    python tools/download_plm.py --model gpt2 --size small
    python tools/download_plm.py --token YOUR_HF_TOKEN    # With authentication

    # Use mirror (set HF_ENDPOINT environment variable):
    # Windows: set HF_ENDPOINT=https://hf-mirror.com && python tools/download_plm.py
    # Linux:   HF_ENDPOINT=https://hf-mirror.com python tools/download_plm.py
"""

import argparse
from pathlib import Path

from transformers import AutoConfig, AutoModel, AutoTokenizer

# Model configurations: local_name -> {size -> HuggingFace ID}
MODEL_CONFIGS = {
    "gpt2": {
        "small": "openai-community/gpt2",
        "base": "openai-community/gpt2-medium",
        "large": "openai-community/gpt2-large",
        "xl": "openai-community/gpt2-xl",
    },
    "llama": {
        "base": "meta-llama/Llama-2-7b-hf",
    },
    "mistral": {
        "base": "mistralai/Mistral-7B-v0.1",
    },
    "opt": {
        "xxs": "facebook/opt-125m",
        "xs": "facebook/opt-350m",
        "small": "facebook/opt-1.3b",
        "base": "facebook/opt-2.7b",
        "large": "facebook/opt-6.7b",
    },
    "t5-lm": {
        "small": "google/t5-v1_1-small",
        "base": "google/t5-v1_1-base",
        "large": "google/t5-v1_1-large",
        "xl": "google/t5-v1_1-xl",
    },
}


def download_model(model_type: str, size: str, output_dir: Path, token: str = None):
    """Download a model from HuggingFace."""
    hf_model_id = MODEL_CONFIGS[model_type][size]
    save_path = output_dir / model_type / size

    print(f"Downloading {model_type}/{size} from {hf_model_id}")
    print(f"Save path: {save_path}")

    save_path.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(hf_model_id, token=token)
    config.save_pretrained(save_path)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, token=token)
    tokenizer.save_pretrained(save_path)

    model = AutoModel.from_pretrained(hf_model_id, token=token)
    model.save_pretrained(save_path)

    print(f"Done: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Download PLM models for NetLLM")
    parser.add_argument("--model", type=str, default="llama")
    parser.add_argument("--size", type=str, default="base")
    parser.add_argument("--output-dir", type=str, default="downloaded_plms")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace access token")
    args = parser.parse_args()

    download_model(args.model, args.size, Path(args.output_dir), args.token)


if __name__ == "__main__":
    main()
