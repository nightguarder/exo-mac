from exo.inference.shard import Shard
from typing import Optional, List
import os

model_cards = {
  ### llama
  "llama-3.3-70b": {
    "layers": 80,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.3-70B-Instruct-4bit",
       "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.3-70B-Instruct",
    },
  },
  "llama-3.2-1b": {
    "layers": 16,
    "repo": {
      "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.2-1B-Instruct-4bit",
      "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.2-1B-Instruct",
    },
  },
  "llama-3.2-1b-8bit": {
    "layers": 16,
    "repo": {
      "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.2-1B-Instruct-8bit",
      "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.2-1B-Instruct",
    },
  },
  "llama-3.2-3b": {
    "layers": 28,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.2-3B-Instruct-4bit",
       "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.2-3B-Instruct",
    },
  },
  "llama-3.2-3b-8bit": {
    "layers": 28,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.2-3B-Instruct-8bit",
       "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.2-3B-Instruct",
    },
  },
  "llama-3.2-3b-bf16": {
    "layers": 28,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.2-3B-Instruct",
       "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.2-3B-Instruct",
    },
  },
  "llama-3.1-8b": {
    "layers": 32,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
       "TinygradDynamicShardInferenceEngine": "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
    },
  },
  "llama-3.1-70b": {
    "layers": 80,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
       "TinygradDynamicShardInferenceEngine": "NousResearch/Meta-Llama-3.1-70B-Instruct",
    },
  },
  "llama-3.1-70b-bf16": {
    "layers": 80,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED",
       "TinygradDynamicShardInferenceEngine": "NousResearch/Meta-Llama-3.1-70B-Instruct",
    },
  },
  "llama-3-8b": {
    "layers": 32,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
       "TinygradDynamicShardInferenceEngine": "TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R",
    },
  },
  "llama-3-70b": {
    "layers": 80,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3-70B-Instruct-4bit",
       "TinygradDynamicShardInferenceEngine": "TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-70B-R",
    },
  },
  "llama-3.1-405b": { "layers": 126, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3.1-405B-4bit", }, },
  "llama-3.1-405b-8bit": { "layers": 126, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3.1-405B-Instruct-8bit", }, },
  ### mistral
  "mistral-nemo": { "layers": 40, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Mistral-Nemo-Instruct-2407-4bit", }, },
  "mistral-large": { "layers": 88, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Mistral-Large-Instruct-2407-4bit", }, },
  ### deepseek
  "deepseek-coder-v2-lite": { "layers": 27, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx", }, },
  "deepseek-coder-v2.5": { "layers": 60, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-V2.5-MLX-AQ4_1_64", }, },
  "deepseek-v3": { "layers": 61, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-V3-4bit", }, },
  "deepseek-v3-3bit": { "layers": 61, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-V3-3bit", }, },
  "deepseek-r1": { "layers": 61, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-4bit", }, },
  "deepseek-r1-3bit": { "layers": 61, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-3bit", }, },
  ### deepseek distills
  "deepseek-r1-distill-qwen-1.5b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/deepseek-r1-distill-qwen-1.5b", }, },
  "deepseek-r1-distill-qwen-1.5b-3bit": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-3bit", }, },
  "deepseek-r1-distill-qwen-1.5b-6bit": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-6bit", }, },
  "deepseek-r1-distill-qwen-1.5b-8bit": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-8bit", }, },
  "deepseek-r1-distill-qwen-1.5b-bf16": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-bf16", }, },
  "deepseek-r1-distill-qwen-7b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit", }, },
  "deepseek-r1-distill-qwen-7b-3bit": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-3bit", }, },
  "deepseek-r1-distill-qwen-7b-6bit": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-6bit", }, },
  "deepseek-r1-distill-qwen-7b-8bit": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-8bit", }, },
  "deepseek-r1-distill-qwen-7b-bf16": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-bf16", }, },
  "deepseek-r1-distill-qwen-14b": { "layers": 48, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit", }, },
  "deepseek-r1-distill-qwen-14b-3bit": { "layers": 48, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-3bit", }, },
  "deepseek-r1-distill-qwen-14b-6bit": { "layers": 48, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-6bit", }, },
  "deepseek-r1-distill-qwen-14b-8bit": { "layers": 48, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-8bit", }, },
  "deepseek-r1-distill-qwen-14b-bf16": { "layers": 48, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-bf16", }, },
  "deepseek-r1-distill-qwen-32b": { "layers": 64, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit", }, },
  "deepseek-r1-distill-qwen-32b-3bit": { "layers": 64, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-3bit", }, },
  "deepseek-r1-distill-qwen-32b-6bit": { "layers": 64, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-6bit", }, },
  "deepseek-r1-distill-qwen-32b-8bit": { "layers": 64, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-MLX-8Bit", }, },
  "deepseek-r1-distill-qwen-32b-bf16": { "layers": 64, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-bf16", }, },
  "deepseek-r1-distill-llama-8b": { "layers": 32, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit", }, },
  "deepseek-r1-distill-llama-8b-3bit": { "layers": 32, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-8B-3bit", }, },
  "deepseek-r1-distill-llama-8b-6bit": { "layers": 32, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-8B-6bit", }, },
  "deepseek-r1-distill-llama-8b-8bit": { "layers": 32, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-8B-8bit", }, },
  "deepseek-r1-distill-llama-8b-bf16": { "layers": 32, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-8B-bf16", }, },
  "deepseek-r1-distill-llama-70b": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-70B-4bit", }, },
  "deepseek-r1-distill-llama-70b-3bit": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-70B-3bit", }, },
  "deepseek-r1-distill-llama-70b-6bit": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-70B-6bit", }, },
  "deepseek-r1-distill-llama-70b-8bit": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-70B-8bit", }, },
  ### llava
  "llava-1.5-7b-hf": { "layers": 32, "repo": { "MLXDynamicShardInferenceEngine": "llava-hf/llava-1.5-7b-hf", }, },
  ### qwen
  "qwen-2.5-0.5b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-0.5B-Instruct-4bit", }, },
  "qwen-2.5-1.5b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-1.5B-Instruct-4bit", }, },
  "qwen-2.5-coder-1.5b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit", }, },
  "qwen-2.5-3b": { "layers": 36, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-3B-Instruct-4bit", }, },
  "qwen-2.5-coder-3b": { "layers": 36, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit", }, },
  "qwen-2.5-7b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-7B-Instruct-4bit", }, },
  "qwen-2.5-coder-7b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", }, },
  "qwen-2.5-math-7b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Math-7B-Instruct-4bit", }, },
  "qwen-2.5-14b": { "layers": 48, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-14B-Instruct-4bit", }, },
  "qwen-2.5-coder-14b": { "layers": 48, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit", }, },
  "qwen-2.5-32b": { "layers": 64, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-32B-Instruct-4bit", }, },
  "qwen-2.5-coder-32b": { "layers": 64, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit", }, },
  "qwen-2.5-72b": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-72B-Instruct-4bit", }, },
  "qwen-2.5-math-72b": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Math-72B-Instruct-4bit", }, },
  ### nemotron
  "nemotron-70b": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/nvidia_Llama-3.1-Nemotron-70B-Instruct-HF_4bit", }, },
  "nemotron-70b-bf16": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.1-Nemotron-70B-Instruct-HF-bf16", }, },
  # gemma
  "gemma2-9b": { "layers": 42, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/gemma-2-9b-it-4bit", }, },
  "gemma2-27b": { "layers": 46, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/gemma-2-27b-it-4bit", }, },
  # stable diffusion
  "stable-diffusion-2-1-base": { "layers": 31, "repo": { "MLXDynamicShardInferenceEngine": "stabilityai/stable-diffusion-2-1-base" } },
  # phi
  "phi-3.5-mini": { "layers": 32, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Phi-3.5-mini-instruct-4bit", }, },
  "phi-4": { "layers": 40, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/phi-4-4bit", }, },
  # HuggingFace supported models
  "microsoft-dialoGPT-medium": {
    "layers": 24,
    "repo": {
       "HuggingFaceInferenceEngine": "microsoft/DialoGPT-medium",
    },
  },
  "huggingface-codellama-7b": {
    "layers": 32,
    "repo": {
       "HuggingFaceInferenceEngine": "codellama/CodeLlama-7b-hf",
    },
  },
  "mistral-7b-instruct": {
    "layers": 32,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
       "TinygradDynamicShardInferenceEngine": "mistralai/Mistral-7B-Instruct-v0.1",
       "HuggingFaceInferenceEngine": "mistralai/Mistral-7B-Instruct-v0.1",
    },
  },
  "huggingface-gpt2": {
    "layers": 12,
    "repo": {
       "HuggingFaceInferenceEngine": "gpt2",
    },
  },
  "huggingface-gpt2-medium": {
    "layers": 24,
    "repo": {
       "HuggingFaceInferenceEngine": "gpt2-medium",
    },
  },
  "huggingface-bloom-560m": {
    "layers": 24,
    "repo": {
       "HuggingFaceInferenceEngine": "bigscience/bloom-560m",
    },
  },
  "huggingface-phi-2": {
    "layers": 32,
    "repo": {
       "HuggingFaceInferenceEngine": "microsoft/phi-2",
    },
  },
  "huggingface-falcon-7b": {
    "layers": 32,
    "repo": {
       "HuggingFaceInferenceEngine": "tiiuae/falcon-7b-instruct",
    },
  },
  "huggingface-flan-t5-large": {
    "layers": 24,
    "repo": {
       "HuggingFaceInferenceEngine": "google/flan-t5-large",
    },
  },
  "huggingface-codegen-2b": {
    "layers": 24,
    "repo": {
       "HuggingFaceInferenceEngine": "Salesforce/codegen-2B-mono",
    },
  },
  "huggingface-distilgpt2": {
    "layers": 6,
    "repo": {
       "HuggingFaceInferenceEngine": "distilbert/distilgpt2",
    },
  },
  "huggingface-opt-1.3b": {
    "layers": 24,
    "repo": {
       "HuggingFaceInferenceEngine": "facebook/opt-1.3b",
    },
  },
  # dummy
  "dummy": { "layers": 8, "repo": { "DummyInferenceEngine": "dummy", }, },
}

pretty_name = {
  "llama-3.3-70b": "Llama 3.3 70B",
  "llama-3.2-1b": "Llama 3.2 1B",
  "llama-3.2-1b-8bit": "Llama 3.2 1B (8-bit)",
  "llama-3.2-3b": "Llama 3.2 3B",
  "llama-3.2-3b-8bit": "Llama 3.2 3B (8-bit)",
  "llama-3.2-3b-bf16": "Llama 3.2 3B (BF16)",
  "llama-3.1-8b": "Llama 3.1 8B",
  "llama-3.1-70b": "Llama 3.1 70B",
  "llama-3.1-70b-bf16": "Llama 3.1 70B (BF16)",
  "llama-3.1-405b": "Llama 3.1 405B",
  "llama-3.1-405b-8bit": "Llama 3.1 405B (8-bit)",
  "gemma2-9b": "Gemma2 9B",
  "gemma2-27b": "Gemma2 27B",
  "nemotron-70b": "Nemotron 70B",
  "nemotron-70b-bf16": "Nemotron 70B (BF16)",
  "mistral-nemo": "Mistral Nemo",
  "mistral-large": "Mistral Large",
  "deepseek-coder-v2-lite": "Deepseek Coder V2 Lite",
  "deepseek-coder-v2.5": "Deepseek Coder V2.5",
  "deepseek-v3": "Deepseek V3 (4-bit)",
  "deepseek-v3-3bit": "Deepseek V3 (3-bit)",
  "deepseek-r1": "Deepseek R1 (4-bit)",
  "deepseek-r1-3bit": "Deepseek R1 (3-bit)",
  "llava-1.5-7b-hf": "LLaVa 1.5 7B (Vision Model)",
  "qwen-2.5-0.5b": "Qwen 2.5 0.5B",
  "qwen-2.5-1.5b": "Qwen 2.5 1.5B",
  "qwen-2.5-coder-1.5b": "Qwen 2.5 Coder 1.5B",
  "qwen-2.5-3b": "Qwen 2.5 3B",
  "qwen-2.5-coder-3b": "Qwen 2.5 Coder 3B",
  "qwen-2.5-7b": "Qwen 2.5 7B",
  "qwen-2.5-coder-7b": "Qwen 2.5 Coder 7B",
  "qwen-2.5-math-7b": "Qwen 2.5 7B (Math)",
  "qwen-2.5-14b": "Qwen 2.5 14B",
  "qwen-2.5-coder-14b": "Qwen 2.5 Coder 14B",
  "qwen-2.5-32b": "Qwen 2.5 32B",
  "qwen-2.5-coder-32b": "Qwen 2.5 Coder 32B",
  "qwen-2.5-72b": "Qwen 2.5 72B",
  "qwen-2.5-math-72b": "Qwen 2.5 72B (Math)",
  "phi-3.5-mini": "Phi-3.5 Mini",
  "phi-4": "Phi-4",
  "llama-3-8b": "Llama 3 8B",
  "llama-3-70b": "Llama 3 70B",
  "stable-diffusion-2-1-base": "Stable Diffusion 2.1",
  "huggingface-gpt2": "GPT-2 (HuggingFace)",
  "huggingface-gpt2-medium": "GPT-2 Medium (HuggingFace)",
  "huggingface-bloom-560m": "BLOOM 560M (HuggingFace)",
  "huggingface-phi-2": "Phi-2 (HuggingFace)",
  "huggingface-falcon-7b": "Falcon 7B (HuggingFace)",
  "huggingface-flan-t5-large": "Flan-T5 Large (HuggingFace)",
  "huggingface-codegen-2b": "CodeGen 2B (HuggingFace)",
  "huggingface-distilgpt2": "DistilGPT-2 (HuggingFace)",
  "huggingface-opt-1.3b": "OPT 1.3B (HuggingFace)",
  "huggingface-codellama-7b": "CodeLlama 7B (HuggingFace)",
  "microsoft-dialoGPT-medium": "DialoGPT Medium (HuggingFace)",
  "mistral-7b-instruct": "Mistral 7B Instruct",
  "deepseek-r1-distill-qwen-1.5b": "DeepSeek R1 Distill Qwen 1.5B",
  "deepseek-r1-distill-qwen-1.5b-3bit": "DeepSeek R1 Distill Qwen 1.5B (3-bit)",
  "deepseek-r1-distill-qwen-1.5b-6bit": "DeepSeek R1 Distill Qwen 1.5B (6-bit)",
  "deepseek-r1-distill-qwen-1.5b-8bit": "DeepSeek R1 Distill Qwen 1.5B (8-bit)",
  "deepseek-r1-distill-qwen-1.5b-bf16": "DeepSeek R1 Distill Qwen 1.5B (BF16)",
  "deepseek-r1-distill-qwen-7b": "DeepSeek R1 Distill Qwen 7B",
  "deepseek-r1-distill-qwen-7b-3bit": "DeepSeek R1 Distill Qwen 7B (3-bit)",
  "deepseek-r1-distill-qwen-7b-6bit": "DeepSeek R1 Distill Qwen 7B (6-bit)",
  "deepseek-r1-distill-qwen-7b-8bit": "DeepSeek R1 Distill Qwen 7B (8-bit)",
  "deepseek-r1-distill-qwen-7b-bf16": "DeepSeek R1 Distill Qwen 7B (BF16)",
  "deepseek-r1-distill-qwen-14b": "DeepSeek R1 Distill Qwen 14B",
  "deepseek-r1-distill-qwen-14b-3bit": "DeepSeek R1 Distill Qwen 14B (3-bit)",
  "deepseek-r1-distill-qwen-14b-6bit": "DeepSeek R1 Distill Qwen 14B (6-bit)",
  "deepseek-r1-distill-qwen-14b-8bit": "DeepSeek R1 Distill Qwen 14B (8-bit)",
  "deepseek-r1-distill-qwen-14b-bf16": "DeepSeek R1 Distill Qwen 14B (BF16)",
  "deepseek-r1-distill-qwen-32b": "DeepSeek R1 Distill Qwen 32B",
  "deepseek-r1-distill-qwen-32b-3bit": "DeepSeek R1 Distill Qwen 32B (3-bit)",
  "deepseek-r1-distill-qwen-32b-8bit": "DeepSeek R1 Distill Qwen 32B (8-bit)",
  "deepseek-r1-distill-qwen-32b-bf16": "DeepSeek R1 Distill Qwen 32B (BF16)",
  "deepseek-r1-distill-llama-8b-8bit": "DeepSeek R1 Distill Llama 8B (8-bit)",
  "deepseek-r1-distill-llama-70b-6bit": "DeepSeek R1 Distill Llama 70B (6-bit)",
  "deepseek-r1-distill-llama-70b-8bit": "DeepSeek R1 Distill Llama 70B (8-bit)",
  "deepseek-r1-distill-llama-8b": "DeepSeek R1 Distill Llama 8B",
  "deepseek-r1-distill-llama-8b-3bit": "DeepSeek R1 Distill Llama 8B (3-bit)",
  "deepseek-r1-distill-llama-8b-6bit": "DeepSeek R1 Distill Llama 8B (6-bit)",
  "deepseek-r1-distill-llama-8b-8bit": "DeepSeek R1 Distill Llama 8B (8-bit)",
  "deepseek-r1-distill-llama-8b-bf16": "DeepSeek R1 Distill Llama 8B (BF16)",
  "deepseek-r1-distill-llama-70b": "DeepSeek R1 Distill Llama 70B",
  "deepseek-r1-distill-llama-70b-3bit": "DeepSeek R1 Distill Llama 70B (3-bit)",
  "deepseek-r1-distill-llama-70b-6bit": "DeepSeek R1 Distill Llama 70B (6-bit)",
  "deepseek-r1-distill-llama-70b-8bit": "DeepSeek R1 Distill Llama 70B (8-bit)",
  "deepseek-r1-distill-qwen-32b-6bit": "DeepSeek R1 Distill Qwen 32B (6-bit)",
}

# Dynamic model registry for runtime additions
_dynamic_models = {}


def add_dynamic_model(model_id: str, repo_id: str, inference_engine_name: str, layers: int = 12):
  """
  Dynamically add a model to the supported list after it gets downloaded.
  This prevents issues where downloaded models aren't in the static registry.
  
  Args:
    model_id: The model identifier (e.g., "huggingface-distilgpt2")
    repo_id: The HuggingFace repository ID (e.g., "distilbert/distilgpt2")
    inference_engine_name: The inference engine class name (e.g., "HuggingFaceInferenceEngine")
    layers: Number of layers in the model (default: 12 for most transformer models)
  """
  global _dynamic_models
  if model_id not in _dynamic_models:
    _dynamic_models[model_id] = {
      "layers": layers,
      "repo": {}
    }
  
  _dynamic_models[model_id]["repo"][inference_engine_name] = repo_id
  
  print(f"DEBUG: Added dynamic model {model_id} with repo {repo_id} for engine {inference_engine_name}")


def get_model_info(model_id: str):
  """Get model info from both static registry and dynamic models."""
  # First check static registry
  if model_id in model_cards:
    return model_cards[model_id]
  
  # Then check dynamic registry
  if model_id in _dynamic_models:
    return _dynamic_models[model_id]
  
  return None


def get_all_model_ids():
  """Get all model IDs from both static and dynamic registries."""
  all_models = set(model_cards.keys())
  all_models.update(_dynamic_models.keys())
  return list(all_models)


def scan_huggingface_cache_and_register():
  """
  Scan the HuggingFace cache directory and automatically register any downloaded models
  that aren't in our static registry.
  """
  cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
  if not os.path.exists(cache_dir):
    return
  
  for item in os.listdir(cache_dir):
    if item.startswith("models--"):
      # Extract repo info from directory name (format: models--org--model-name)
      repo_parts = item[8:].split("--")  # Remove "models--" prefix
      if len(repo_parts) >= 2:
        org = repo_parts[0]
        model_name = "--".join(repo_parts[1:])  # Handle multi-part model names
        repo_id = f"{org}/{model_name}"
        
        # Create a model_id for our registry
        model_id = f"huggingface-{model_name.lower().replace('_', '-')}"
        
        # Check if this model is already in our static registry
        existing_repo = get_repo(model_id, "HuggingFaceInferenceEngine")
        if not existing_repo:
          # Auto-register with reasonable defaults
          layers = 12  # Default for most transformer models
          
          # Try to guess layer count from common model patterns
          if "large" in model_name.lower():
            layers = 24
          elif "7b" in model_name.lower():
            layers = 32
          elif "13b" in model_name.lower():
            layers = 40
          elif "bloom-560m" in model_name.lower():
            layers = 24
          elif "phi-2" in model_name.lower():
            layers = 32
          elif "falcon-7b" in model_name.lower():
            layers = 32
          elif "distilgpt2" in model_name.lower():
            layers = 6
          elif "opt-1.3b" in model_name.lower():
            layers = 24
          
          add_dynamic_model(model_id, repo_id, "HuggingFaceInferenceEngine", layers)
          print(f"Auto-registered cached model: {model_id} -> {repo_id}")


def get_repo_with_dynamic_fallback(model_id: str, inference_engine_classname: str) -> Optional[str]:
  """
  Enhanced version of get_repo that also checks dynamic models and can auto-register
  from HuggingFace cache if needed.
  """
  print(f"DEBUG: get_repo_with_dynamic_fallback called with model_id='{model_id}', engine='{inference_engine_classname}'")
  
  # First try static registry directly (not via get_repo to avoid recursion)
  static_repo = model_cards.get(model_id, {}).get("repo", {}).get(inference_engine_classname, None)
  if static_repo:
    print(f"DEBUG: Found in static registry: {static_repo}")
    return static_repo
  
  # Then try dynamic registry
  if model_id in _dynamic_models:
    dynamic_repo = _dynamic_models[model_id].get("repo", {}).get(inference_engine_classname, None)
    if dynamic_repo:
      print(f"DEBUG: Found in dynamic registry: {dynamic_repo}")
      return dynamic_repo
  
  # If not found and it's a HuggingFace model, try to auto-register from cache
  if inference_engine_classname == "HuggingFaceInferenceEngine":
    print("DEBUG: Scanning HuggingFace cache for auto-registration")
    scan_huggingface_cache_and_register()
    # Try dynamic registry again after scanning
    if model_id in _dynamic_models:
      final_repo = _dynamic_models[model_id].get("repo", {}).get(inference_engine_classname, None)
      if final_repo:
        print(f"DEBUG: Found after cache scan: {final_repo}")
        return final_repo
  
  print(f"DEBUG: No repo found for model_id='{model_id}', engine='{inference_engine_classname}'")
  return None


def build_base_shard(model_id: str, inference_engine_name: Optional[str] = None, start_layer: int = 0, end_layer: Optional[int] = None, n_layers: Optional[int] = None):
  """Build a base shard for the given model."""
  
  # Use enhanced function that can auto-register models
  model_info = get_model_info(model_id)
  if not model_info:
    raise ValueError(f"Model {model_id} not found in model cards")
  
  total_layers = model_info["layers"]
  
  if end_layer is None:
    end_layer = total_layers - 1
  if n_layers is None:
    n_layers = total_layers
    
  return Shard(
    model_id=model_id,
    start_layer=start_layer,
    end_layer=min(end_layer if end_layer is not None else total_layers - 1, total_layers - 1),
    n_layers=n_layers if n_layers is not None else total_layers
  )


def build_full_shard(model_id: str, inference_engine_name: Optional[str] = None):
  """Build a full shard that covers the entire model."""
  
  # Use enhanced function that can auto-register models
  model_info = get_model_info(model_id)
  if not model_info:
    raise ValueError(f"Model {model_id} not found in model cards")
  
  total_layers = model_info["layers"]
  
  return build_base_shard(model_id, inference_engine_name, 0, total_layers - 1, total_layers)


def get_repo(model_id: str, inference_engine_classname: str) -> Optional[str]:
  """
  Get the repository ID for a given model and inference engine.
  Now with dynamic model registration support.
  """
  return get_repo_with_dynamic_fallback(model_id, inference_engine_classname)

def get_pretty_name(model_id: str) -> Optional[str]:
  return pretty_name.get(model_id, None)

def get_supported_models(engine_lists):
  """Get supported models for the given engine lists."""
  if not engine_lists:
    # If no engine lists provided, return all models
    return list(model_cards.keys())
  
  supported_models = set()
  
  for engine_list in engine_lists:
    for model_id, model_info in model_cards.items():
      repo_info = model_info.get("repo", {})
      
      # Check if any of the engines in the list support this model
      for engine in engine_list:
        if engine in repo_info:
          supported_models.add(model_id)
          break
    
    # Also check dynamic models
    for model_id, model_info in _dynamic_models.items():
      repo_info = model_info.get("repo", {})
      
      # Check if any of the engines in the list support this model
      for engine in engine_list:
        if engine in repo_info:
          supported_models.add(model_id)
          break
  
  return list(supported_models)
