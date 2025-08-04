import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from exo.inference.shard import Shard
from exo.inference.inference_engine import InferenceEngine
from exo.helpers import DEBUG

# Force CPU-only for HuggingFace - MLX handles Apple Silicon optimization
device = "cpu"
HF_DEBUG = os.getenv("HF_DEBUG", "False").lower() == "true"

class HuggingFaceInferenceEngine(InferenceEngine):
    """
    CPU-only HuggingFace inference engine for models not available in MLX.
    Use MLX engine for Apple Silicon optimized models.
    """
    
    def __init__(self, shard_downloader=None):
        if HF_DEBUG:
            print("🔧 HuggingFaceInferenceEngine.__init__ starting (CPU-only mode)")
        
        self.model = None
        self.tokenizer = None
        self.shard = None
        self.shard_downloader = shard_downloader
        self.executor = ThreadPoolExecutor(max_workers=1)  # Single thread for CPU stability
        
        if HF_DEBUG:
            print(f"🔧 Using device: {device} (forced CPU for compatibility)")
            print("🔧 HuggingFaceInferenceEngine.__init__ completed")

    async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
        """Encode prompt to tokens"""
        if HF_DEBUG:
            print(f"🔧 encode() starting for shard {shard.model_id}, prompt length: {len(prompt)}")
        
        await self.ensure_shard(shard)
        
        def encode_fn():
            if HF_DEBUG:
                print("🔧 encode_fn() starting tokenization")
            
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer not loaded")
            
            # Use tokenizer to encode prompt
            tokens = self.tokenizer.encode(prompt, return_tensors="pt")
            input_ids = tokens.squeeze(0)  # Remove batch dimension
            
            if HF_DEBUG:
                print(f"🔧 encode_fn() tokenization complete, input_ids shape: {input_ids.shape}")
            
            # Convert to numpy
            result = input_ids.cpu().numpy()
            
            if HF_DEBUG:
                print(f"🔧 encode_fn() numpy conversion complete, result shape: {result.shape}")
            
            return result
        
        if HF_DEBUG:
            print("🔧 encode() submitting to executor")
        
        result = await asyncio.get_running_loop().run_in_executor(
            self.executor, encode_fn
        )
        
        if HF_DEBUG:
            print(f"🔧 encode() completed, tokens shape: {result.shape}")
        
        return result

    async def sample(self, x: np.ndarray, temp: float = 0.0, top_p: float = 0.0) -> np.ndarray:
        """Sample next token from logits"""
        if HF_DEBUG:
            print(f"🔧 sample() starting, input shape: {x.shape}, temp: {temp}, top_p: {top_p}")
        
        def sample_fn():
            if HF_DEBUG:
                print("🔧 sample_fn() starting")
            
            # Get logits for the last token
            logits = torch.tensor(x[0, -1, :])  # Last token's logits
            
            if HF_DEBUG:
                print(f"🔧 sample_fn() logits shape: {logits.shape}")
            
            # Apply temperature (use minimum to avoid division by zero)
            temp_to_use = max(temp, 0.1)
            logits = logits / temp_to_use
            
            # Convert to probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Sample token
            if top_p > 0:
                # Nucleus sampling
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probs[indices_to_remove] = 0
                probs = probs / probs.sum()
            
            # Sample from distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            if HF_DEBUG:
                print(f"🔧 sample_fn() completed, sampled token: {next_token.item()}")
            
            return next_token.numpy().astype(int)
        
        if HF_DEBUG:
            print("🔧 sample() submitting to executor")
        
        result = await asyncio.get_running_loop().run_in_executor(
            self.executor, sample_fn
        )
        
        if HF_DEBUG:
            print(f"🔧 sample() completed, result: {result}")
        
        return result

    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        """Decode tokens to text"""
        await self.ensure_shard(shard)
        
        def decode_fn():
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer not loaded")
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        return await asyncio.get_running_loop().run_in_executor(
            self.executor, decode_fn
        )

    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
        """Run inference on tensor data"""
        if HF_DEBUG:
            print(f"🔧 infer_tensor() starting for request {request_id}, shard {shard.model_id}, input shape: {input_data.shape}")
        
        await self.ensure_shard(shard)
        
        if inference_state is None:
            inference_state = {}
        
        def inference_fn():
            if HF_DEBUG:
                print("🔧 inference_fn() starting")
            
            # Convert to tensor and ensure CPU
            input_ids = torch.tensor(input_data, dtype=torch.long)
            if len(input_ids.shape) == 1:
                input_ids = input_ids.unsqueeze(0)  # Add batch dimension
            
            # Keep on CPU
            input_ids = input_ids.cpu()
            
            if HF_DEBUG:
                print(f"🔧 inference_fn() input_ids shape: {input_ids.shape}")
            
            if self.model is None:
                raise RuntimeError("Model not loaded")
            
            # Run forward pass
            with torch.no_grad():
                # Ensure model is on CPU
                self.model = self.model.cpu()
                outputs = self.model(input_ids)
                logits = outputs.logits
                
                if HF_DEBUG:
                    print(f"🔧 inference_fn() model forward completed, logits shape: {logits.shape}")
                
                # Convert to numpy
                result = logits.cpu().numpy()
                
                if HF_DEBUG:
                    print(f"🔧 inference_fn() completed, result shape: {result.shape}")
                
                return result
        
        if HF_DEBUG:
            print("🔧 infer_tensor() submitting to executor")
        
        output_data = await asyncio.get_running_loop().run_in_executor(
            self.executor, inference_fn
        )
        
        if HF_DEBUG:
            print(f"🔧 infer_tensor() completed successfully")
        
        return output_data, inference_state

    async def ensure_shard(self, shard: Shard):
        """Ensure model and tokenizer are loaded"""
        if HF_DEBUG:
            print(f"🔧 ensure_shard() starting for {shard.model_id}")
        
        if self.shard == shard and self.model is not None:
            if HF_DEBUG:
                print("🔧 ensure_shard() shard already loaded, returning")
            return
        
        def load_model():
            if HF_DEBUG:
                print("🔧 load_model() starting model load")
            
            # Get the actual HuggingFace repo ID from the model registry
            from exo.models import get_repo_with_dynamic_fallback
            
            repo_id = get_repo_with_dynamic_fallback(shard.model_id, "HuggingFaceInferenceEngine")
            
            if HF_DEBUG:
                print(f"🔧 load_model() loading {repo_id} on {device}")
            
            # Load model and tokenizer - CPU only for maximum compatibility
            model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map="cpu",  # Force CPU
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Ensure model is on CPU
            model = model.cpu()
            model.eval()
            
            tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
            
            # Set pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            if HF_DEBUG:
                print("🔧 load_model() completed successfully")
            
            return model, tokenizer
        
        if HF_DEBUG:
            print("🔧 ensure_shard() submitting load to executor")
        
        self.model, self.tokenizer = await asyncio.get_running_loop().run_in_executor(
            self.executor, load_model
        )
        
        self.shard = shard
        
        if HF_DEBUG:
            print(f"🔧 ensure_shard() completed successfully for {shard.model_id}")

    async def load_checkpoint(self, shard: Shard, path: str):
        """Load checkpoint (not implemented for HuggingFace)"""
        pass
