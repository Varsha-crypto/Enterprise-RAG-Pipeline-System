"""
Legacy LLM Service for RAG Pipeline, using Hugging Face Transformers for local inference.
Supports: Llama-3.2-3B, Mistral-7B-Instruct, Phi-3-mini
"""

import logging
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time

logger = logging.getLogger(__name__)


class LegacyLLMService:
    """
    LLM-based summarization and Q&A.
    Loads models on-demand and caches them in memory.
    """
    
    def __init__(self):
        self.loaded_models = {}  # Cache: model_name -> (model, tokenizer)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"LegacyLLMService initialized with device: {self.device}")
        
        # Model configurations
        self.model_configs = {
            "llama-3.2-3b": {
                "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
                "max_length": 2048,
                "temperature": 0.7
            },
            "mistral-7b": {
                "hf_id": "mistralai/Mistral-7B-Instruct-v0.2",
                "max_length": 2048,
                "temperature": 0.7
            },
            "phi-3-mini": {
                "hf_id": "microsoft/Phi-3-mini-4k-instruct",
                "max_length": 2048,
                "temperature": 0.7
            }
        }
    
    def load_model(self, model_name: str) -> tuple:
        if model_name in self.loaded_models:
            logger.info(f"Using cached model: {model_name}")
            return self.loaded_models[model_name]
        
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.model_configs.keys())}")
        
        config = self.model_configs[model_name]
        hf_id = config["hf_id"]
        
        logger.info(f"Loading model: {model_name} ({hf_id})...")
        start_time = time.time()
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(hf_id)
            
            # Load model with appropriate settings for laptop
            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                model = model.to(self.device)
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f}s")
            
            # Cache
            self.loaded_models[model_name] = (model, tokenizer)
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def generate_summary(
        self,
        query: str,
        retrieved_chunks: List[str],
        model_name: str = "llama-3.2-3b",
        max_new_tokens: int = 300
    ) -> Dict[str, Any]:
        
        logger.info(f"Generating summary with {model_name}")
        start_time = time.time()
        
        # Load model
        model, tokenizer = self.load_model(model_name)
        config = self.model_configs[model_name]
        
        # Build context from chunks
        context = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(retrieved_chunks)])
        
        # Truncate context if too long
        max_context_tokens = 1500  # Leave room for prompt and response
        context_tokens = tokenizer.encode(context)
        if len(context_tokens) > max_context_tokens:
            context_tokens = context_tokens[:max_context_tokens]
            context = tokenizer.decode(context_tokens)
            logger.warning(f"Context truncated to {max_context_tokens} tokens")
        
        # Build prompt
        prompt = self._build_prompt(query, context, model_name)
        
        # Generate
        try:
            gen_start = time.time()
            
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=config["temperature"],
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode only the newly generated tokens
            input_length = inputs.input_ids.shape[1]
            new_tokens = outputs[0][input_length:]
            summary = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            gen_time = time.time() - gen_start
            total_time = time.time() - start_time
            
            logger.info(f"Summary generated in {gen_time:.2f}s")
            
            return {
                "summary": summary,
                "model_used": model_name,
                "num_chunks_used": len(retrieved_chunks),
                "context_length": len(context_tokens) if 'context_tokens' in locals() else 0,
                "generation_time_seconds": round(gen_time, 2),
                "total_time_seconds": round(total_time, 2)
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _build_prompt(self, query: str, context: str, model_name: str) -> str:
        """
        Build model-specific prompt. Different models expect different formats.
        """
        if "llama" in model_name.lower():
            # Llama-3 chat format
            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant. Answer the user's question based on the provided context. Be concise and accurate.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
{context}

Question: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        elif "mistral" in model_name.lower():
            # Mistral instruct format
            return f"""[INST] You are a helpful AI assistant. Answer the question based on the provided context.

Context:
{context}

Question: {query} [/INST]

"""
        
        elif "phi" in model_name.lower():
            # Phi-3 format
            return f"""<|system|>
You are a helpful AI assistant. Answer based on the provided context.<|end|>
<|user|>
Context:
{context}

Question: {query}<|end|>
<|assistant|>
"""
        
        else:
            # Generic format
            return f"""Answer the following question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""
    
    def warm_up_model(self, model_name: str):
        logger.info(f"Warming up model: {model_name}")
        self.load_model(model_name)
        logger.info(f"Model {model_name} ready for inference")
    
    def unload_model(self, model_name: str):
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info(f"Unloaded model: {model_name}")
    
    def get_available_models(self) -> List[str]:
        return list(self.model_configs.keys())


# Global service instance
#llm_service = LLMService()