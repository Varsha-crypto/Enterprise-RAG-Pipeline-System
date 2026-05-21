"""
LLM Service — local inference with HuggingFace transformers.
Auto-selects CUDA > CPU. Lazy model loading on first request.
"""

import logging
import threading
import torch
from typing import List, Dict, Any, Optional, Iterator

logger = logging.getLogger(__name__)

# Map short names used by the frontend to HuggingFace model IDs
_MODEL_MAP: Dict[str, str] = {
    "qwen3-0.6b":        "Qwen/Qwen3-0.6B",
    "qwen2.5-0.5b":      "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B":   "Qwen/Qwen3-0.6B",
}
_DEFAULT_MODEL = "qwen3-0.6b"


def _best_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class LLMService:
    def __init__(self):
        self.device = _best_device()
        self.loaded_models: Dict[str, Any] = {}   # name → {"model": ..., "tokenizer": ...}
        self.model_configs: Dict[str, Any] = {}
        self._lock = threading.Lock()
        logger.info(f"LLMService initialized on device: {self.device}")

    # ── internal ──────────────────────────────────────────────────────────────

    def _resolve(self, model_name: str) -> str:
        return _MODEL_MAP.get(model_name, model_name)

    def load_model(self, model_name: str):
        hf_id = self._resolve(model_name)
        if hf_id in self.loaded_models:
            return
        with self._lock:
            if hf_id in self.loaded_models:
                return
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info(f"Loading LLM: {hf_id} on {self.device}")
            tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                trust_remote_code=True,
            )
            model.eval()
            self.loaded_models[hf_id] = {"model": model, "tokenizer": tokenizer}
            logger.info(f"LLM {hf_id} ready on {self.device}")

    def unload_model(self, model_name: str):
        hf_id = self._resolve(model_name)
        if hf_id in self.loaded_models:
            del self.loaded_models[hf_id]
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info(f"Unloaded LLM: {hf_id}")

    def warm_up_model(self, model_name: str):
        try:
            self.load_model(model_name)
        except Exception as e:
            logger.warning(f"Warm-up failed for {model_name}: {e}")

    def get_available_models(self) -> List[str]:
        return list(_MODEL_MAP.keys())

    # ── prompt ────────────────────────────────────────────────────────────────

    def _build_prompt(self, query: str, context: str, model_name: str, tokenizer=None, system_prompt: str = None) -> str:
        system = system_prompt if system_prompt and system_prompt.strip() else (
            "You are a helpful assistant. Answer thoroughly and in detail based only on the "
            "context provided. Use all relevant information from the context to give a complete answer."
        )
        user = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Using the context above, provide a detailed and comprehensive answer. "
            "Include all relevant facts, figures, and details from the context."
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                # Qwen3 supports enable_thinking=False to skip the <think> block
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False
                )
            except TypeError:
                # Older tokenizer without enable_thinking param
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

    # ── generation ────────────────────────────────────────────────────────────

    @staticmethod
    def _strip_think(text: str) -> str:
        """Remove <think>...</think> block (complete or truncated) from Qwen3 output."""
        import re
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)
        return cleaned.strip()

    @staticmethod
    def _clean_token(tok: str) -> str:
        """Strip chat-template special tokens that survive skip_special_tokens=False."""
        import re
        return re.sub(r'<\|[^|]+\|>', '', tok)

    def generate_summary(
        self,
        query: str,
        retrieved_chunks: List[str],
        model_name: str = _DEFAULT_MODEL,
        max_new_tokens: int = 1200,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not retrieved_chunks:
            return {"summary": None}

        # Cap tokens on CPU to avoid multi-minute waits on laptops without GPU
        if self.device == "cpu":
            max_new_tokens = min(max_new_tokens, 512)

        try:
            self.load_model(model_name)
            hf_id = self._resolve(model_name)
            entry = self.loaded_models[hf_id]
            model, tokenizer = entry["model"], entry["tokenizer"]

            context = "\n\n".join(retrieved_chunks[:10])
            prompt = self._build_prompt(query, context, model_name, tokenizer, system_prompt)

            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            summary = self._strip_think(raw)
            return {"summary": summary or None}

        except Exception as e:
            logger.error(f"LLM generate_summary failed: {e}", exc_info=True)
            return {"summary": None}

    def stream_summary(
        self,
        query: str,
        retrieved_chunks: List[str],
        model_name: str = _DEFAULT_MODEL,
        max_new_tokens: int = 1200,
        system_prompt: Optional[str] = None,
    ) -> Iterator[tuple]:
        # Cap tokens on CPU to avoid multi-minute waits on laptops without GPU
        if self.device == "cpu":
            max_new_tokens = min(max_new_tokens, 512)
        """
        Yield (type, token) tuples where type is 'think' or 'answer'.
        Qwen3 emits <think>...</think> before the actual answer.
        """
        if not retrieved_chunks:
            return

        try:
            self.load_model(model_name)
            hf_id = self._resolve(model_name)
            entry = self.loaded_models[hf_id]
            model, tokenizer = entry["model"], entry["tokenizer"]

            context = "\n\n".join(retrieved_chunks[:10])
            prompt = self._build_prompt(query, context, model_name, tokenizer, system_prompt)

            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

            from transformers import TextIteratorStreamer
            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=False
            )

            gen_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )

            thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
            thread.start()

            # Track whether we are inside a <think> block
            in_think = False
            buf = ""
            for raw_token in streamer:
                buf += raw_token
                # Detect opening tag
                if "<think>" in buf and not in_think:
                    before, _, after = buf.partition("<think>")
                    if before:
                        cleaned = self._clean_token(before)
                        if cleaned:
                            yield ("answer", cleaned)
                    in_think = True
                    buf = after
                    continue
                # Detect closing tag
                if "</think>" in buf and in_think:
                    think_part, _, after = buf.partition("</think>")
                    if think_part:
                        yield ("think", think_part)
                    in_think = False
                    buf = after
                    continue
                # Flush buffered text when safe (no partial tag at the end)
                if in_think:
                    if not buf.endswith("<") and not buf.endswith("</"):
                        yield ("think", buf)
                        buf = ""
                else:
                    if not buf.endswith("<") and not buf.endswith("</"):
                        cleaned = self._clean_token(buf)
                        if cleaned:
                            yield ("answer", cleaned)
                        buf = ""

            # Flush remainder
            if buf:
                cleaned = self._clean_token(buf)
                if cleaned:
                    yield ("think" if in_think else "answer", cleaned)

            thread.join()

        except Exception as e:
            logger.error(f"LLM stream_summary failed: {e}", exc_info=True)


llm_service = LLMService()
