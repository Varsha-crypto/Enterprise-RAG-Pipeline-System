"""
Embedding Service - Uses SentenceTransformer for text encoding.
Auto-selects CUDA > CPU based on availability.
"""

import os
import logging
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv
from typing import Optional

logger = logging.getLogger(__name__)
load_dotenv()

MODEL_NAME = os.getenv('MODEL_NAME', 'BAAI/bge-m3')
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '1024'))


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class EmbeddingService:
    def __init__(self):
        self.models = {}
        self.current_model_name = None
        self.device = _best_device()
        logger.info(f"EmbeddingService initialized on device: {self.device}")

    def load_model(self, model_name: str):
        if model_name not in self.models:
            logger.info(f"Loading embedding model: {model_name} on {self.device}")
            kwargs = {"device": self.device}

            if "nomic" in model_name.lower():
                kwargs["trust_remote_code"] = True
            elif "bge-m3" in model_name.lower():
                kwargs["revision"] = "refs/pr/130"

            try:
                self.models[model_name] = SentenceTransformer(model_name, **kwargs)
                logger.info(f"Model {model_name} loaded on {self.device}")
            except Exception as e:
                if self.device != "cpu":
                    logger.warning(f"GPU load failed ({e}), retrying on CPU")
                    kwargs["device"] = "cpu"
                    self.models[model_name] = SentenceTransformer(model_name, **kwargs)
                    logger.info(f"Model {model_name} loaded on cpu (fallback)")
                else:
                    logger.error(f"Failed to load {model_name}: {e}")
                    raise

        self.current_model_name = model_name

    def encode(self, text: str, model_name: Optional[str] = None, normalize: bool = True) -> np.ndarray:  # type: ignore
        if model_name:
            self.load_model(model_name)
        elif self.current_model_name is None:
            self.load_model(os.getenv('MODEL_NAME', 'BAAI/bge-m3'))

        model = self.models[self.current_model_name]
        embedding = model.encode([text], normalize_embeddings=normalize, show_progress_bar=False)[0]
        return embedding.astype('float32')

    def encode_batch(self, texts: list, model_name: Optional[str] = None, normalize: bool = True) -> np.ndarray:  # type: ignore
        if model_name:
            self.load_model(model_name)
        elif self.current_model_name is None:
            self.load_model(os.getenv('MODEL_NAME', 'BAAI/bge-m3'))

        model = self.models[self.current_model_name]
        embeddings = model.encode(texts, normalize_embeddings=normalize, show_progress_bar=False)
        return embeddings.astype('float32')

    def get_dimension(self, model_name: str = None) -> int:  # type: ignore
        if model_name:
            self.load_model(model_name)
        model = self.models[self.current_model_name]
        return model.get_sentence_embedding_dimension()

    def get_loaded_models(self) -> list:
        return list(self.models.keys())


# Global instance
embedding_service = EmbeddingService()
