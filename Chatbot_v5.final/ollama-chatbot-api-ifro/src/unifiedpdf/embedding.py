from __future__ import annotations

from typing import List, Optional


class EmbeddingModel:
    def embed_texts(self, texts: List[str]) -> "np.ndarray":  # pragma: no cover - interface
        raise NotImplementedError

    def embed_query(self, text: str) -> "np.ndarray":  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def dim(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError


class SentenceTransformerEmbedder(EmbeddingModel):
    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask", device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer  # type: ignore
        import torch  # type: ignore

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self._dim = self.model.get_sentence_embedding_dimension()
        self._model_name = model_name
        self._device = device
    
    def __del__(self):
        """모델 메모리 해제"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                # GPU 메모리 정리
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 모델 참조 해제
                del self.model
                self.model = None
        except Exception:
            pass  # 정리 실패 시 무시

    @property
    def dim(self) -> int:
        return int(self._dim)

    def embed_texts(self, texts: List[str]):
        import numpy as np

        embs = self.model.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return embs.astype("float32")

    def embed_query(self, text: str):
        import numpy as np

        emb = self.embed_texts([text])[0]
        return emb.astype("float32")


# 전역 임베딩 모델 캐시 (메모리 누수 방지)
_embedder_cache = {}
_cache_lock = None

def get_embedder(model_name: str, use_gpu: bool) -> Optional[EmbeddingModel]:
    global _cache_lock
    if _cache_lock is None:
        import threading
        _cache_lock = threading.Lock()
    
    try:
        device = "cuda" if use_gpu else None
        cache_key = f"{model_name}_{device}"
        
        with _cache_lock:
            if cache_key not in _embedder_cache:
                _embedder_cache[cache_key] = SentenceTransformerEmbedder(model_name=model_name, device=device)
            return _embedder_cache[cache_key]
    except Exception:
        return None

def clear_embedder_cache():
    """임베딩 모델 캐시 정리"""
    global _embedder_cache, _cache_lock
    if _cache_lock is None:
        import threading
        _cache_lock = threading.Lock()
    
    with _cache_lock:
        for embedder in _embedder_cache.values():
            if hasattr(embedder, '__del__'):
                embedder.__del__()
        _embedder_cache.clear()

