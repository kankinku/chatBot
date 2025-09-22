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


def get_embedder(model_name: str, use_gpu: bool) -> Optional[EmbeddingModel]:
    try:
        device = "cuda" if use_gpu else None
        return SentenceTransformerEmbedder(model_name=model_name, device=device)
    except Exception:
        return None

