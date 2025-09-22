import argparse
import json
from pathlib import Path
import sys
import warnings

# PyTorch 경고 메시지 숨기기
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from unifiedpdf.config import PipelineConfig
from unifiedpdf.types import Chunk
from unifiedpdf.embedding import get_embedder


def load_corpus(path: str):
    p = Path(path)
    chunks = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            chunks.append(
                Chunk(
                    doc_id=obj.get("doc_id", obj.get("filename", "doc")),
                    filename=obj.get("filename", "doc"),
                    page=obj.get("page"),
                    start_offset=int(obj.get("start", 0)),
                    length=int(obj.get("length", len(obj.get("text", "")))),
                    text=obj.get("text", ""),
                    extra=obj.get("extra", {}),
                )
            )
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus_v1.jsonl")
    ap.add_argument("--backend", default="faiss", choices=["faiss", "hnsw"]) 
    ap.add_argument("--outdir", default="vector_store")
    # 벡터 레벨 중복 제거 (선택적)
    ap.add_argument("--dedup-vectors", action="store_true", help="Remove duplicate vectors before building index")
    ap.add_argument("--vector-similarity-threshold", type=float, default=0.99, help="Cosine similarity threshold for vector deduplication")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    chunks = load_corpus(args.corpus)
    embedder = get_embedder(PipelineConfig().embedding_model, use_gpu=False)
    if embedder is None:
        print("sentence-transformers not available; cannot build index.")
        return
    
    # 벡터 중복 제거 로직
    if args.dedup_vectors:
        print("벡터 중복 제거 중...")
        texts = [c.text for c in chunks]
        embs = embedder.embed_texts(texts)
        
        # 코사인 유사도 기반 중복 벡터 제거
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        unique_indices = []
        seen_vectors = []
        
        for i, emb in enumerate(embs):
            is_duplicate = False
            for seen_emb in seen_vectors:
                similarity = cosine_similarity([emb], [seen_emb])[0][0]
                if similarity >= args.vector_similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_indices.append(i)
                seen_vectors.append(emb)
        
        # 중복 제거된 데이터로 업데이트
        chunks = [chunks[i] for i in unique_indices]
        embs = np.array([embs[i] for i in unique_indices])
        print(f"벡터 중복 제거 완료: {len(unique_indices)}개 유니크 벡터 유지")
    else:
        texts = [c.text for c in chunks]
        embs = embedder.embed_texts(texts)

    if args.backend == "faiss":
        try:
            import faiss  # type: ignore
        except Exception:
            print("faiss-cpu not available; cannot build FAISS index.")
            return
        index = faiss.index_factory(embedder.dim, "Flat")
        index.add(embs)
        faiss.write_index(index, str(outdir / "index.faiss"))
        meta = {"dim": embedder.dim, "space": "l2"}
        with (outdir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        with (outdir / "mapping.json").open("w", encoding="utf-8") as f:
            json.dump(list(range(len(chunks))), f, ensure_ascii=False)
        print(f"FAISS index written to {outdir}")
    else:
        try:
            import hnswlib  # type: ignore
        except Exception:
            print("hnswlib not available; cannot build HNSW index.")
            return
        p = hnswlib.Index(space="cosine", dim=embedder.dim)
        p.init_index(max_elements=len(chunks), ef_construction=200, M=16)
        import numpy as np
        labels = np.arange(len(chunks))
        p.add_items(embs, labels)
        p.save_index(str(outdir / "index.hnsw"))
        meta = {"dim": embedder.dim, "space": "cosine"}
        with (outdir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        with (outdir / "mapping.json").open("w", encoding="utf-8") as f:
            json.dump(list(range(len(chunks))), f, ensure_ascii=False)
        print(f"HNSW index written to {outdir}")


if __name__ == "__main__":
    main()
