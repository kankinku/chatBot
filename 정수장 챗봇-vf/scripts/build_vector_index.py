import argparse
import json
from pathlib import Path
import sys

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
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--backend", default="faiss", choices=["faiss", "hnsw"]) 
    ap.add_argument("--outdir", default="vector_store")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    chunks = load_corpus(args.corpus)
    embedder = get_embedder(PipelineConfig().embedding_model, use_gpu=False)
    if embedder is None:
        print("sentence-transformers not available; cannot build index.")
        return

    texts = [c.text for c in chunks]
    embs = embedder.embed_texts(texts)

    if args.backend == "faiss":
        try:
            import faiss  # type: ignore
        except Exception:
            print("faiss-cpu not available; cannot build FAISS index.")
            return
        index = faiss.index_factory(embedder.dim, "FlatIP")
        index.add(embs)
        faiss.write_index(index, str(outdir / "index.faiss"))
        meta = {"dim": embedder.dim, "space": "ip"}
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
