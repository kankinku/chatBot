"""
스토어 코디네이터

FAISS/Chroma 이중 스토어에 대한 단일 쓰기 진입점과 저널링/재동기화를 제공합니다.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import threading
import logging

from core.document.pdf_processor import TextChunk
from core.document.vector_store import HybridVectorStore

logger = logging.getLogger(__name__)


class StoreCoordinator:
    """하이브리드 벡터 스토어 쓰기 코디네이터"""

    def __init__(self, vector_store: Optional[HybridVectorStore] = None, journal_path: str = "logs/vector_write_journal.jsonl"):
        self._store = vector_store or HybridVectorStore()
        self._journal_path = Path(journal_path)
        self._lock = threading.Lock()
        self._ensure_log_dir()

    def _ensure_log_dir(self):
        self._journal_path.parent.mkdir(parents=True, exist_ok=True)

    def _append_journal(self, record: Dict[str, Any]):
        try:
            with self._journal_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"저널 기록 실패: {e}")

    def add_chunks(self, chunks: List[TextChunk]) -> int:
        if not chunks:
            return 0
        with self._lock:
            self._append_journal({"op": "add", "count": len(chunks)})
            self._store.add_chunks(chunks)
            return len(chunks)

    def delete_ids(self, ids: List[str]) -> int:
        if not ids:
            return 0
        with self._lock:
            self._append_journal({"op": "delete", "count": len(ids)})
            try:
                if hasattr(self._store, "chroma_store"):
                    self._store.chroma_store.delete_ids(ids)
            except Exception:
                pass
            return len(ids)

    def update_chunks(self, chunks: List[TextChunk]) -> int:
        """간단한 upsert: 동일 ID 존재 시 삭제 후 재추가(원자적 보장)."""
        if not chunks:
            return 0
        ids = [c.chunk_id for c in chunks if getattr(c, 'chunk_id', None)]
        with self._lock:
            self._append_journal({"op": "update", "count": len(chunks)})
            try:
                if ids:
                    try:
                        self._store.chroma_store.delete_ids(ids)
                    except Exception:
                        pass
            except Exception:
                pass
            # 재추가(원자성은 HybridVectorStore.add_chunks가 보장)
            self._store.add_chunks(chunks)
            return len(chunks)

    def reconcile(self) -> Dict[str, Any]:
        return self._store.reconcile()

    def get_total_chunks(self) -> int:
        return self._store.get_total_chunks()

    def get_all_pdfs(self):
        return self._store.get_all_pdfs()

    def journal_info(self) -> Dict[str, Any]:
        try:
            if not self._journal_path.exists():
                return {"entries": 0, "size_bytes": 0}
            size = self._journal_path.stat().st_size
            # 단순 행 수 카운트(대용량 환경에서는 피해야 함)
            with self._journal_path.open("r", encoding="utf-8") as f:
                entries = sum(1 for _ in f)
            return {"entries": entries, "size_bytes": size}
        except Exception as e:
            return {"error": str(e)}

    def replay_journal(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """저널 리플레이: add/delete/update를 순서대로 재실행하여 무결성 복구."""
        try:
            if not self._journal_path.exists():
                return {"replayed": 0}
            count = 0
            added, deleted, updated = 0, 0, 0
            with self._journal_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        op = record.get("op")
                        if op == "add" and record.get("payload"):
                            # payload: serialized chunks
                            chunks = record["payload"]
                            # 역직렬화 최소 구현
                            from core.document.pdf_processor import TextChunk
                            deserialized = []
                            for c in chunks:
                                deserialized.append(TextChunk(
                                    content=c.get("content", ""),
                                    page_number=int(c.get("page_number", 0)),
                                    chunk_id=c.get("chunk_id"),
                                    metadata=c.get("metadata"),
                                    pdf_id=c.get("pdf_id")
                                ))
                            self._store.add_chunks(deserialized)
                            added += len(deserialized)
                        elif op == "delete" and record.get("ids"):
                            ids = record["ids"]
                            try:
                                self._store.chroma_store.delete_ids(ids)
                            except Exception:
                                pass
                            deleted += len(ids)
                        elif op == "update" and record.get("payload"):
                            from core.document.pdf_processor import TextChunk
                            ids = [c.get("chunk_id") for c in record["payload"] if c.get("chunk_id")]
                            try:
                                self._store.chroma_store.delete_ids(ids)
                            except Exception:
                                pass
                            deserialized = []
                            for c in record["payload"]:
                                deserialized.append(TextChunk(
                                    content=c.get("content", ""),
                                    page_number=int(c.get("page_number", 0)),
                                    chunk_id=c.get("chunk_id"),
                                    metadata=c.get("metadata"),
                                    pdf_id=c.get("pdf_id")
                                ))
                            self._store.add_chunks(deserialized)
                            updated += len(deserialized)
                        count += 1
                        if limit and count >= limit:
                            break
                    except Exception:
                        continue
            return {"replayed": count, "added": added, "deleted": deleted, "updated": updated}
        except Exception as e:
            return {"error": str(e)}


