"""
자동 품질 루프 스켈레톤
- 골든셋 저장/조회/삭제
- 오프라인 지표(nDCG/MRR/Recall/정답률) 계산
- 설정/모델/인덱스 변경점 교차표 생성(간단 버전)
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import math

GOLDENSET_PATH = Path("data/golden_set.json")


def _ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def load_golden_set() -> List[Dict[str, Any]]:
    if not GOLDENSET_PATH.exists():
        return []
    try:
        with GOLDENSET_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_golden_set(items: List[Dict[str, Any]]):
    _ensure_parent(GOLDENSET_PATH)
    with GOLDENSET_PATH.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def add_golden_item(item: Dict[str, Any]) -> int:
    items = load_golden_set()
    items.append(item)
    save_golden_set(items)
    return len(items)


def delete_golden_item(idx: int) -> bool:
    items = load_golden_set()
    if 0 <= idx < len(items):
        items.pop(idx)
        save_golden_set(items)
        return True
    return False


def compute_metrics(golden: List[Dict[str, Any]], predictions: List[Dict[str, Any]], k: int = 5) -> Dict[str, float]:
    """간단한 nDCG@k / MRR@k / Recall@k / Accuracy 계산.
    golden: [{question, answer, relevant_chunk_ids: [id,...]}]
    predictions: [{question, predicted_answer, retrieved_chunk_ids: [id,...]}]
    매칭은 질문 텍스트 기준으로 조인(간이).
    """
    if not golden or not predictions:
        return {"ndcg": 0.0, "mrr": 0.0, "recall": 0.0, "accuracy": 0.0}

    # 골든 맵 구성
    gmap: Dict[str, Dict[str, Any]] = {}
    for g in golden:
        q = (g.get("question") or "").strip()
        gmap[q] = g

    total = 0
    ndcg_sum = 0.0
    mrr_sum = 0.0
    recall_sum = 0.0
    acc_sum = 0.0

    for p in predictions:
        q = (p.get("question") or "").strip()
        if q not in gmap:
            continue
        total += 1
        g = gmap[q]
        rel_ids = list(g.get("relevant_chunk_ids") or [])
        pred_ids = list(p.get("retrieved_chunk_ids") or [])

        # DCG@k
        dcg = 0.0
        for rank, cid in enumerate(pred_ids[:k], start=1):
            gain = 1.0 if cid in rel_ids else 0.0
            if gain > 0:
                dcg += gain / math.log2(rank + 1)
        # IDCG@k
        ideal_gains = [1.0] * min(len(rel_ids), k)
        idcg = 0.0
        for rank, gain in enumerate(ideal_gains, start=1):
            idcg += gain / math.log2(rank + 1)
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        ndcg_sum += ndcg

        # MRR@k
        rr = 0.0
        for rank, cid in enumerate(pred_ids[:k], start=1):
            if cid in rel_ids:
                rr = 1.0 / rank
                break
        mrr_sum += rr

        # Recall@k
        if rel_ids:
            hit = len(set(rel_ids) & set(pred_ids[:k]))
            recall_sum += hit / float(len(rel_ids))
        else:
            recall_sum += 0.0

        # Accuracy (간단: 정답 문자열 부분 매칭)
        gold_ans = (g.get("answer") or "").strip()
        pred_ans = (p.get("predicted_answer") or "").strip()
        acc_sum += 1.0 if gold_ans and gold_ans[:30] in pred_ans else 0.0

    if total == 0:
        return {"ndcg": 0.0, "mrr": 0.0, "recall": 0.0, "accuracy": 0.0}

    return {
        "ndcg": ndcg_sum / total,
        "mrr": mrr_sum / total,
        "recall": recall_sum / total,
        "accuracy": acc_sum / total,
    }


def generate_change_matrix(current_config: Dict[str, Any], baseline: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """설정/모델/인덱스 변경점 간단 교차표.
    baseline이 없으면 현재 설정의 핵심 키만 요약.
    """
    keys = [
        "LLM_MODEL", "EMBEDDING_MODEL", "CROSS_ENCODER_MODEL",
        "SLA_P95_TARGET_MS", "SEARCH_TIMEOUT_S", "LLM_TIMEOUT_S",
    ]
    summary = {k: current_config.get(k) for k in keys}
    if not baseline:
        return {"current": summary, "diff": {}}
    diff = {}
    for k in keys:
        if baseline.get(k) != current_config.get(k):
            diff[k] = {"before": baseline.get(k), "after": current_config.get(k)}
    return {"current": summary, "diff": diff}


