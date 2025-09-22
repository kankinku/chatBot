import time
import json
import os
import sys

# 워크스페이스 루트를 파이썬 경로에 추가
sys.path.insert(0, os.getcwd())

from core.legal import (
    create_optimized_router,
    LegalMode,
)


def run_benchmark(queries):
    router = create_optimized_router()
    router.initialize_components()

    results = []
    for q in queries:
        t0 = time.time()
        resp = router.route_legal_query(q, mode=LegalMode.ACCURACY)
        t1 = time.time()

        # 수집: 상위 결과 간단 요약 및 근거
        items = []
        for r in getattr(resp, 'results', [])[:5]:
            md = getattr(r.chunk, 'metadata', None)
            items.append({
                'law_title': getattr(md, 'law_title', None) if md else None,
                'article_no': getattr(md, 'article_no', None) if md else None,
                'confidence': getattr(r, 'confidence', None),
            })

        results.append({
            'query': q,
            'latency_sec': round(t1 - t0, 3),
            'confidence': getattr(resp, 'confidence', None),
            'results': items,
        })

    return results


if __name__ == "__main__":
    # 대표 질의 (정확성 평가를 위한 보수적 세트)
    queries = [
        "근로기준법에 따른 휴가 규정은?",
        "개인정보 유출 시 사업자의 법적 조치는?",
        "수질오염 행위에 대한 처벌 기준은?",
    ]

    out = run_benchmark(queries)
    print(json.dumps(out, ensure_ascii=False, indent=2))


