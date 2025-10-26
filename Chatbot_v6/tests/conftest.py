"""
Pytest 설정 및 공통 fixture
"""

import pytest
from pathlib import Path
from typing import List

from modules.core.types import Chunk, RetrievedSpan


@pytest.fixture
def sample_chunks() -> List[Chunk]:
    """테스트용 샘플 청크"""
    return [
        Chunk(
            doc_id="test_doc_1",
            filename="test.pdf",
            page=1,
            start_offset=0,
            length=100,
            text="고산 정수장 AI플랫폼 URL은 waio-portal-vip:10011입니다. 로그인 계정은 관리자(admin)입니다.",
        ),
        Chunk(
            doc_id="test_doc_1",
            filename="test.pdf",
            page=1,
            start_offset=100,
            length=120,
            text="수질 모니터링 시스템은 pH, 탁도, 온도를 실시간으로 측정합니다. pH 기준은 6.5-8.5입니다.",
        ),
        Chunk(
            doc_id="test_doc_1",
            filename="test.pdf",
            page=2,
            start_offset=220,
            length=150,
            text="고산 정수장의 설계 용량은 50,000 m³/day입니다. 현재 처리량은 평균 35,000 m³/day입니다.",
        ),
        Chunk(
            doc_id="test_doc_2",
            filename="manual.pdf",
            page=5,
            start_offset=0,
            length=80,
            text="응집 공정에서는 PAC를 사용하며, 주입률은 20-30 mg/L입니다.",
        ),
        Chunk(
            doc_id="test_doc_2",
            filename="manual.pdf",
            page=5,
            start_offset=80,
            length=90,
            text="침전지 체류시간은 2-3시간이며, 슬러지는 매일 배출합니다.",
        ),
    ]


@pytest.fixture
def sample_retrieved_spans(sample_chunks) -> List[RetrievedSpan]:
    """테스트용 검색 결과"""
    return [
        RetrievedSpan(
            chunk=sample_chunks[0],
            source="hybrid",
            score=0.95,
            rank=1,
            aux_scores={"vector": 0.92, "bm25": 0.98},
        ),
        RetrievedSpan(
            chunk=sample_chunks[1],
            source="hybrid",
            score=0.82,
            rank=2,
            aux_scores={"vector": 0.85, "bm25": 0.79},
        ),
        RetrievedSpan(
            chunk=sample_chunks[2],
            source="hybrid",
            score=0.75,
            rank=3,
            aux_scores={"vector": 0.70, "bm25": 0.80},
        ),
    ]


@pytest.fixture
def test_data_dir(tmp_path) -> Path:
    """임시 테스트 데이터 디렉토리"""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def domain_dict_path(test_data_dir) -> str:
    """테스트용 도메인 사전"""
    import json
    
    domain_dict = {
        "keywords": ["AI", "플랫폼", "정수장", "수질"],
        "units": ["mg/L", "m³/day", "℃", "pH", "NTU"],
        "technical_spec": ["모델", "알고리즘", "성능"],
        "system_info": ["URL", "계정", "비밀번호"],
    }
    
    dict_path = test_data_dir / "domain_dictionary.json"
    with open(dict_path, "w", encoding="utf-8") as f:
        json.dump(domain_dict, f, ensure_ascii=False, indent=2)
    
    return str(dict_path)

