import os
import pytest
from src.scenario_inference.service import InferenceService
from src.scenario_inference.schemas import InferenceRequest

TEST_DB_PATH = "./test_data/test_traces.db"

@pytest.fixture
def service():
    # Setup: DB 파일 삭제
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except:
            pass
            
    # 서비스 초기화
    svc = InferenceService(db_path=TEST_DB_PATH)
    yield svc
    
    # Teardown: DB 파일 삭제
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except:
            pass

def test_inference_service_direct_call(service):
    """서버 없이 라이브러리로 직접 호출 테스트"""
    
    # 2. 요청 생성
    req = InferenceRequest(
        query="금리가 인상되면 성장주는?",
        budget_max_tests=5,
        include_trace=True
    )
    
    # 3. 실행
    response = service.infer(req)
    
    # 4. 검증
    print(f"Trace ID: {response.trace_id}")
    print(f"Scenarios: {len(response.scenarios)}")
    
    assert response.trace_id is not None
    assert response.query == req.query
    assert response.processing_time_seconds > 0
    
    if response.scenarios:
        top = response.scenarios[0]
        assert top.confidence >= 0.0
