"""
법률 검색 및 유추 모듈

이 패키지는 법률 문서에서 관련 조문을 정확하게 검색하고,
사용자의 모호한 질문에 대해 필요한 법률을 유추하는 기능을 제공합니다.

주요 구성 요소:
- LegalSchema: 법률 문서 표준 스키마 및 정규화
- LegalChunker: 법률 문서 청킹 (조문 기준)
- LegalIndexer: 하이브리드 인덱싱 (벡터 + BM25)
- LegalRetriever: 검색 최적화 (RRF + MMR)
- LegalReranker: 크로스엔코더 재순위화
- LawInference: 법률 유추 및 검증
- LegalRouter: 통합 라우팅 및 모드 관리

사용 예시:
    from core.legal import LegalRouter, LegalMode
    
    # 라우터 초기화
    router = LegalRouter()
    router.initialize_components()
    
    # 법률 검색
    response = router.route_legal_query(
        "근로자의 휴가 권리는?", 
        mode=LegalMode.ACCURACY
    )
    
    print(f"신뢰도: {response.confidence}")
    for result in response.results:
        print(f"- {result.chunk.metadata.law_title} 제{result.chunk.metadata.article_no}조")
"""

from .legal_schema import (
    LegalDocument, LegalChunk, LegalMetadata, LegalNormalizer,
    LegalDocumentType
)
from .legal_chunker import LegalChunker, ChunkingConfig
from .legal_indexer import LegalIndexer
from .legal_retriever import LegalRetriever, SearchConfig, SearchResult
from .legal_reranker import LegalReranker, RerankConfig, RerankResult
from .law_inference import (
    LawInference, InferenceConfig, InferenceResult, LawCandidate
)
from .legal_router import (
    LegalRouter, LegalRouterConfig, LegalResponse, LegalMode
)

# 버전 정보
__version__ = "1.0.0"

# 주요 클래스 및 설정
__all__ = [
    # 스키마 및 데이터 타입
    'LegalDocument',
    'LegalChunk', 
    'LegalMetadata',
    'LegalNormalizer',
    'LegalDocumentType',
    
    # 청킹
    'LegalChunker',
    'ChunkingConfig',
    
    # 인덱싱
    'LegalIndexer',
    
    # 검색
    'LegalRetriever',
    'SearchConfig',
    'SearchResult',
    
    # 재순위화
    'LegalReranker',
    'RerankConfig', 
    'RerankResult',
    
    # 법률 유추
    'LawInference',
    'InferenceConfig',
    'InferenceResult',
    'LawCandidate',
    
    # 라우팅 (메인 인터페이스)
    'LegalRouter',
    'LegalRouterConfig',
    'LegalResponse',
    'LegalMode',
    
    # 버전
    '__version__'
]

# 기본 설정값
DEFAULT_EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
DEFAULT_CROSS_ENCODER = "BAAI/bge-reranker-v2-m3"
DEFAULT_VECTOR_STORE_PATH = "vector_store/legal"

# 설정 헬퍼 함수
def create_default_router(embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> LegalRouter:
    """기본 설정으로 법률 라우터 생성"""
    config = LegalRouterConfig(
        default_mode=LegalMode.ACCURACY,
        accuracy_threshold=0.7,
        enable_reranking=True,
        enable_inference=True,
        conservative_response=True
    )
    
    return LegalRouter(config=config, embedding_model=embedding_model)

def create_optimized_router(embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> LegalRouter:
    """최적화된 설정으로 법률 라우터 생성"""
    config = LegalRouterConfig(
        default_mode=LegalMode.ACCURACY,
        accuracy_threshold=0.75,  # 더 높은 정확도 요구
        exploration_threshold=0.35,  # 탐색 모드 임계값 조정
        max_results_accuracy=3,
        max_results_exploration=8,
        enable_reranking=True,
        enable_inference=True,
        conservative_response=True
    )
    
    return LegalRouter(config=config, embedding_model=embedding_model)

def get_supported_domains():
    """지원하는 법률 도메인 목록 반환"""
    return [
        'labor',              # 근로/노동
        'privacy',            # 개인정보보호  
        'environment',        # 환경
        'commerce',           # 상거래/공정거래
        'criminal',           # 형사
        'civil',              # 민사
        'administrative',     # 행정
        'intellectual_property',  # 지적재산권
        'tax',                # 세무
        'construction'        # 건설
    ]

# 모듈 정보
__author__ = "Legal AI System"
__description__ = "한국어 법률 문서 검색 및 유추 시스템"
__license__ = "MIT"
