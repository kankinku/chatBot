# 법률 데이터 디렉토리

이 디렉토리는 법률 문서와 데이터를 저장하는 공간입니다.

## 디렉토리 구조

```
data/legal/
├── pdfs/           # 법률 PDF 문서
│   ├── labor/      # 근로/노동 관련 법률
│   ├── privacy/    # 개인정보보호 관련 법률
│   ├── civil/      # 민법 관련 법률
│   ├── criminal/   # 형법 관련 법률
│   └── admin/      # 행정법 관련 법률
├── json/           # 구조화된 법률 JSON 데이터
│   ├── laws/       # 개별 법률 JSON 파일
│   ├── articles/   # 조문별 JSON 데이터
│   └── metadata/   # 법률 메타데이터
└── README.md       # 이 파일
```

## 지원하는 파일 형식

### PDF 파일
- **위치**: `data/legal/pdfs/`
- **형식**: `.pdf`
- **예시**: 
  - `근로기준법.pdf`
  - `개인정보보호법.pdf`
  - `민법.pdf`

### JSON 파일
- **위치**: `data/legal/json/`
- **형식**: `.json`
- **구조 예시**:
```json
{
  "law_id": "labor_standards_act",
  "law_title": "근로기준법",
  "effective_date": "2021-01-01",
  "articles": [
    {
      "article_no": "제1조",
      "title": "목적",
      "content": "이 법은 헌법에 따라 근로조건의 기준을 정함으로써...",
      "clauses": [
        {
          "clause_no": "제1항",
          "content": "..."
        }
      ]
    }
  ],
  "aliases": ["근기법", "근로기준법"],
  "domain": "labor",
  "source": "국가법령정보센터"
}
```

## 법률 모듈 사용 방법

### 1. PDF 파일 추가
```python
from core.legal import create_optimized_router

# 법률 라우터 생성
router = create_optimized_router()

# PDF 파일 처리 (자동으로 data/legal/pdfs/ 스캔)
router.initialize_components()
```

### 2. JSON 파일 추가
```python
# JSON 법률 데이터 로드
router.load_legal_documents("data/legal/json/")
```

### 3. 법률 검색 사용
```python
from core.legal import LegalMode

# 정확도 중심 검색
response = router.route_legal_query(
    "근로자 휴가 권리에 대해 알려주세요",
    mode=LegalMode.ACCURACY
)

print(f"답변: {response.answer}")
print(f"관련 법률: {[law.law_title for law in response.relevant_laws]}")
```

## 법률 도메인 분류

시스템에서 지원하는 법률 도메인:

- **labor**: 근로/노동법 (근로기준법, 산업안전보건법 등)
- **privacy**: 개인정보보호 (개인정보보호법, 정보통신망법 등)
- **civil**: 민법 (민법, 상법 등)
- **criminal**: 형법 (형법, 형사소송법 등)
- **admin**: 행정법 (행정기본법, 행정절차법 등)
- **construction**: 건설 (건설산업기본법, 건축법 등)
- **environment**: 환경 (환경정책기본법, 대기환경보전법 등)
- **tax**: 세무 (국세기본법, 소득세법 등)
- **intellectual**: 지적재산권 (저작권법, 특허법 등)
- **finance**: 금융 (은행법, 자본시장법 등)

## 주의사항

1. **파일명 규칙**: 한글 법률명을 권장합니다 (예: `근로기준법.pdf`)
2. **인코딩**: JSON 파일은 UTF-8 인코딩으로 저장해주세요
3. **메타데이터**: 가능한 한 상세한 메타데이터를 포함해주세요
4. **업데이트**: 법률 개정 시 `effective_date`를 업데이트해주세요

## 벡터 저장소

처리된 법률 데이터는 다음 위치에 자동 저장됩니다:
- **벡터 인덱스**: `vector_store/legal/`
- **메타데이터**: 법률 문서의 메타데이터와 함께 저장
- **검색 인덱스**: BM25 + 벡터 하이브리드 인덱스 생성

## 문제 해결

### 파일이 인식되지 않는 경우
1. 파일 경로 확인: `data/legal/pdfs/` 또는 `data/legal/json/`
2. 파일 권한 확인
3. 파일 형식 확인 (PDF 또는 JSON)

### 검색 결과가 부정확한 경우
1. 법률 도메인 분류 확인
2. 메타데이터 정확성 검증
3. 임베딩 모델 재처리 (`router.reindex_documents()`)

## 연락처

문제가 발생하면 시스템 관리자에게 문의하세요.
