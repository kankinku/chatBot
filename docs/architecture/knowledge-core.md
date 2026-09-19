# Knowledge Core Architecture

`ontology_system_v13`에서 검증한 최소 graph/ontology 코어를 canonical repository에 통합한 구조다.

## 책임

Knowledge Core는 문서 RAG와 별도 책임을 가진다.

1. **Extraction**: 텍스트를 fragment → entity → relation 후보로 변환
2. **Validation**: schema, sign, semantic, confidence gate 적용
3. **Domain Update**: 검증된 relation을 domain graph에 반영
4. **Reasoning**: graph retrieval → path fusion → path reasoning → conclusion 생성

RAG는 문서 근거 검색을 담당하고, Knowledge Core는 구조화된 entity/relation과 관계 경로 추론을 담당한다. 두 계층을 하나의 거대한 pipeline class로 합치지 않는다.

## Canonical layout

```text
src/chatbot/knowledge/
  bootstrap.py
  settings.py
  extraction/
  validation/
  domain/
  reasoning/
  storage/
  llm/
  shared/

config/ontology/
  entity_types.yaml
  relation_types.yaml
  alias_dictionary.yaml
  validation_schema.yaml
  static_domain.yaml
  infrastructure.yaml

data/ontology/
  domain/entities.json
  domain/relations.json
  samples/sample_documents.json

scripts/knowledge_core_demo.py
```

`ontology_system_v13/` wrapper는 더 이상 존재하지 않는다.

## Configuration

기본 project root는 repository root로 계산한다. 다른 배치 위치에서는 `CHATBOT_PROJECT_ROOT` 환경변수로 config/data root를 지정할 수 있다.

Storage backend 기본값은 `inmemory`다. Neo4j를 사용할 경우 root `requirements.txt`의 Neo4j driver를 설치한 상태에서 `config/ontology/infrastructure.yaml`의 backend를 변경하고 `NEO4J_PASSWORD`를 환경변수로 제공한다. 비밀번호 기본값은 source에 저장하지 않는다.

LLM backend는 Ollama 또는 mock adapter를 사용할 수 있다. Knowledge Core는 LLM 없이 rule-based 경로로도 실행 가능하도록 유지한다.

## Runtime entry point

```bash
python3 scripts/knowledge_core_demo.py --help
python3 scripts/knowledge_core_demo.py --no_llm
```

샘플 입력은 `data/ontology/samples/sample_documents.json`을 사용한다.

## Storage boundary

`GraphRepository`가 storage abstraction이다.

- `InMemoryGraphRepository`: 테스트·개발 기본값
- `Neo4jGraphRepository`: optional production graph backend
- `KGTransactionManager`: graph mutation transaction boundary
- `DomainKGAdapter`: domain relation namespace와 repository 사이 adapter

Neo4j query data 값은 parameter binding을 사용하고 label/relation identifier는 별도 quoting을 거친다.

## Evolution boundary

v12의 대형 시스템을 그대로 되살리지 않는다. 다음 메커니즘만 필요할 때 독립적으로 재구현한다.

- Evidence binding / provenance
- Incremental + idempotent ingestion
- Dependency invalidation / blast radius
- Replay / evaluation
- Scenario / regime

원본 reference는 `docs/history/retired-snapshot-inventory.md`와 archive refs에 보존되어 있다.
