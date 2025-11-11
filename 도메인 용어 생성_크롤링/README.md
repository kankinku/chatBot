# 정수장 용어 도메인 생성 도구

정수장 운용 및 정수처리 관련 전문 용어를 수집하고 정제하여 JSON 형태의 도메인 데이터를 생성하는 도구입니다.

## 주요 기능

- **한국상하수도협회 용어집 스크래핑**: 웹사이트에서 정수장 관련 용어 자동 수집
- **PDF 문서 텍스트 추출**: 정수처리기준 해설서, K-water 수질항목 백과사전에서 용어 추출
- **데이터 정제 및 통합**: 중복 제거, 용어 정제, 카테고리 분류
- **JSON 형태 출력**: 구조화된 용어 도메인 데이터 생성

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. Python 스크립트 실행:
```bash
python main.py
```

## 파일 구조

- `main.py`: 메인 실행 스크립트
- `scraper_kwwa.py`: 한국상하수도협회 웹 스크래핑
- `pdf_extractor.py`: PDF 문서 텍스트 추출
- `term_processor.py`: 용어 데이터 정제 및 통합
- `requirements.txt`: 필요한 Python 패키지 목록

## 출력 파일

- `water_treatment_terms.json`: 최종 용어 도메인 데이터
- `kwwa_terms.json`: 한국상하수도협회 용어 데이터
- `pdf_terms.json`: PDF 추출 용어 데이터

## 용어 카테고리

- **시설**: 정수장, 정수시설, 정수지, 여과지, 침전지 등
- **공정**: 응집, 침전, 여과, 소독, 정수처리 등
- **약품**: 염소, 오존, 응집제, 소독제 등
- **수질**: 수질, 탁도, pH, 알칼리도, 잔류염소 등
- **운영**: 운영, 관리, 운전, 제어, 모니터링 등
- **설비**: 펌프, 배관, 밸브, 계량기, 설비 등

## 사용 예시

```python
import json

# 생성된 용어 도메인 데이터 로드
with open('water_treatment_terms.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 용어 검색
for term in data['terms']:
    if '정수지' in term['term']:
        print(f"{term['term']}: {term['definition']}")
```

## 주의사항

- 웹 스크래핑 시 요청 간격을 조절하여 서버에 부하를 주지 않도록 합니다
- PDF 파일이 있는 경우에만 PDF 추출이 실행됩니다
- 수집된 데이터의 정확성을 위해 수동 검토가 필요할 수 있습니다
