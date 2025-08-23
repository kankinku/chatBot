# PDF 파일 관리 가이드

이 문서는 PDF QA 시스템에서 PDF 파일을 효율적으로 관리하는 방법을 설명합니다.

## 📁 폴더 구조

시스템을 실행하면 다음과 같은 폴더 구조가 자동으로 생성됩니다:

```
PDF_module_mk.2/
├── data/
│   ├── pdfs/                    # PDF 파일 저장소
│   │   ├── academic/            # 학술 자료
│   │   ├── manuals/            # 매뉴얼 및 가이드
│   │   ├── reports/            # 보고서
│   │   └── misc/               # 기타 문서
│   ├── vector_store/           # 벡터 데이터 (자동 생성)
│   ├── conversation_history/   # 대화 기록
│   └── temp/                   # 임시 파일
```

## 🚀 PDF 파일 추가하기

### 1. 대화형 모드에서 추가

```bash
# 시스템 실행
python main.py --mode interactive

# 대화형 모드에서 PDF 추가
질문: /add C:\Users\user\Documents\my_document.pdf
```

**단계별 과정:**
1. 사용 가능한 카테고리 목록 표시
2. 카테고리 선택 (숫자 입력 또는 새 카테고리 생성)
3. 파일 자동 복사 및 저장
4. 즉시 처리 여부 선택

### 2. 명령줄에서 직접 추가

```bash
# PDF와 함께 시스템 시작 (자동으로 data/pdfs/misc/ 폴더로 복사됨)
python main.py --mode interactive --pdf my_document.pdf

# 단일 처리 모드
python main.py --mode process --pdf document.pdf --question "이 문서의 주요 내용은?"
```

### 3. 수동으로 폴더에 복사

```bash
# 직접 카테고리 폴더에 복사
cp my_document.pdf data/pdfs/academic/
cp user_manual.pdf data/pdfs/manuals/
```

## 📋 PDF 파일 관리 명령어

### 대화형 모드 명령어

| 명령어 | 설명 | 예시 |
|--------|------|------|
| `/pdfs` | 저장된 PDF 목록 조회 | `/pdfs` |
| `/categories` | 카테고리 및 저장소 정보 | `/categories` |
| `/add <경로>` | PDF 파일 추가 | `/add C:\docs\report.pdf` |
| `/status` | 시스템 상태 조회 | `/status` |

### 사용 예시

```
질문: /pdfs

저장된 PDF 파일 (3개):
------------------------------------------------------------
 1. 연구보고서_2024.pdf
    카테고리: academic
    크기: 2.5MB
    수정일: 2024-01-15 14:30:22

 2. 사용자매뉴얼.pdf
    카테고리: manuals
    크기: 1.2MB
    수정일: 2024-01-14 09:15:30

 3. 분석자료.pdf
    카테고리: reports
    크기: 3.8MB
    수정일: 2024-01-13 16:45:10
```

## 🗂️ 카테고리 관리

### 기본 카테고리

- **academic**: 학술 논문, 연구 자료
- **manuals**: 사용자 매뉴얼, 가이드 문서  
- **reports**: 보고서, 분석 자료
- **misc**: 기타 문서

### 새 카테고리 생성

1. **대화형 모드에서:**
   ```
   질문: /add my_document.pdf
   
   사용 가능한 카테고리:
     1. academic
     2. manuals
     3. reports
     4. misc
     5. 새 카테고리 생성
   
   카테고리를 선택하세요: 5
   새 카테고리 이름: contracts
   ```

2. **Python 코드로:**
   ```python
   from utils.file_manager import setup_pdf_storage
   
   manager = setup_pdf_storage()
   manager.create_category("새카테고리명")
   ```

## 💡 모범 사례

### 1. 파일 명명 규칙

```
좋은 예:
- 연구보고서_AI_2024.pdf
- 사용자매뉴얼_v2.1.pdf
- 분기별_실적보고서_Q1.pdf

피해야 할 예:
- 문서1.pdf
- 임시.pdf
- untitled.pdf
```

### 2. 카테고리 분류

```
academic/
├── 논문_인공지능_transformer.pdf
├── 연구보고서_딥러닝_2024.pdf
└── 학회발표자료_NLP.pdf

manuals/
├── API_사용법_v1.0.pdf
├── 설치가이드_윈도우.pdf
└── 문제해결_FAQ.pdf

reports/
├── 월간보고서_2024_01.pdf
├── 성과분석_Q4.pdf
└── 시장조사_결과.pdf
```

### 3. 정기적인 정리

```python
# 저장소 정보 확인
질문: /categories

사용 가능한 카테고리:
  - academic: 5개 파일
  - manuals: 3개 파일
  - reports: 8개 파일
  - misc: 12개 파일

저장소 정보:
  - 전체 파일 수: 28개
  - 전체 크기: 156.7MB
  - 저장 위치: C:\Users\user\Desktop\chatBot\PDF_module_mk.2\data\pdfs
```

## 🔧 고급 관리 기능

### 1. Python API 사용

```python
from utils.file_manager import PDFFileManager

# 파일 매니저 생성
manager = PDFFileManager()

# PDF 저장
result = manager.save_pdf(
    source_path="원본경로.pdf",
    category="academic",
    custom_name="새파일명.pdf"
)

# 파일 목록 조회
pdfs = manager.list_pdfs(category="academic")

# 파일 삭제
manager.delete_pdf("파일명.pdf", category="academic")

# 저장소 정보
info = manager.get_storage_info()
```

### 2. 배치 작업

```python
import os
from utils.file_manager import setup_pdf_storage

def batch_import_pdfs(source_folder, category="misc"):
    """폴더의 모든 PDF를 배치로 가져오기"""
    manager = setup_pdf_storage()
    
    for filename in os.listdir(source_folder):
        if filename.lower().endswith('.pdf'):
            source_path = os.path.join(source_folder, filename)
            try:
                result = manager.save_pdf(source_path, category)
                print(f"✓ {filename} 저장 완료")
            except Exception as e:
                print(f"✗ {filename} 저장 실패: {e}")

# 사용 예시
batch_import_pdfs("C:/Downloads/pdfs", "reports")
```

## 🚨 주의사항

### 1. 파일 크기 제한

- 개별 파일: 최대 50MB (설정에서 변경 가능)
- 전체 저장소: 디스크 공간에 따라 제한
- 대용량 파일은 처리 시간이 오래 걸릴 수 있음

### 2. 지원 형식

- ✅ **지원**: `.pdf` 파일만
- ❌ **미지원**: `.doc`, `.docx`, `.txt`, 이미지 파일 등

### 3. 파일 안전성

- 원본 파일은 복사되어 저장 (원본 파일 안전)
- 중복 파일명 자동 처리 (`파일명_1.pdf`, `파일명_2.pdf`)
- 파일 무결성 검증 (MD5 해시)

### 4. 백업 권장사항

```bash
# 전체 데이터 폴더 백업
cp -r data/ backup/data_$(date +%Y%m%d)/

# 특정 카테고리만 백업
cp -r data/pdfs/academic/ backup/academic_$(date +%Y%m%d)/
```

## 🔍 문제 해결

### Q: PDF 파일이 추가되지 않아요

**A:** 다음을 확인해보세요:
1. 파일 경로가 올바른지 확인
2. `.pdf` 확장자인지 확인
3. 파일 크기가 50MB 이하인지 확인
4. 파일이 손상되지 않았는지 확인

### Q: 처리된 PDF 파일을 삭제하면 어떻게 되나요?

**A:** 
- PDF 파일 삭제: 질문 답변 불가능
- 벡터 데이터는 남아있어서 시스템 오류 가능
- 전체 재처리 권장

### Q: 대용량 PDF 처리가 너무 느려요

**A:** 
1. 하드웨어 사양 확인 (RAM, CPU)
2. 청크 크기 조정 (기본 512 토큰)
3. 병렬 처리 고려
4. SSD 사용 권장

이 가이드를 참고하여 PDF 파일을 체계적으로 관리하고 효율적으로 활용해보세요!
