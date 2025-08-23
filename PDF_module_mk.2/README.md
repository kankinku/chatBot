# PDF 질문 답변 시스템 (PDF QA Module)

로컬 LLM/SLM을 사용하는 PDF 기반 질문 답변 시스템입니다. TypeScript/Django 웹서비스에 모듈로 통합 가능하며, API 의존성 없이 완전히 로컬에서 동작합니다.

## ✨ 주요 기능

### 🔍 **지능형 질문 분석**
- 자연어 질문의 유형 자동 분류 (사실형, 개념형, 비교형 등)
- 키워드 및 개체명 자동 추출
- 이전 대화 맥락을 고려한 질문 의도 분석

### 📚 **고성능 문서 검색**
- 다중 PDF 라이브러리를 통한 강건한 텍스트 추출
- 하이브리드 벡터 저장소 (FAISS + ChromaDB)
- 의미적 검색과 키워드 검색의 최적 조합

### 🤖 **로컬 LLM 기반 답변 생성**
- Ollama, HuggingFace, llama.cpp 지원
- 한국어 특화 모델 최적화
- 이전 대화를 고려한 자연스러운 답변

### 📊 **종합적 성능 평가**
- ROUGE, BLEU, BERTScore 등 다양한 메트릭
- 질문 분석 및 답변 생성 품질 평가
- 자동 개선점 도출 및 제안

### 🌐 **완전한 API 인터페이스**
- TypeScript 클라이언트 라이브러리
- Django 연동 모듈
- React/Vue 컴포저블 함수

## 🏗️ 시스템 구조

```
PDF_module_mk.2/
├── core/                    # 핵심 모듈
│   ├── pdf_processor.py     # PDF 텍스트 추출 및 임베딩
│   ├── vector_store.py      # 벡터 저장소 관리 (FAISS + ChromaDB)
│   ├── question_analyzer.py # 질문 분석 및 컨텍스트 관리
│   ├── answer_generator.py  # 로컬 LLM 기반 답변 생성
│   └── evaluator.py        # 종합적 성능 평가
├── api/                     # API 인터페이스
│   ├── endpoints.py         # FastAPI 엔드포인트
│   ├── typescript_client.ts # TypeScript 클라이언트
│   └── django_client.py     # Django 연동 모듈
├── docs/                    # 문서
│   ├── FINETUNING_GUIDE.md  # 모델 파인튜닝 가이드
│   └── IMPROVEMENTS.md      # 개선사항 및 한계점 분석
├── main.py                  # 메인 실행 파일
├── requirements.txt         # 의존성 목록
└── README.md               # 프로젝트 문서
```

## 📁 PDF 파일 관리

시스템은 PDF 파일을 체계적으로 관리할 수 있는 폴더 구조를 제공합니다:

```
data/
├── pdfs/                    # PDF 파일 저장소
│   ├── academic/           # 학술 자료
│   ├── manuals/           # 매뉴얼 및 가이드
│   ├── reports/           # 보고서
│   └── misc/              # 기타 문서
├── vector_store/          # 벡터 데이터 (자동 생성)
└── conversation_history/  # 대화 기록
```

**PDF 추가 방법:**
```bash
# 대화형 모드에서
python main.py --mode interactive
질문: /add C:\path\to\your\document.pdf

# 명령줄에서 (자동으로 data/pdfs/misc/로 복사됨)
python main.py --mode interactive --model-name mistral --pdf document.pdf
```

**관리 명령어:**
- `/pdfs`: 저장된 PDF 목록 조회
- `/categories`: 카테고리 및 저장소 정보
- `/add <경로>`: PDF 파일 추가

상세한 PDF 관리 가이드는 [`docs/PDF_MANAGEMENT_GUIDE.md`](docs/PDF_MANAGEMENT_GUIDE.md)를 참조하세요.

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd PDF_module_mk.2

# Python 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 로컬 모델 설정

**Ollama 사용 (권장)**
```bash
# Ollama 설치: https://ollama.ai/
# 한국어 최적화 모델 다운로드
ollama pull mistral
ollama pull llama2:7b
```

**HuggingFace 모델 사용**
```bash
# GPU 메모리 8GB 이상 권장
# 자동으로 모델 다운로드됨
```

### 3. 시스템 실행

**대화형 모드 (추천)**
```bash
# PDF 파일과 함께 대화형 모드 시작 (한국어 최적화 모델)
python main.py --mode interactive --model-name mistral --pdf sample.pdf

# 기본 한국어 최적화 모드
python main.py --mode interactive --model-name mistral
```

**API 서버 모드**
```bash
# FastAPI 서버 시작 (한국어 최적화 모델)
python main.py --mode server --model-name mistral --port 8000

# API 문서 확인: http://localhost:8000/docs
```

**단일 처리 모드**
```bash
# 특정 PDF와 질문 처리 (한국어 최적화 모델)
python main.py --mode process --model-name mistral --pdf document.pdf --question "이 문서의 주요 내용은 무엇인가요?"
```

## 💻 사용 예시

### Python에서 직접 사용

```python
from main import PDFQASystem

# 시스템 초기화 (한국어 최적화 모델)
system = PDFQASystem(model_type="ollama", model_name="mistral")
system.initialize_components()

# PDF 처리
result = system.process_pdf("document.pdf")
print(f"처리 완료: {result['total_chunks']}개 청크")

# 질문하기
answer = system.ask_question("이 문서의 핵심 내용은 무엇인가요?")
print(f"답변: {answer['answer']}")
print(f"신뢰도: {answer['confidence_score']:.2f}")
```

### TypeScript에서 사용

```typescript
import { PDFQAClient } from './api/typescript_client';

const client = new PDFQAClient({ 
  baseURL: 'http://localhost:8000' 
});

// PDF 업로드
const uploadResult = await client.uploadPDF(file);
console.log(`업로드 완료: ${uploadResult.pdf_id}`);

// 질문하기
const answer = await client.askQuestion({
  question: '이 문서의 주요 개념은 무엇인가요?',
  pdf_id: uploadResult.pdf_id
});

console.log(`답변: ${answer.answer}`);
console.log(`신뢰도: ${answer.confidence_score}`);
```

### Django에서 사용

```python
# views.py
from api.django_client import PDFQAClient

def ask_question_view(request):
    client = PDFQAClient()
    
    result = client.ask_question(
        question=request.POST['question'],
        pdf_id=request.POST['pdf_id']
    )
    
    return JsonResponse(result)

# urls.py
from api.django_client import create_django_urls

urlpatterns = [
    path('api/pdfqa/', include(create_django_urls())),
]
```

## 🔧 고급 설정

### 모델 설정 변경

```python
# 다양한 모델 타입 지원
system = PDFQASystem(
    model_type="huggingface",
    model_name="beomi/KoAlpaca-Polyglot-5.8B",
    embedding_model="jhgan/ko-sroberta-multitask"
)
```

### 생성 파라미터 조정

```python
from core.answer_generator import GenerationConfig

config = GenerationConfig(
    max_length=1024,      # 최대 생성 길이
    temperature=0.7,      # 창의성 조절 (0.1-1.0)
    top_p=0.9,           # 확률 임계값
    top_k=50             # 상위 k개 토큰만 고려
)
```

## 📈 성능 평가

### 자동 평가 실행

```python
from core.evaluator import PDFQAEvaluator

evaluator = PDFQAEvaluator()

# 시스템 성능 평가
evaluation = evaluator.evaluate_system(
    question_analysis_results,
    answer_generation_results
)

print(f"전체 시스템 점수: {evaluation.overall_system_score:.3f}")
print("개선 제안:")
for suggestion in evaluation.improvement_suggestions:
    print(f"- {suggestion}")
```

### 벤치마크 테스트

```bash
# 평가용 데이터셋으로 성능 측정
python -m core.evaluator --benchmark-data ./data/test_qa_pairs.json
```

## 🎯 모델 파인튜닝

자세한 파인튜닝 가이드는 [`docs/FINETUNING_GUIDE.md`](docs/FINETUNING_GUIDE.md)를 참조하세요.

### 임베딩 모델 파인튜닝

```python
from core.pdf_processor import prepare_training_data

# 훈련 데이터 준비
training_data = prepare_training_data(pdf_chunks, qa_pairs)

# 파인튜닝 실행 (상세 코드는 가이드 참조)
finetuned_model = finetune_embedding_model(training_data)
```

### 생성 모델 LoRA 파인튜닝

```python
# QLoRA를 사용한 메모리 효율적 파인튜닝
model = setup_qlora_model("beomi/KoAlpaca-Polyglot-5.8B")
trainer = setup_trainer(model, training_data)
trainer.train()
```

## 🔍 시스템 분석 및 개선

현재 시스템의 한계점과 개선 방안은 [`docs/IMPROVEMENTS.md`](docs/IMPROVEMENTS.md)에서 확인할 수 있습니다.

### 주요 개선 필요 사항

1. **텍스트 추출 고도화**: OCR 통합, 구조화된 데이터 처리
2. **AI 기반 질문 분석**: 규칙 기반에서 ML 모델로 전환
3. **다중 모델 검색**: 앙상블 검색으로 정확도 향상
4. **확장성 개선**: 분산 처리 및 데이터베이스 최적화

## 🛠️ 개발 로드맵

### Phase 1: 기반 개선 (1-2개월)
- [ ] OCR 기능 통합
- [ ] 구조화된 데이터 추출
- [ ] 에러 처리 강화
- [ ] 캐싱 시스템

### Phase 2: 고급 기능 (2-3개월)
- [ ] AI 기반 질문 분석
- [ ] 다중 모델 검색
- [ ] 시각적 콘텐츠 이해
- [ ] 실시간 성능 모니터링

### Phase 3: 확장성 (3-4개월)
- [ ] 분산 처리 아키텍처
- [ ] 데이터베이스 최적화
- [ ] 자동 파인튜닝
- [ ] API 생태계 구축

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 지원 및 문의

- 이슈 리포트: GitHub Issues
- 문서: [`docs/`](docs/) 디렉토리
- 예제: [`examples/`](examples/) 디렉토리 (추후 추가)

## 🔗 관련 링크

- [Ollama](https://ollama.ai/) - 로컬 LLM 실행 도구
- [HuggingFace](https://huggingface.co/) - 오픈소스 ML 모델 허브
- [FAISS](https://github.com/facebookresearch/faiss) - 고성능 벡터 검색
- [ChromaDB](https://www.trychroma.com/) - 벡터 데이터베이스

---

**⚡ 빠른 테스트:**
```bash
# 시스템이 정상 작동하는지 확인 (한국어 최적화 모델)
python main.py --mode interactive --model-name mistral
```
