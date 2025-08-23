# 모델 파인튜닝 가이드

이 문서는 PDF QA 시스템의 성능 향상을 위한 모델 파인튜닝 방법을 설명합니다.

## 목차

1. [파인튜닝 개요](#파인튜닝-개요)
2. [데이터 준비](#데이터-준비)
3. [임베딩 모델 파인튜닝](#임베딩-모델-파인튜닝)
4. [생성 모델 파인튜닝](#생성-모델-파인튜닝)
5. [성능 평가 및 최적화](#성능-평가-및-최적화)
6. [배포 및 모니터링](#배포-및-모니터링)

## 파인튜닝 개요

### 왜 파인튜닝이 필요한가?

1. **도메인 특화**: 특정 분야의 전문 용어와 개념을 더 잘 이해
2. **한국어 성능**: 한국어 문맥과 뉘앙스를 더 정확히 파악
3. **검색 정확도**: 질문과 관련 문서 간의 의미적 매칭 향상
4. **답변 품질**: 자연스럽고 정확한 답변 생성

### 파인튜닝 대상 모델

1. **임베딩 모델**: 텍스트의 의미적 표현 생성 (검색 성능 향상)
2. **생성 모델**: 답변 텍스트 생성 (답변 품질 향상)

## 데이터 준비

### 1. 질문-답변 쌍 수집

```python
# 예시 데이터 형식
qa_data = [
    {
        "question": "인공지능의 정의는 무엇인가요?",
        "answer": "인공지능(AI)은 인간의 지능을 모방하여 학습, 추론, 인식 등의 인지 기능을 수행하는 컴퓨터 시스템을 의미합니다.",
        "relevant_chunks": ["chunk_001", "chunk_002"],
        "domain": "컴퓨터과학",
        "difficulty": "easy"
    },
    # ... 더 많은 데이터
]
```

### 2. 데이터 품질 기준

- **최소 데이터량**: 100-500개의 고품질 QA 쌍
- **다양성**: 다양한 질문 유형과 난이도 포함
- **정확성**: 전문가 검증을 거친 정확한 답변
- **일관성**: 일관된 답변 스타일과 형식

### 3. 데이터 전처리

```python
from core.pdf_processor import PDFProcessor
from core.answer_generator import prepare_finetuning_data

# 1. PDF 처리 및 청크 생성
processor = PDFProcessor()
chunks, metadata = processor.process_pdf("training_document.pdf")

# 2. 파인튜닝 데이터 준비
training_data = prepare_finetuning_data(qa_data, chunks)

print(f"총 훈련 예제: {len(training_data['training_data'])}개")
print(f"평균 컨텍스트 길이: {training_data['statistics']['avg_context_length']:.1f}")
```

## 임베딩 모델 파인튜닝

### 1. 대조 학습 (Contrastive Learning)

임베딩 모델은 관련 있는 텍스트는 가깝게, 관련 없는 텍스트는 멀게 표현하도록 학습합니다.

```python
# 파인튜닝 스크립트 예시
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def finetune_embedding_model():
    """임베딩 모델 파인튜닝"""
    
    # 1. 기본 모델 로드
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    
    # 2. 훈련 데이터 준비
    train_examples = []
    
    # 긍정 쌍 (질문-관련 텍스트)
    for qa_pair in qa_data:
        for chunk_id in qa_pair['relevant_chunks']:
            chunk_text = get_chunk_text(chunk_id)
            train_examples.append(InputExample(
                texts=[qa_pair['question'], chunk_text],
                label=1.0  # 유사함
            ))
    
    # 부정 쌍 (질문-무관한 텍스트)
    for qa_pair in qa_data:
        negative_chunks = get_negative_chunks(qa_pair['relevant_chunks'])
        for chunk_text in negative_chunks[:2]:  # 2개씩만
            train_examples.append(InputExample(
                texts=[qa_pair['question'], chunk_text],
                label=0.0  # 유사하지 않음
            ))
    
    # 3. 데이터로더 생성
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    
    # 4. 손실 함수 정의
    train_loss = losses.CosineSimilarityLoss(model)
    
    # 5. 파인튜닝 실행
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=100,
        output_path='./models/finetuned_embedding'
    )
    
    return model

# 파인튜닝 실행
finetuned_model = finetune_embedding_model()
```

### 2. 파인튜닝 설정

```python
# 하이퍼파라미터 설정
EMBEDDING_FINETUNING_CONFIG = {
    "learning_rate": 2e-5,
    "batch_size": 16,
    "epochs": 3,
    "warmup_steps": 100,
    "max_seq_length": 512,
    "evaluation_steps": 500
}
```

### 3. 성능 평가

```python
def evaluate_embedding_model(model, test_data):
    """임베딩 모델 성능 평가"""
    
    # 검색 성능 평가
    retrieval_scores = []
    
    for qa_pair in test_data:
        question_embedding = model.encode([qa_pair['question']])
        
        # 모든 청크와 유사도 계산
        chunk_similarities = []
        for chunk in all_chunks:
            chunk_embedding = model.encode([chunk.content])
            similarity = cosine_similarity(question_embedding, chunk_embedding)[0][0]
            chunk_similarities.append((chunk.chunk_id, similarity))
        
        # 상위 k개 검색
        top_k = sorted(chunk_similarities, key=lambda x: x[1], reverse=True)[:5]
        retrieved_ids = [chunk_id for chunk_id, _ in top_k]
        
        # 성능 메트릭 계산
        metrics = calculate_retrieval_metrics(
            qa_pair['relevant_chunks'], 
            retrieved_ids
        )
        retrieval_scores.append(metrics)
    
    # 평균 성능
    avg_precision = sum(score['precision'] for score in retrieval_scores) / len(retrieval_scores)
    avg_recall = sum(score['recall'] for score in retrieval_scores) / len(retrieval_scores)
    
    print(f"평균 Precision: {avg_precision:.3f}")
    print(f"평균 Recall: {avg_recall:.3f}")
    
    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    }
```

## 생성 모델 파인튜닝

### 1. LoRA (Low-Rank Adaptation) 사용

메모리 효율적인 파인튜닝 방법으로, 큰 모델을 적은 리소스로 학습할 수 있습니다.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

def setup_lora_model(model_name: str):
    """LoRA 설정으로 모델 준비"""
    
    # 1. 기본 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 2. LoRA 설정
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # 랭크 (낮을수록 파라미터 적음)
        lora_alpha=32,  # 스케일링 팩터
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]  # 적용할 레이어
    )
    
    # 3. LoRA 모델 생성
    peft_model = get_peft_model(model, lora_config)
    
    print(f"훈련 가능한 파라미터: {peft_model.num_parameters()}")
    
    return tokenizer, peft_model

# 모델 설정
tokenizer, model = setup_lora_model("beomi/KoAlpaca-Polyglot-5.8B")
```

### 2. 훈련 데이터 형식

```python
def prepare_generation_training_data(qa_pairs, chunks):
    """생성 모델용 훈련 데이터 준비"""
    
    training_texts = []
    
    for qa_pair in qa_pairs:
        # 관련 청크들로 컨텍스트 구성
        context_parts = []
        for chunk_id in qa_pair['relevant_chunks']:
            chunk = get_chunk_by_id(chunk_id, chunks)
            context_parts.append(chunk.content)
        
        context = "\n\n".join(context_parts)
        
        # 프롬프트 템플릿
        prompt = f"""다음은 문서의 내용입니다:

{context}

질문: {qa_pair['question']}

위 문서 내용을 바탕으로 정확하고 도움이 되는 답변을 제공해주세요.

답변: {qa_pair['answer']}<|endoftext|>"""

        training_texts.append(prompt)
    
    return training_texts
```

### 3. 훈련 스크립트

```python
from transformers import Trainer, TrainingArguments

def finetune_generation_model():
    """생성 모델 파인튜닝"""
    
    # 1. 훈련 데이터 준비
    training_texts = prepare_generation_training_data(qa_data, chunks)
    
    # 2. 토크나이제이션
    def tokenize_function(examples):
        return tokenizer(
            examples,
            truncation=True,
            padding=True,
            max_length=1024,
            return_tensors="pt"
        )
    
    train_encodings = tokenize_function(training_texts)
    
    # 3. 훈련 인자 설정
    training_args = TrainingArguments(
        output_dir='./models/finetuned_generator',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,  # 메모리에 맞게 조정
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=True,  # 메모리 절약
        dataloader_pin_memory=False
    )
    
    # 4. 트레이너 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_encodings,
        tokenizer=tokenizer
    )
    
    # 5. 훈련 실행
    trainer.train()
    
    # 6. 모델 저장
    trainer.save_model()
    tokenizer.save_pretrained('./models/finetuned_generator')
    
    return model, tokenizer
```

### 4. QLoRA (Quantized LoRA) 사용

더 적은 메모리로 파인튜닝하기 위한 양자화 기법:

```python
from transformers import BitsAndBytesConfig

def setup_qlora_model(model_name: str):
    """QLoRA 설정으로 모델 준비"""
    
    # 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 모델 로드 (4bit 양자화)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRA 설정 (동일)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    
    return model
```

## 성능 평가 및 최적화

### 1. 자동 평가

```python
from core.evaluator import PDFQAEvaluator

def evaluate_finetuned_model():
    """파인튜닝된 모델 평가"""
    
    evaluator = PDFQAEvaluator()
    
    # 테스트 데이터로 평가
    test_questions = [qa['question'] for qa in test_data]
    test_answers = [qa['answer'] for qa in test_data]
    
    # 모델로 답변 생성
    generated_answers = []
    for question in test_questions:
        # 답변 생성 로직
        answer = generate_answer(question)
        generated_answers.append(answer)
    
    # 평가 실행
    evaluation_result = evaluator.evaluate_system(
        test_questions,
        generated_answers,
        test_answers
    )
    
    print(f"전체 시스템 점수: {evaluation_result.overall_system_score:.3f}")
    
    return evaluation_result
```

### 2. A/B 테스트

```python
def ab_test_models(model_a, model_b, test_questions):
    """두 모델 간 A/B 테스트"""
    
    results_a = []
    results_b = []
    
    for question in test_questions:
        # 모델 A 결과
        answer_a = model_a.generate_answer(question)
        results_a.append(answer_a)
        
        # 모델 B 결과
        answer_b = model_b.generate_answer(question)
        results_b.append(answer_b)
    
    # 성능 비교
    metrics_a = calculate_metrics(results_a, ground_truth)
    metrics_b = calculate_metrics(results_b, ground_truth)
    
    print("모델 A vs 모델 B 성능 비교:")
    print(f"정확도: {metrics_a['accuracy']:.3f} vs {metrics_b['accuracy']:.3f}")
    print(f"유창성: {metrics_a['fluency']:.3f} vs {metrics_b['fluency']:.3f}")
    
    return metrics_a, metrics_b
```

### 3. 하이퍼파라미터 튜닝

```python
import optuna

def objective(trial):
    """Optuna를 사용한 하이퍼파라미터 최적화"""
    
    # 하이퍼파라미터 샘플링
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    lora_r = trial.suggest_int("lora_r", 4, 16)
    
    # 모델 훈련
    model = train_model_with_params(learning_rate, batch_size, lora_r)
    
    # 성능 평가
    score = evaluate_model(model)
    
    return score

# 최적화 실행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print(f"최적 하이퍼파라미터: {study.best_params}")
```

## 배포 및 모니터링

### 1. 모델 배포

```python
def deploy_finetuned_model():
    """파인튜닝된 모델 배포"""
    
    # 1. 모델 로드
    from core.answer_generator import AnswerGenerator, ModelType
    
    generator = AnswerGenerator(
        model_type=ModelType.HUGGINGFACE,
        model_name="./models/finetuned_generator"
    )
    
    # 2. 모델 검증
    test_question = "테스트 질문입니다."
    test_answer = generator.generate_answer(test_question)
    
    if test_answer.confidence_score > 0.5:
        print("모델 배포 성공")
        return True
    else:
        print("모델 검증 실패")
        return False

# 배포 실행
deploy_finetuned_model()
```

### 2. 성능 모니터링

```python
import time
from collections import defaultdict

class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
    
    def log_request(self, question, answer, response_time, confidence_score):
        """요청 로깅"""
        self.metrics['response_times'].append(response_time)
        self.metrics['confidence_scores'].append(confidence_score)
        self.metrics['question_lengths'].append(len(question))
        self.metrics['answer_lengths'].append(len(answer))
    
    def get_stats(self):
        """통계 조회"""
        import statistics
        
        return {
            'avg_response_time': statistics.mean(self.metrics['response_times']),
            'avg_confidence': statistics.mean(self.metrics['confidence_scores']),
            'total_requests': len(self.metrics['response_times']),
            'uptime': time.time() - self.start_time
        }

# 모니터 사용
monitor = PerformanceMonitor()

# API 엔드포인트에서 사용
def monitored_ask_question(question, pdf_id):
    start_time = time.time()
    
    # 답변 생성
    result = ask_question(question, pdf_id)
    
    # 성능 로깅
    response_time = time.time() - start_time
    monitor.log_request(
        question, 
        result['answer'], 
        response_time, 
        result['confidence_score']
    )
    
    return result
```

## 주의사항 및 팁

### 1. 데이터 품질이 가장 중요

- **고품질 데이터**: 적은 양이라도 정확하고 일관된 데이터가 중요
- **도메인 전문가**: 해당 분야 전문가의 검토를 받은 데이터 사용
- **다양성**: 다양한 질문 패턴과 답변 스타일 포함

### 2. 단계적 접근

1. **베이스라인 설정**: 파인튜닝 전 성능 측정
2. **작은 실험**: 적은 데이터로 개념 검증
3. **점진적 확장**: 데이터와 모델 크기를 점진적으로 확장

### 3. 리소스 관리

- **GPU 메모리**: LoRA/QLoRA 사용으로 메모리 사용량 최적화
- **훈련 시간**: 체크포인트 저장으로 중단 상황 대비
- **모델 크기**: 배포 환경에 맞는 모델 크기 선택

### 4. 성능 측정

- **자동 평가**: BLEU, ROUGE 등 자동 메트릭
- **인간 평가**: 실제 사용자 피드백
- **A/B 테스트**: 실제 서비스에서 성능 비교

이 가이드를 참고하여 도메인에 특화된 고성능 PDF QA 시스템을 구축하세요!
