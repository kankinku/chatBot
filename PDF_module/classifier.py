import json
import os
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EvalPrediction
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }


class JSONClassifier:
    def __init__(self, model_name="klue/roberta-large"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"기기 정보: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"가용 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        self.tokenizer = None
        self.model = None
        self.label_map = {}
        self.label_descriptions = {}
        self.num_labels = 0

    def preprocess_json_content(self, content):
        if isinstance(content, dict):
            parts = []
            for key, value in content.items():
                key_text = key.replace('_', ' ').replace('-', ' ')
                if isinstance(value, (dict, list)):
                    value_text = self.preprocess_json_content(value)
                else:
                    value_text = str(value)
                parts.append(f"{key_text}: {value_text}")
            return ' '.join(parts)
        if isinstance(content, list):
            return ' '.join(self.preprocess_json_content(v) for v in content if v is not None)
        return str(content)

    def create_simple_samples(self, content, label_text, label_idx):
        """각 JSON 파일에서 간단한 샘플들을 생성"""
        samples = []
        
        # 전체 내용으로 기본 샘플 생성
        full_text = self.preprocess_json_content(content)
        samples.append({
            'text': f"문서 제목: {label_text}\n내용: {full_text}",
            'label_text': label_text,
            'label': label_idx,
        })
        
        # JSON의 주요 섹션별로 간단한 샘플 생성 (최대 3-4개 추가)
        if isinstance(content, dict):
            for key, value in list(content.items())[:3]:  # 상위 3개 섹션만
                if isinstance(value, (dict, list)) and len(str(value)) > 100:
                    section_text = self.preprocess_json_content(value)
                    key_formatted = key.replace('_', ' ').replace('-', ' ')
                    samples.append({
                        'text': f"문서: {label_text}\n섹션: {key_formatted}\n내용: {section_text}",
                        'label_text': label_text,
                        'label': label_idx,
                    })
        
        return samples[:3]  # 클래스당 최대 3개로 제한 (더 균등하게)

    def load_json_data(self, json_dir):
        print("JSON 데이터 로드 중...")
        data = []
        for idx, filename in enumerate(sorted(os.listdir(json_dir))):
            if not filename.endswith('.json'):
                continue
            label_text = filename.replace('.json', '').replace('_', ' ').replace('-', ' ')
            self.label_map[idx] = filename.replace('.json', '')
            self.label_descriptions[idx] = label_text
            with open(os.path.join(json_dir, filename), 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            # 간단한 샘플 생성
            samples = self.create_simple_samples(content, label_text, idx)
            data.extend(samples)
            
        self.num_labels = len(self.label_map)
        
        # 클래스별 샘플 수 확인
        from collections import Counter
        label_distribution = Counter([item['label'] for item in data])
        print(f"총 {self.num_labels}개의 클래스, {len(data)}개의 학습 샘플 생성")
        print("클래스별 샘플 분포:")
        for label_idx, count in sorted(label_distribution.items()):
            label_name = self.label_map[label_idx]
            print(f"  {label_name}: {count}개")
        
        return data

    def load_model(self):
        print("모델 로딩 중...")
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            special_tokens = {"additional_special_tokens": ["문서 제목:", "내용:"]}
            self.tokenizer.add_special_tokens(special_tokens)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = prepare_model_for_kbit_training(self.model)
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,  # 기본값으로 복원
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=["query", "key", "value", "dense"],
            modules_to_save=["classifier"],
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        print("모델 로딩 완료")

    def prepare_training_data(self, json_dir):
        print("학습 데이터 준비 중...")
        data = self.load_json_data(json_dir)
        
        # 클래스별로 균등하게 분할하기 위한 stratified split
        # 작은 데이터셋이므로 80/20 대신 85/15로 분할
        labels = [item['label'] for item in data]
        
        # 각 클래스당 최소 2개의 샘플이 있는지 확인
        from collections import Counter
        label_counts = Counter(labels)
        min_samples = min(label_counts.values())
        
        if min_samples >= 2 and len(data) >= self.num_labels * 2:
            # stratified split 사용
            test_size = max(0.15, self.num_labels / len(data))  # 최소한 각 클래스당 1개는 검증 세트에
            train_data, eval_data = train_test_split(
                data, 
                test_size=test_size, 
                random_state=42, 
                stratify=labels
            )
        else:
            # 단순 split 사용
            train_data, eval_data = train_test_split(
                data, 
                test_size=0.2, 
                random_state=42
            )
        
        print(f"학습 데이터: {len(train_data)}개, 검증 데이터: {len(eval_data)}개")
        
        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(eval_data)

        def preprocess(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors=None,
            )

        train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=['text', 'label_text'])
        eval_dataset = eval_dataset.map(preprocess, batched=True, remove_columns=['text', 'label_text'])
        return train_dataset, eval_dataset

    def train(self, json_dir, output_dir="fine_tuned_model", epochs=5):
        print("파인튜닝 시작...")
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            special_tokens = {"additional_special_tokens": ["문서 제목:", "내용:"]}
            self.tokenizer.add_special_tokens(special_tokens)
        train_dataset, eval_dataset = self.prepare_training_data(json_dir)
        self.load_model()
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            fp16=False,
            logging_steps=5,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            remove_unused_columns=False,
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
        )
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True, return_tensors="pt")
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        print("학습 시작...")
        train_result = trainer.train()
        print("\n학습 결과:")
        if 'train_runtime' in train_result.metrics:
            print(f"Total training time: {train_result.metrics['train_runtime']:.2f}s")
        if 'train_loss' in train_result.metrics:
            print(f"Training loss: {train_result.metrics['train_loss']:.4f}")
        print("\n최종 평가 결과:")
        eval_result = trainer.evaluate()
        for k in ['eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall']:
            if k in eval_result:
                print(f"{k.replace('eval_', '').title()}: {eval_result[k]:.4f}")
        print(f"\n모델 저장 중... ({output_dir})")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        with open(os.path.join(output_dir, 'label_info.json'), 'w', encoding='utf-8') as f:
            json.dump({'label_map': self.label_map, 'label_descriptions': self.label_descriptions}, f, ensure_ascii=False, indent=2)

    def predict(self, question, top_k=3):
        self.model.eval()
        formatted = f"문서 제목: 질문\n내용: {question}"
        inputs = self.tokenizer(formatted, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            top_probs, top_idx = torch.topk(probs, k=min(top_k, len(self.label_map)))
        results = []
        for p, i in zip(top_probs, top_idx):
            results.append((self.label_map[i.item()], p.item()))
        return results


def main():
    try:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
        classifier = JSONClassifier()
        print("\n파인튜닝 시작...")
        classifier.train('json', epochs=10)
        print("\n테스트 시작...")
        test_questions = [
            ("정수 처리 과정에서 응집 공정은 어떻게 진행되나요?", "06_process_coagulation"),
            ("탄소중립 실현을 위한 방안은 무엇인가요?", "13_carbon_neutral"),
            ("알람 시스템의 주요 기능은 무엇인가요?", "10_alarm_system"),
            ("여과 공정에서 역세척은 언제 수행하나요?", "08_process_filtration"),
            ("침전 공정의 운전 방식은 무엇인가요?", "07_process_sedimentation"),
            ("소독 공정에서 염소 주입량은 어떻게 결정하나요?", "09_process_disinfection"),
            ("플랫폼 접근 권한은 어떻게 관리하나요?", "02_platform_access"),
        ]
        
        correct_predictions = 0
        total_predictions = len(test_questions)
        
        for q, expected_label in test_questions:
            preds = classifier.predict(q)
            predicted_label = preds[0][0] if preds else None
            
            if predicted_label == expected_label:
                correct_predictions += 1
                result_mark = "✓ 정답"
            else:
                result_mark = f"✗ 오답 (정답: {expected_label})"
            
            print(f"\n질문: {q}")
            print(f"예측 결과: {result_mark}")
            print("관련성 높은 문서들:")
            for rank, (fname, conf) in enumerate(preds, 1):
                print(f"{rank}. {fname}.json (신뢰도: {conf*100:.2f}%)")
        
        accuracy = correct_predictions / total_predictions * 100
        print(f"\n=== 전체 테스트 결과 ===")
        print(f"정확도: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)")
    except Exception as e:
        print(f"오류 발생: {e}")
        if torch.cuda.is_available():
            print("\nGPU 메모리 상태:")
            print(torch.cuda.memory_summary())


if __name__ == "__main__":
    main()