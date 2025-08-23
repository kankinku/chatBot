"""
질문 분석 및 답변 생성 정확도 평가 모듈

이 모듈은 PDF QA 시스템의 성능을 다양한 메트릭으로 평가하고,
개선점을 식별하여 시스템 최적화를 지원합니다.
"""

import re
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict, Counter
import statistics

# 평가 메트릭 라이브러리들
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("경고: rouge-score 라이브러리가 없습니다. ROUGE 평가가 제한됩니다.")

try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    print("경고: bert-score 라이브러리가 없습니다. BERTScore 평가가 제한됩니다.")

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("경고: nltk 라이브러리가 없습니다. BLEU 평가가 제한됩니다.")

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .question_analyzer import AnalyzedQuestion, QuestionType
from .answer_generator import Answer
from .pdf_processor import TextChunk
from .vector_store import calculate_retrieval_metrics

import logging
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """평가 결과 데이터 클래스"""
    metric_name: str
    score: float
    max_score: float
    description: str
    details: Optional[Dict] = None

@dataclass
class QuestionAnalysisEval:
    """질문 분석 평가 결과"""
    question_id: str
    original_question: str
    
    # 질문 분류 평가
    predicted_type: str
    actual_type: str
    type_correct: bool
    
    # 키워드 추출 평가
    predicted_keywords: List[str]
    actual_keywords: List[str]
    keyword_precision: float
    keyword_recall: float
    keyword_f1: float
    
    # 의도 분석 평가
    predicted_intent: str
    actual_intent: str
    intent_correct: bool
    
    # 전체 점수
    overall_score: float

@dataclass 
class AnswerGenerationEval:
    """답변 생성 평가 결과"""
    question_id: str
    question: str
    generated_answer: str
    reference_answer: str
    
    # 텍스트 유사도 메트릭
    rouge_scores: Dict[str, float]
    bleu_score: float
    bert_score: float
    semantic_similarity: float
    
    # 내용 평가
    factual_accuracy: float
    completeness: float
    relevance: float
    fluency: float
    
    # 검색 품질 평가
    retrieval_precision: float
    retrieval_recall: float
    retrieval_f1: float
    
    # 전체 점수
    overall_score: float

@dataclass
class SystemEvaluation:
    """전체 시스템 평가 결과"""
    evaluation_date: datetime
    
    # 질문 분석 성능
    question_analysis_scores: List[QuestionAnalysisEval]
    avg_question_analysis_score: float
    
    # 답변 생성 성능
    answer_generation_scores: List[AnswerGenerationEval]
    avg_answer_generation_score: float
    
    # 검색 성능
    avg_retrieval_scores: Dict[str, float]
    
    # 전체 성능
    overall_system_score: float
    
    # 성능 분석
    performance_by_question_type: Dict[str, float]
    performance_by_difficulty: Dict[str, float]
    
    # 개선 제안
    improvement_suggestions: List[str]

class PDFQAEvaluator:
    """
    PDF QA 시스템 평가기
    
    주요 기능:
    1. 질문 분석 정확도 평가
    2. 답변 생성 품질 평가
    3. 검색 성능 평가
    4. 전체 시스템 성능 분석
    5. 개선점 도출
    """
    
    def __init__(self, embedding_model: str = "jhgan/ko-sroberta-multitask"):
        """
        PDFQAEvaluator 초기화
        
        Args:
            embedding_model: 의미 유사도 계산용 임베딩 모델
        """
        # 임베딩 모델 로드 (의미 유사도 계산용)
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"평가용 임베딩 모델 로드: {embedding_model}")
        except Exception as e:
            logger.warning(f"임베딩 모델 로드 실패: {e}")
            self.embedding_model = None
        
        # ROUGE 스코어러 초기화
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=False
            )
        else:
            self.rouge_scorer = None
        
        # BLEU 스무딩 함수
        if NLTK_AVAILABLE:
            self.smoothing_function = SmoothingFunction().method1
        else:
            self.smoothing_function = None
        
        logger.info("PDFQAEvaluator 초기화 완료")
    
    def evaluate_question_analysis(self, 
                                 analyzed_questions: List[AnalyzedQuestion],
                                 ground_truth: List[Dict]) -> List[QuestionAnalysisEval]:
        """
        질문 분석 성능 평가
        
        Args:
            analyzed_questions: 분석된 질문들
            ground_truth: 정답 데이터 [{"question": str, "type": str, "keywords": List[str], "intent": str}]
            
        Returns:
            질문 분석 평가 결과 리스트
        """
        evaluations = []
        
        for i, analyzed_q in enumerate(analyzed_questions):
            if i >= len(ground_truth):
                break
            
            gt = ground_truth[i]
            
            # 1. 질문 유형 분류 평가
            predicted_type = analyzed_q.question_type.value
            actual_type = gt['type']
            type_correct = predicted_type == actual_type
            
            # 2. 키워드 추출 평가
            predicted_keywords = set(analyzed_q.keywords)
            actual_keywords = set(gt['keywords'])
            
            keyword_precision = self._calculate_precision(predicted_keywords, actual_keywords)
            keyword_recall = self._calculate_recall(predicted_keywords, actual_keywords)
            keyword_f1 = self._calculate_f1(keyword_precision, keyword_recall)
            
            # 3. 의도 분석 평가
            predicted_intent = analyzed_q.intent
            actual_intent = gt['intent']
            intent_correct = predicted_intent == actual_intent
            
            # 4. 전체 점수 계산
            overall_score = (
                (1.0 if type_correct else 0.0) * 0.4 +
                keyword_f1 * 0.4 +
                (1.0 if intent_correct else 0.0) * 0.2
            )
            
            eval_result = QuestionAnalysisEval(
                question_id=f"q_{i}",
                original_question=analyzed_q.original_question,
                predicted_type=predicted_type,
                actual_type=actual_type,
                type_correct=type_correct,
                predicted_keywords=list(predicted_keywords),
                actual_keywords=list(actual_keywords),
                keyword_precision=keyword_precision,
                keyword_recall=keyword_recall,
                keyword_f1=keyword_f1,
                predicted_intent=predicted_intent,
                actual_intent=actual_intent,
                intent_correct=intent_correct,
                overall_score=overall_score
            )
            
            evaluations.append(eval_result)
        
        logger.info(f"질문 분석 평가 완료: {len(evaluations)}개 질문")
        return evaluations
    
    def evaluate_answer_generation(self,
                                 questions: List[str],
                                 generated_answers: List[Answer],
                                 reference_answers: List[str],
                                 relevant_chunks_gt: List[List[str]] = None) -> List[AnswerGenerationEval]:
        """
        답변 생성 성능 평가
        
        Args:
            questions: 질문들
            generated_answers: 생성된 답변들
            reference_answers: 참조 답변들 (정답)
            relevant_chunks_gt: 실제 관련 청크들 (검색 평가용)
            
        Returns:
            답변 생성 평가 결과 리스트
        """
        evaluations = []
        
        for i, (question, gen_answer, ref_answer) in enumerate(zip(questions, generated_answers, reference_answers)):
            
            # 1. 텍스트 유사도 메트릭
            rouge_scores = self._calculate_rouge_scores(gen_answer.content, ref_answer)
            bleu_score = self._calculate_bleu_score(gen_answer.content, ref_answer)
            bert_score_val = self._calculate_bert_score(gen_answer.content, ref_answer)
            semantic_similarity = self._calculate_semantic_similarity(gen_answer.content, ref_answer)
            
            # 2. 내용 평가
            factual_accuracy = self._evaluate_factual_accuracy(gen_answer.content, ref_answer)
            completeness = self._evaluate_completeness(gen_answer.content, ref_answer)
            relevance = self._evaluate_relevance(question, gen_answer.content)
            fluency = self._evaluate_fluency(gen_answer.content)
            
            # 3. 검색 품질 평가
            retrieval_scores = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            if relevant_chunks_gt and i < len(relevant_chunks_gt):
                gt_chunks = relevant_chunks_gt[i]
                predicted_chunks = gen_answer.used_chunks
                
                retrieval_scores = self._evaluate_retrieval_quality(predicted_chunks, gt_chunks)
            
            # 4. 전체 점수 계산
            overall_score = (
                semantic_similarity * 0.25 +
                factual_accuracy * 0.25 +
                completeness * 0.2 +
                relevance * 0.15 +
                fluency * 0.1 +
                retrieval_scores["f1"] * 0.05
            )
            
            eval_result = AnswerGenerationEval(
                question_id=f"q_{i}",
                question=question,
                generated_answer=gen_answer.content,
                reference_answer=ref_answer,
                rouge_scores=rouge_scores,
                bleu_score=bleu_score,
                bert_score=bert_score_val,
                semantic_similarity=semantic_similarity,
                factual_accuracy=factual_accuracy,
                completeness=completeness,
                relevance=relevance,
                fluency=fluency,
                retrieval_precision=retrieval_scores["precision"],
                retrieval_recall=retrieval_scores["recall"],
                retrieval_f1=retrieval_scores["f1"],
                overall_score=overall_score
            )
            
            evaluations.append(eval_result)
        
        logger.info(f"답변 생성 평가 완료: {len(evaluations)}개 답변")
        return evaluations
    
    def evaluate_system(self,
                       question_analysis_evals: List[QuestionAnalysisEval],
                       answer_generation_evals: List[AnswerGenerationEval],
                       additional_metrics: Dict = None) -> SystemEvaluation:
        """
        전체 시스템 성능 평가
        
        Args:
            question_analysis_evals: 질문 분석 평가 결과들
            answer_generation_evals: 답변 생성 평가 결과들
            additional_metrics: 추가 메트릭들
            
        Returns:
            전체 시스템 평가 결과
        """
        # 평균 점수 계산
        avg_qa_score = statistics.mean([eval.overall_score for eval in question_analysis_evals])
        avg_ag_score = statistics.mean([eval.overall_score for eval in answer_generation_evals])
        
        # 검색 성능 평균
        avg_retrieval_scores = {
            "precision": statistics.mean([eval.retrieval_precision for eval in answer_generation_evals]),
            "recall": statistics.mean([eval.retrieval_recall for eval in answer_generation_evals]),
            "f1": statistics.mean([eval.retrieval_f1 for eval in answer_generation_evals])
        }
        
        # 전체 시스템 점수
        overall_score = (avg_qa_score * 0.3 + avg_ag_score * 0.7)
        
        # 질문 유형별 성능 분석
        performance_by_type = self._analyze_performance_by_question_type(
            question_analysis_evals, answer_generation_evals
        )
        
        # 난이도별 성능 분석 (간단한 휴리스틱)
        performance_by_difficulty = self._analyze_performance_by_difficulty(
            answer_generation_evals
        )
        
        # 개선 제안 생성
        improvement_suggestions = self._generate_improvement_suggestions(
            question_analysis_evals, answer_generation_evals, avg_retrieval_scores
        )
        
        system_eval = SystemEvaluation(
            evaluation_date=datetime.now(),
            question_analysis_scores=question_analysis_evals,
            avg_question_analysis_score=avg_qa_score,
            answer_generation_scores=answer_generation_evals,
            avg_answer_generation_score=avg_ag_score,
            avg_retrieval_scores=avg_retrieval_scores,
            overall_system_score=overall_score,
            performance_by_question_type=performance_by_type,
            performance_by_difficulty=performance_by_difficulty,
            improvement_suggestions=improvement_suggestions
        )
        
        logger.info(f"전체 시스템 평가 완료: 점수 {overall_score:.3f}")
        return system_eval
    
    # 내부 메트릭 계산 함수들
    def _calculate_precision(self, predicted: set, actual: set) -> float:
        """정밀도 계산"""
        if not predicted:
            return 0.0
        return len(predicted & actual) / len(predicted)
    
    def _calculate_recall(self, predicted: set, actual: set) -> float:
        """재현율 계산"""
        if not actual:
            return 0.0
        return len(predicted & actual) / len(actual)
    
    def _calculate_f1(self, precision: float, recall: float) -> float:
        """F1 점수 계산"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_rouge_scores(self, generated: str, reference: str) -> Dict[str, float]:
        """ROUGE 점수 계산"""
        if not self.rouge_scorer:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        try:
            scores = self.rouge_scorer.score(reference, generated)
            return {
                "rouge1": scores['rouge1'].fmeasure,
                "rouge2": scores['rouge2'].fmeasure,
                "rougeL": scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.warning(f"ROUGE 계산 실패: {e}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    def _calculate_bleu_score(self, generated: str, reference: str) -> float:
        """BLEU 점수 계산"""
        if not NLTK_AVAILABLE:
            return 0.0
        
        try:
            # 단어 단위로 분할
            reference_tokens = reference.split()
            generated_tokens = generated.split()
            
            # BLEU 계산
            score = sentence_bleu(
                [reference_tokens], 
                generated_tokens,
                smoothing_function=self.smoothing_function
            )
            return score
        except Exception as e:
            logger.warning(f"BLEU 계산 실패: {e}")
            return 0.0
    
    def _calculate_bert_score(self, generated: str, reference: str) -> float:
        """BERTScore 계산"""
        if not BERT_SCORE_AVAILABLE:
            return 0.0
        
        try:
            P, R, F1 = bert_score([generated], [reference], lang="ko", verbose=False)
            return F1.item()
        except Exception as e:
            logger.warning(f"BERTScore 계산 실패: {e}")
            return 0.0
    
    def _calculate_semantic_similarity(self, generated: str, reference: str) -> float:
        """의미적 유사도 계산"""
        if not self.embedding_model:
            return 0.0
        
        try:
            embeddings = self.embedding_model.encode([generated, reference])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return max(0.0, similarity)  # 음수 유사도는 0으로 처리
        except Exception as e:
            logger.warning(f"의미적 유사도 계산 실패: {e}")
            return 0.0
    
    def _evaluate_factual_accuracy(self, generated: str, reference: str) -> float:
        """사실적 정확도 평가 (간단한 키워드 매칭 기반)"""
        # 참조 답변에서 중요한 키워드 추출
        reference_words = set(re.findall(r'\b\w+\b', reference.lower()))
        generated_words = set(re.findall(r'\b\w+\b', generated.lower()))
        
        # 중요 키워드 매칭률
        if not reference_words:
            return 1.0
        
        overlap = len(reference_words & generated_words)
        return overlap / len(reference_words)
    
    def _evaluate_completeness(self, generated: str, reference: str) -> float:
        """답변 완성도 평가"""
        # 길이 기반 완성도 (너무 짧으면 감점)
        ref_length = len(reference)
        gen_length = len(generated)
        
        if ref_length == 0:
            return 1.0 if gen_length > 0 else 0.0
        
        length_ratio = min(gen_length / ref_length, 1.0)
        
        # 문장 완성도 (적절한 종결어미)
        completion_score = 1.0 if generated.strip().endswith(('.', '!', '?', '다', '습니다', '요')) else 0.7
        
        return length_ratio * 0.7 + completion_score * 0.3
    
    def _evaluate_relevance(self, question: str, answer: str) -> float:
        """답변 관련성 평가"""
        # 질문과 답변 간의 키워드 매칭
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
        
        # 불용어 제거 (간단한 버전)
        stopwords = {'은', '는', '이', '가', '을', '를', '의', '에', '와', '과', '도', '만'}
        question_words -= stopwords
        answer_words -= stopwords
        
        if not question_words:
            return 1.0
        
        overlap = len(question_words & answer_words)
        return overlap / len(question_words)
    
    def _evaluate_fluency(self, text: str) -> float:
        """답변 유창성 평가 (간단한 휴리스틱)"""
        if not text.strip():
            return 0.0
        
        score = 1.0
        
        # 반복되는 문구 체크
        words = text.split()
        if len(words) > len(set(words)) * 1.5:  # 반복이 많으면 감점
            score -= 0.2
        
        # 문장 부호 체크
        if not any(punct in text for punct in '.!?'):
            score -= 0.2
        
        # 길이 체크 (너무 짧거나 길면 감점)
        if len(text) < 20:
            score -= 0.3
        elif len(text) > 1000:
            score -= 0.2
        
        return max(0.0, score)
    
    def _evaluate_retrieval_quality(self, predicted_chunks: List[str], 
                                  actual_chunks: List[str]) -> Dict[str, float]:
        """검색 품질 평가"""
        predicted_set = set(predicted_chunks)
        actual_set = set(actual_chunks)
        
        precision = self._calculate_precision(predicted_set, actual_set)
        recall = self._calculate_recall(predicted_set, actual_set)
        f1 = self._calculate_f1(precision, recall)
        
        return {"precision": precision, "recall": recall, "f1": f1}
    
    def _analyze_performance_by_question_type(self,
                                            qa_evals: List[QuestionAnalysisEval],
                                            ag_evals: List[AnswerGenerationEval]) -> Dict[str, float]:
        """질문 유형별 성능 분석"""
        type_scores = defaultdict(list)
        
        for qa_eval, ag_eval in zip(qa_evals, ag_evals):
            combined_score = (qa_eval.overall_score + ag_eval.overall_score) / 2
            type_scores[qa_eval.actual_type].append(combined_score)
        
        return {q_type: statistics.mean(scores) for q_type, scores in type_scores.items()}
    
    def _analyze_performance_by_difficulty(self, 
                                         ag_evals: List[AnswerGenerationEval]) -> Dict[str, float]:
        """난이도별 성능 분석 (답변 길이와 복잡도 기반)"""
        easy_scores = []
        medium_scores = []
        hard_scores = []
        
        for eval in ag_evals:
            ref_length = len(eval.reference_answer)
            
            # 간단한 난이도 분류
            if ref_length < 100:
                easy_scores.append(eval.overall_score)
            elif ref_length < 300:
                medium_scores.append(eval.overall_score)
            else:
                hard_scores.append(eval.overall_score)
        
        difficulty_scores = {}
        if easy_scores:
            difficulty_scores["easy"] = statistics.mean(easy_scores)
        if medium_scores:
            difficulty_scores["medium"] = statistics.mean(medium_scores)
        if hard_scores:
            difficulty_scores["hard"] = statistics.mean(hard_scores)
        
        return difficulty_scores
    
    def _generate_improvement_suggestions(self,
                                        qa_evals: List[QuestionAnalysisEval],
                                        ag_evals: List[AnswerGenerationEval],
                                        retrieval_scores: Dict[str, float]) -> List[str]:
        """개선 제안 생성"""
        suggestions = []
        
        # 질문 분석 개선
        avg_qa_score = statistics.mean([eval.overall_score for eval in qa_evals])
        if avg_qa_score < 0.7:
            suggestions.append("질문 분류 모델의 파인튜닝이 필요합니다. 더 많은 레이블된 데이터를 수집하세요.")
        
        # 키워드 추출 개선
        avg_keyword_f1 = statistics.mean([eval.keyword_f1 for eval in qa_evals])
        if avg_keyword_f1 < 0.6:
            suggestions.append("키워드 추출 성능 개선이 필요합니다. 도메인 특화 사전을 구축하거나 TF-IDF 가중치를 조정하세요.")
        
        # 답변 생성 개선
        avg_ag_score = statistics.mean([eval.overall_score for eval in ag_evals])
        if avg_ag_score < 0.7:
            suggestions.append("답변 생성 모델의 성능 개선이 필요합니다. 더 큰 모델을 사용하거나 도메인 특화 파인튜닝을 고려하세요.")
        
        # 검색 성능 개선
        if retrieval_scores["f1"] < 0.6:
            suggestions.append("문서 검색 성능 개선이 필요합니다. 임베딩 모델을 업그레이드하거나 청크 크기를 조정하세요.")
        
        # 유창성 개선
        avg_fluency = statistics.mean([eval.fluency for eval in ag_evals])
        if avg_fluency < 0.8:
            suggestions.append("답변 유창성 개선이 필요합니다. 생성 모델의 후처리 로직을 강화하세요.")
        
        return suggestions
    
    def save_evaluation_report(self, system_eval: SystemEvaluation, 
                             output_path: str) -> None:
        """평가 리포트 저장"""
        # JSON 직렬화를 위한 데이터 변환
        report_data = {
            "evaluation_date": system_eval.evaluation_date.isoformat(),
            "summary": {
                "overall_system_score": system_eval.overall_system_score,
                "avg_question_analysis_score": system_eval.avg_question_analysis_score,
                "avg_answer_generation_score": system_eval.avg_answer_generation_score,
                "avg_retrieval_scores": system_eval.avg_retrieval_scores
            },
            "performance_by_question_type": system_eval.performance_by_question_type,
            "performance_by_difficulty": system_eval.performance_by_difficulty,
            "improvement_suggestions": system_eval.improvement_suggestions,
            "detailed_results": {
                "question_analysis": [asdict(eval) for eval in system_eval.question_analysis_scores],
                "answer_generation": [asdict(eval) for eval in system_eval.answer_generation_scores]
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"평가 리포트 저장 완료: {output_path}")

# 벤치마크 데이터셋 생성 함수
def create_benchmark_dataset(pdf_chunks: List[TextChunk], 
                           num_questions: int = 50) -> Dict:
    """
    평가용 벤치마크 데이터셋 생성
    
    Args:
        pdf_chunks: PDF 청크들
        num_questions: 생성할 질문 수
        
    Returns:
        벤치마크 데이터셋
        
    주의: 실제로는 사람이 직접 라벨링한 고품질 데이터가 필요합니다.
    이 함수는 예시 구조를 보여주는 것입니다.
    """
    benchmark = {
        "metadata": {
            "creation_date": datetime.now().isoformat(),
            "num_questions": num_questions,
            "num_chunks": len(pdf_chunks)
        },
        "questions": [],
        "evaluation_guidelines": {
            "question_types": list(QuestionType),
            "scoring_criteria": {
                "factual_accuracy": "0-1, 사실적 정확도",
                "completeness": "0-1, 답변 완성도",
                "relevance": "0-1, 질문 관련성",
                "fluency": "0-1, 유창성"
            }
        }
    }
    
    # 예시 질문들 (실제로는 도메인 전문가가 작성해야 함)
    example_questions = [
        {
            "question": "이 문서의 주요 개념은 무엇인가요?",
            "type": "factual",
            "keywords": ["주요", "개념", "문서"],
            "intent": "정보_요청",
            "reference_answer": "문서에서 제시하는 주요 개념은...",
            "relevant_chunks": ["chunk_0", "chunk_1"],
            "difficulty": "easy"
        }
        # 더 많은 예시 질문들...
    ]
    
    benchmark["questions"] = example_questions[:num_questions]
    
    return benchmark

if __name__ == "__main__":
    # 테스트 코드
    evaluator = PDFQAEvaluator()
    
    print("PDFQAEvaluator 모듈이 정상적으로 로드되었습니다.")
    print(f"ROUGE 사용 가능: {ROUGE_AVAILABLE}")
    print(f"BERTScore 사용 가능: {BERT_SCORE_AVAILABLE}")
    print(f"NLTK 사용 가능: {NLTK_AVAILABLE}")
    
    # 간단한 테스트
    if evaluator.embedding_model:
        similarity = evaluator._calculate_semantic_similarity(
            "안녕하세요", "반갑습니다"
        )
        print(f"의미적 유사도 테스트: {similarity:.3f}")
