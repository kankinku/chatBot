"""
RAG 시스템 평가 지표 모듈
"""
import re
from typing import List, Dict, Any
from collections import Counter


class EvaluationMetrics:
    """평가 지표 계산 클래스"""
    
    @staticmethod
    def keyword_accuracy(answer: str, keywords: List[str]) -> Dict[str, Any]:
        """키워드 정확도 계산"""
        if not keywords:
            return {
                "matched_count": 0,
                "total_keywords": 0,
                "accuracy": 0.0,
                "matched_keywords": []
            }
        
        answer_lower = answer.lower()
        matched_keywords = []
        
        for keyword in keywords:
            if keyword.lower() in answer_lower:
                matched_keywords.append(keyword)
        
        accuracy = len(matched_keywords) / len(keywords) if keywords else 0.0
        
        return {
            "matched_count": len(matched_keywords),
            "total_keywords": len(keywords),
            "accuracy": accuracy,
            "matched_keywords": matched_keywords
        }
    
    @staticmethod
    def token_overlap(prediction: str, reference: str) -> Dict[str, float]:
        """토큰 기반 Precision, Recall, F1 계산"""
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())
        
        if not pred_tokens or not ref_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        common = pred_tokens & ref_tokens
        
        precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(common) / len(ref_tokens) if ref_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    @staticmethod
    def exact_match(prediction: str, reference: str) -> float:
        """정확히 일치하는지 확인 (정규화 후)"""
        pred_norm = prediction.lower().strip()
        ref_norm = reference.lower().strip()
        return 1.0 if pred_norm == ref_norm else 0.0
    
    @staticmethod
    def contains_match(prediction: str, reference: str) -> float:
        """답변이 참조를 포함하거나 참조가 답변을 포함하는지"""
        pred_norm = prediction.lower().strip()
        ref_norm = reference.lower().strip()
        
        if ref_norm in pred_norm or pred_norm in ref_norm:
            return 1.0
        return 0.0
    
    @staticmethod
    def extract_numbers(text: str) -> List[str]:
        """텍스트에서 숫자 추출"""
        # 숫자와 소수점, 콤마 포함
        pattern = r'\d+(?:[.,]\d+)*'
        numbers = re.findall(pattern, text)
        return numbers
    
    @staticmethod
    def numeric_accuracy(prediction: str, reference: str) -> Dict[str, Any]:
        """숫자 정확도 평가"""
        pred_numbers = set(EvaluationMetrics.extract_numbers(prediction))
        ref_numbers = set(EvaluationMetrics.extract_numbers(reference))
        
        if not ref_numbers:
            return {
                "accuracy": 1.0 if not pred_numbers else 0.5,
                "matched_numbers": [],
                "total_numbers": 0
            }
        
        matched = pred_numbers & ref_numbers
        accuracy = len(matched) / len(ref_numbers) if ref_numbers else 0.0
        
        return {
            "accuracy": accuracy,
            "matched_numbers": list(matched),
            "total_numbers": len(ref_numbers)
        }
    
    @staticmethod
    def calculate_bleu_2(prediction: str, reference: str) -> float:
        """간단한 BLEU-2 (bi-gram) 계산"""
        def get_ngrams(text: str, n: int) -> List[tuple]:
            tokens = text.lower().split()
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        pred_bigrams = get_ngrams(prediction, 2)
        ref_bigrams = get_ngrams(reference, 2)
        
        if not pred_bigrams or not ref_bigrams:
            return 0.0
        
        pred_counter = Counter(pred_bigrams)
        ref_counter = Counter(ref_bigrams)
        
        overlap = sum((pred_counter & ref_counter).values())
        total = sum(pred_counter.values())
        
        return overlap / total if total > 0 else 0.0
    
    @staticmethod
    def calculate_rouge_l(prediction: str, reference: str) -> float:
        """간단한 ROUGE-L (최장 공통 부분수열 기반) 계산"""
        def lcs_length(s1: List[str], s2: List[str]) -> int:
            """최장 공통 부분수열 길이"""
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        lcs_len = lcs_length(pred_tokens, ref_tokens)
        
        precision = lcs_len / len(pred_tokens) if pred_tokens else 0.0
        recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    @staticmethod
    def extract_units(text: str) -> List[str]:
        """텍스트에서 단위 추출 (kg, kWh, %, m, cm 등)"""
        # 단위 패턴
        unit_pattern = r'\b(\d+(?:[.,]\d+)*)\s*([a-zA-Z]+|%|℃|°C)\b'
        matches = re.findall(unit_pattern, text)
        units = [match[1] for match in matches]
        return units
    
    @staticmethod
    def unit_accuracy(prediction: str, reference: str) -> Dict[str, Any]:
        """단위 정확도 평가"""
        pred_units = set(EvaluationMetrics.extract_units(prediction))
        ref_units = set(EvaluationMetrics.extract_units(reference))
        
        if not ref_units:
            return {
                "accuracy": 1.0 if not pred_units else 0.5,
                "matched_units": [],
                "total_units": 0
            }
        
        matched = pred_units & ref_units
        accuracy = len(matched) / len(ref_units) if ref_units else 0.0
        
        return {
            "accuracy": accuracy,
            "matched_units": list(matched),
            "total_units": len(ref_units)
        }
    
    @staticmethod
    def faithfulness_score(prediction: str, contexts: List[str]) -> float:
        """
        Faithfulness (충실성): 답변이 제공된 컨텍스트에 얼마나 충실한지
        간단한 구현: 답변의 토큰이 컨텍스트에 포함되어 있는지 확인
        """
        if not contexts or not prediction:
            return 0.0
        
        pred_tokens = set(prediction.lower().split())
        context_text = " ".join(contexts).lower()
        context_tokens = set(context_text.split())
        
        if not pred_tokens:
            return 0.0
        
        # 답변의 토큰 중 컨텍스트에 포함된 비율
        matched = pred_tokens & context_tokens
        faithfulness = len(matched) / len(pred_tokens)
        
        return faithfulness
    
    @staticmethod
    def answer_correctness(prediction: str, reference: str) -> float:
        """
        Answer Correctness: 답변의 정확도
        토큰 F1과 의미적 유사도의 조합
        """
        token_results = EvaluationMetrics.token_overlap(prediction, reference)
        rouge = EvaluationMetrics.calculate_rouge_l(prediction, reference)
        
        # F1과 ROUGE-L의 평균
        correctness = (token_results["f1"] + rouge) / 2
        
        return correctness
    
    @staticmethod
    def context_precision(contexts: List[str], reference: str) -> float:
        """
        Context Precision: 검색된 컨텍스트의 정밀도
        컨텍스트가 정답과 얼마나 관련있는지
        """
        if not contexts or not reference:
            return 0.0
        
        ref_tokens = set(reference.lower().split())
        
        relevant_count = 0
        for context in contexts:
            context_tokens = set(context.lower().split())
            overlap = context_tokens & ref_tokens
            
            # 오버랩이 충분하면 관련있다고 판단
            if len(overlap) / len(ref_tokens) > 0.1:  # 10% 이상 겹침
                relevant_count += 1
        
        precision = relevant_count / len(contexts) if contexts else 0.0
        
        return precision
    
    @staticmethod
    def evaluate_answer(prediction: str, reference: str, keywords: List[str], 
                       contexts: List[str] = None) -> Dict[str, Any]:
        """답변에 대한 모든 평가 지표 계산"""
        
        # 1. 기본 지표
        keyword_results = EvaluationMetrics.keyword_accuracy(prediction, keywords)
        token_results = EvaluationMetrics.token_overlap(prediction, reference)
        numeric_results = EvaluationMetrics.numeric_accuracy(prediction, reference)
        unit_results = EvaluationMetrics.unit_accuracy(prediction, reference)
        
        # 2. 텍스트 유사도
        bleu_2 = EvaluationMetrics.calculate_bleu_2(prediction, reference)
        rouge_l = EvaluationMetrics.calculate_rouge_l(prediction, reference)
        exact_match = EvaluationMetrics.exact_match(prediction, reference)
        contains_match = EvaluationMetrics.contains_match(prediction, reference)
        
        # 3. RAG 특화 지표
        if contexts:
            faithfulness = EvaluationMetrics.faithfulness_score(prediction, contexts)
            context_precision = EvaluationMetrics.context_precision(contexts, reference)
        else:
            faithfulness = 0.0
            context_precision = 0.0
        
        answer_correctness = EvaluationMetrics.answer_correctness(prediction, reference)
        
        # 4. 종합 점수들 계산
        
        # 기본 Score (v5) - 실무 중심
        basic_score = (
            keyword_results["accuracy"] * 0.35 +
            token_results["f1"] * 0.25 +
            numeric_results["accuracy"] * 0.20 +
            rouge_l * 0.15 +
            bleu_2 * 0.05
        )
        
        # 도메인 특화 종합
        domain_score = (
            numeric_results["accuracy"] * 0.5 +
            unit_results["accuracy"] * 0.3 +
            keyword_results["accuracy"] * 0.2
        )
        
        # RAG 종합 점수
        rag_overall = (
            faithfulness * 0.4 +
            answer_correctness * 0.4 +
            context_precision * 0.2
        ) if contexts else answer_correctness
        
        return {
            # 종합 점수들
            "basic_score": basic_score,
            "domain_score": domain_score,
            "rag_overall": rag_overall,
            
            # 상세 지표
            "keyword": keyword_results,
            "token_overlap": token_results,
            "numeric": numeric_results,
            "unit": unit_results,
            
            # RAG 지표
            "faithfulness": faithfulness,
            "answer_correctness": answer_correctness,
            "context_precision": context_precision,
            
            # 텍스트 유사도
            "text_similarity": {
                "bleu_2": bleu_2,
                "rouge_l": rouge_l,
                "exact_match": exact_match,
                "contains_match": contains_match
            }
        }

