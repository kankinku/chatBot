#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
학술 논문용 평가 지표 구현

표준 평가 지표: Token F1, ROUGE-L, BLEU, Exact Match
"""

import re
from typing import List, Dict, Any, Tuple
from collections import Counter


class AcademicMetrics:
    """학술 논문용 표준 평가 지표"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """텍스트 정규화"""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """토크나이저 (한글 지원)"""
        tokens = re.findall(r'[\w]+', text)
        return [t for t in tokens if t]
    
    # ============================================================
    # 1. Exact Match (EM) - SQuAD 표준
    # ============================================================
    
    @staticmethod
    def exact_match(pred: str, gold: str) -> float:
        """
        Exact Match (완전 일치)
        
        SQuAD, KorQuAD 등에서 사용하는 가장 엄격한 지표
        
        Returns:
            1.0 if exact match, else 0.0
            
        Reference:
            Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for 
            Machine Comprehension of Text. EMNLP 2016.
        """
        pred_norm = AcademicMetrics.normalize_text(pred)
        gold_norm = AcademicMetrics.normalize_text(gold)
        return 1.0 if pred_norm == gold_norm else 0.0
    
    # ============================================================
    # 2. Token F1 Score - SQuAD 표준
    # ============================================================
    
    @staticmethod
    def token_f1_score(pred: str, gold: str) -> Dict[str, float]:
        """
        Token-level F1 Score
        
        토큰 단위 정밀도(Precision)와 재현율(Recall)의 조화평균
        SQuAD, KorQuAD의 주요 평가 지표
        
        Returns:
            {'precision': float, 'recall': float, 'f1': float}
            
        Reference:
            Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for 
            Machine Comprehension of Text. EMNLP 2016.
        """
        pred_tokens = AcademicMetrics.tokenize(AcademicMetrics.normalize_text(pred))
        gold_tokens = AcademicMetrics.tokenize(AcademicMetrics.normalize_text(gold))
        
        if not pred_tokens or not gold_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4)
        }
    
    # ============================================================
    # 3. ROUGE-L - 요약 평가 표준
    # ============================================================
    
    @staticmethod
    def _lcs_length(x: List[str], y: List[str]) -> int:
        """최장 공통 부분수열 (LCS) 길이"""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    @staticmethod
    def rouge_l(pred: str, gold: str) -> Dict[str, float]:
        """
        ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)
        
        최장 공통 부분수열(LCS) 기반 평가
        순서를 고려한 유사도 측정
        
        Returns:
            {'precision': float, 'recall': float, 'f1': float}
            
        Reference:
            Lin, C. Y. (2004). ROUGE: A Package for Automatic Evaluation 
            of Summaries. ACL Workshop 2004.
        """
        pred_tokens = AcademicMetrics.tokenize(AcademicMetrics.normalize_text(pred))
        gold_tokens = AcademicMetrics.tokenize(AcademicMetrics.normalize_text(gold))
        
        if not pred_tokens or not gold_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        lcs_len = AcademicMetrics._lcs_length(pred_tokens, gold_tokens)
        
        if lcs_len == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        precision = lcs_len / len(pred_tokens)
        recall = lcs_len / len(gold_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4)
        }
    
    # ============================================================
    # 4. BLEU - 기계번역 표준
    # ============================================================
    
    @staticmethod
    def bleu_n(pred: str, gold: str, n: int = 2) -> float:
        """
        BLEU-n (Bilingual Evaluation Understudy)
        
        n-gram 정밀도 기반 평가
        기계번역 및 텍스트 생성 평가의 표준
        
        Args:
            n: n-gram 크기 (기본 2)
        
        Returns:
            BLEU-n score (0.0 ~ 1.0)
            
        Reference:
            Papineni et al. (2002). BLEU: a Method for Automatic 
            Evaluation of Machine Translation. ACL 2002.
        """
        pred_tokens = AcademicMetrics.tokenize(AcademicMetrics.normalize_text(pred))
        gold_tokens = AcademicMetrics.tokenize(AcademicMetrics.normalize_text(gold))
        
        if len(pred_tokens) < n or len(gold_tokens) < n:
            return 0.0
        
        # n-gram 생성
        pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)]
        gold_ngrams = [tuple(gold_tokens[i:i+n]) for i in range(len(gold_tokens) - n + 1)]
        
        pred_counter = Counter(pred_ngrams)
        gold_counter = Counter(gold_ngrams)
        
        # Clipped counts
        overlap = pred_counter & gold_counter
        num_match = sum(overlap.values())
        num_pred = len(pred_ngrams)
        
        if num_pred == 0:
            return 0.0
        
        return round(num_match / num_pred, 4)
    
    # ============================================================
    # 5. 종합 평가
    # ============================================================
    
    @staticmethod
    def evaluate_all(pred: str, gold: str) -> Dict[str, Any]:
        """
        모든 학술 지표로 평가
        
        Returns:
            Dict with all academic metrics
        """
        return {
            'exact_match': AcademicMetrics.exact_match(pred, gold),
            'token_f1': AcademicMetrics.token_f1_score(pred, gold),
            'rouge_l': AcademicMetrics.rouge_l(pred, gold),
            'bleu_1': AcademicMetrics.bleu_n(pred, gold, n=1),
            'bleu_2': AcademicMetrics.bleu_n(pred, gold, n=2),
        }


def main():
    """테스트"""
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("=" * 70)
    print("학술 논문용 평가 지표 테스트")
    print("=" * 70)
    
    # 테스트 예시
    pred = "발주기관은 한국수자원공사이며, 사업명은 금강 유역 스마트 정수장입니다."
    gold = "발주기관은 한국수자원공사이며, 사업명은 금강 유역 남부 스마트 정수장 확대 구축 용역입니다."
    
    print(f"\n예측: {pred}")
    print(f"정답: {gold}")
    print("\n" + "=" * 70)
    
    metrics = AcademicMetrics.evaluate_all(pred, gold)
    
    print("\n📊 학술 평가 결과:")
    print(f"\n1. Exact Match (SQuAD)")
    print(f"   Score: {metrics['exact_match']:.3f}")
    
    print(f"\n2. Token F1 (SQuAD)")
    print(f"   Precision: {metrics['token_f1']['precision']:.3f}")
    print(f"   Recall:    {metrics['token_f1']['recall']:.3f}")
    print(f"   F1:        {metrics['token_f1']['f1']:.3f}")
    
    print(f"\n3. ROUGE-L (ACL 2004)")
    print(f"   Precision: {metrics['rouge_l']['precision']:.3f}")
    print(f"   Recall:    {metrics['rouge_l']['recall']:.3f}")
    print(f"   F1:        {metrics['rouge_l']['f1']:.3f}")
    
    print(f"\n4. BLEU (ACL 2002)")
    print(f"   BLEU-1: {metrics['bleu_1']:.3f}")
    print(f"   BLEU-2: {metrics['bleu_2']:.3f}")
    
    print("\n" + "=" * 70)
    print("✅ 테스트 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()

