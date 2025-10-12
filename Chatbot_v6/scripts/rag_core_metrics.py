#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 시스템 핵심 평가 지표 (RAGAs Framework 기반)

3대 핵심 지표:
1. Faithfulness (충실성) - 환각 방지
2. Answer Correctness (답변 정확도) - 사실적 일치
3. Context Precision (문맥 정밀도) - 검색 효율성
"""

import re
from typing import List, Dict, Any, Set
from collections import Counter


class RAGCoreMetrics:
    """RAG 시스템 핵심 평가 지표"""
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """토크나이저"""
        tokens = re.findall(r'[\w]+', text.lower())
        return [t for t in tokens if len(t) > 1]
    
    # ============================================================
    # 1. Faithfulness (충실성) - 환각 방지
    # ============================================================
    
    @staticmethod
    def faithfulness(answer: str, contexts: List[str]) -> Dict[str, Any]:
        """
        Faithfulness (충실성/근거성)
        
        답변의 모든 사실이 참고 자료(contexts)에 의해 뒷받침되는 정도를 측정.
        환각(hallucination) 방지의 핵심 지표.
        
        평가 질문: "답변이 자료 밖의 거짓말을 했나?"
        
        Args:
            answer: 생성된 답변
            contexts: 참고 자료 리스트
            
        Returns:
            {
                'score': float (0~1),
                'supported_claims': int,
                'total_claims': int,
                'support_ratio': float
            }
            
        Reference:
            Es et al. (2023). RAGAS: Automated Evaluation of 
            Retrieval Augmented Generation. arXiv:2309.15217
        """
        if not answer or not contexts:
            return {
                'score': 0.0,
                'supported_claims': 0,
                'total_claims': 0,
                'support_ratio': 0.0
            }
        
        # 답변을 문장 단위로 분리 (claim으로 간주)
        answer_sentences = [s.strip() for s in re.split(r'[.!?]\s+', answer) if len(s.strip()) > 10]
        
        if not answer_sentences:
            return {
                'score': 0.0,
                'supported_claims': 0,
                'total_claims': 0,
                'support_ratio': 0.0
            }
        
        # 모든 contexts를 하나로 합침
        combined_context = ' '.join(contexts).lower()
        context_tokens = set(RAGCoreMetrics.tokenize(combined_context))
        
        # 각 문장(claim)이 context에서 지지되는지 확인
        supported = 0
        
        for sentence in answer_sentences:
            sentence_tokens = set(RAGCoreMetrics.tokenize(sentence))
            
            # 문장의 주요 토큰들이 context에 있는지 확인
            if sentence_tokens:
                overlap_ratio = len(sentence_tokens & context_tokens) / len(sentence_tokens)
                
                # 토큰의 70% 이상이 context에 있으면 지지됨으로 간주
                if overlap_ratio >= 0.7:
                    supported += 1
        
        total_claims = len(answer_sentences)
        support_ratio = supported / total_claims
        
        # Faithfulness score
        score = support_ratio
        
        return {
            'score': round(score, 4),
            'supported_claims': supported,
            'total_claims': total_claims,
            'support_ratio': round(support_ratio, 4)
        }
    
    # ============================================================
    # 2. Answer Correctness (답변 정확도) - 사실적 일치
    # ============================================================
    
    @staticmethod
    def answer_correctness(answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        Answer Correctness (답변 정확도)
        
        생성된 답변이 정답(Ground Truth)과 의미적/사실적으로 
        얼마나 일치하는지 측정.
        
        평가 질문: "답변이 정답과 사실상 동일한가?"
        
        Args:
            answer: 생성된 답변
            ground_truth: 정답
            
        Returns:
            {
                'score': float (0~1),
                'semantic_similarity': float,
                'factual_correctness': float
            }
            
        Reference:
            Es et al. (2023). RAGAS: Automated Evaluation of 
            Retrieval Augmented Generation. arXiv:2309.15217
        """
        if not answer or not ground_truth:
            return {
                'score': 0.0,
                'semantic_similarity': 0.0,
                'factual_correctness': 0.0
            }
        
        answer_tokens = set(RAGCoreMetrics.tokenize(answer))
        truth_tokens = set(RAGCoreMetrics.tokenize(ground_truth))
        
        if not answer_tokens or not truth_tokens:
            return {
                'score': 0.0,
                'semantic_similarity': 0.0,
                'factual_correctness': 0.0
            }
        
        # 1. 의미적 유사도 (Token overlap 기반)
        common_tokens = answer_tokens & truth_tokens
        semantic_similarity = len(common_tokens) / len(truth_tokens)
        
        # 2. 사실적 정확도 (특히 숫자/단위 중심)
        # 정답의 숫자 추출
        truth_nums = set(re.findall(r'\d+(?:[.,]\d+)?', ground_truth))
        answer_nums = set(re.findall(r'\d+(?:[.,]\d+)?', answer))
        
        # 정답의 단위 추출
        truth_units = set(re.findall(r'[%°℃㎎]+|(?:mg|ppm|rpm|kwh|kg)/[lL]?', ground_truth, re.IGNORECASE))
        answer_units = set(re.findall(r'[%°℃㎎]+|(?:mg|ppm|rpm|kwh|kg)/[lL]?', answer, re.IGNORECASE))
        
        # 숫자 일치율
        num_match = 0.0
        if truth_nums:
            num_match = len(truth_nums & answer_nums) / len(truth_nums)
        else:
            num_match = 1.0  # 숫자가 없으면 만점
        
        # 단위 일치율
        unit_match = 0.0
        if truth_units:
            unit_match = len(truth_units & answer_units) / len(truth_units)
        else:
            unit_match = 1.0  # 단위가 없으면 만점
        
        # 사실적 정확도: 숫자(50%) + 단위(30%) + 토큰(20%)
        factual_correctness = 0.5 * num_match + 0.3 * unit_match + 0.2 * semantic_similarity
        
        # 최종 점수: 의미 유사도(40%) + 사실적 정확도(60%)
        final_score = 0.4 * semantic_similarity + 0.6 * factual_correctness
        
        return {
            'score': round(final_score, 4),
            'semantic_similarity': round(semantic_similarity, 4),
            'factual_correctness': round(factual_correctness, 4),
            'details': {
                'num_match_ratio': round(num_match, 4),
                'unit_match_ratio': round(unit_match, 4),
                'truth_nums': list(truth_nums),
                'answer_nums': list(answer_nums)
            }
        }
    
    # ============================================================
    # 3. Context Precision (문맥 정밀도) - 검색 효율성
    # ============================================================
    
    @staticmethod
    def context_precision(
        question: str,
        contexts: List[str],
        answer: str,
        ground_truth: str
    ) -> Dict[str, Any]:
        """
        Context Precision (문맥 정밀도)
        
        검색된 자료(contexts) 중 실제로 답변에 사용된/필요한 자료의 비율.
        검색 효율성을 측정.
        
        평가 질문: "엉뚱한 자료를 가져와서 헷갈리지 않았나?"
        
        Args:
            question: 질문
            contexts: 검색된 자료 리스트
            answer: 생성된 답변
            ground_truth: 정답
            
        Returns:
            {
                'score': float (0~1),
                'relevant_contexts': int,
                'total_contexts': int,
                'precision': float
            }
            
        Reference:
            Es et al. (2023). RAGAS: Automated Evaluation of 
            Retrieval Augmented Generation. arXiv:2309.15217
        """
        if not contexts:
            return {
                'score': 0.0,
                'relevant_contexts': 0,
                'total_contexts': 0,
                'precision': 0.0
            }
        
        # 질문과 정답의 키 토큰 추출
        question_tokens = set(RAGCoreMetrics.tokenize(question))
        truth_tokens = set(RAGCoreMetrics.tokenize(ground_truth))
        answer_tokens = set(RAGCoreMetrics.tokenize(answer))
        
        # 관련성 있는 토큰 = 질문 + 정답 + 답변의 합집합
        relevant_tokens = question_tokens | truth_tokens | answer_tokens
        
        # 각 context가 관련성이 있는지 평가
        relevant_count = 0
        
        for context in contexts:
            context_tokens = set(RAGCoreMetrics.tokenize(context))
            
            if not context_tokens:
                continue
            
            # Context가 관련 토큰과 얼마나 겹치는지
            overlap = len(context_tokens & relevant_tokens) / len(context_tokens)
            
            # 30% 이상 겹치면 관련성 있음으로 간주
            if overlap >= 0.3:
                relevant_count += 1
        
        total_contexts = len(contexts)
        precision = relevant_count / total_contexts
        
        return {
            'score': round(precision, 4),
            'relevant_contexts': relevant_count,
            'total_contexts': total_contexts,
            'precision': round(precision, 4)
        }
    
    # ============================================================
    # 종합 평가
    # ============================================================
    
    @staticmethod
    def evaluate_all(
        question: str,
        answer: str,
        ground_truth: str,
        contexts: List[str]
    ) -> Dict[str, Any]:
        """
        RAG 시스템 핵심 3대 지표 종합 평가
        
        Returns:
            {
                'faithfulness': {...},
                'answer_correctness': {...},
                'context_precision': {...},
                'overall_score': float
            }
        """
        # 1. Faithfulness
        faith = RAGCoreMetrics.faithfulness(answer, contexts)
        
        # 2. Answer Correctness
        correctness = RAGCoreMetrics.answer_correctness(answer, ground_truth)
        
        # 3. Context Precision
        precision = RAGCoreMetrics.context_precision(question, contexts, answer, ground_truth)
        
        # 종합 점수 (가중 평균)
        # Faithfulness(40%) + Answer Correctness(40%) + Context Precision(20%)
        overall = (
            0.4 * faith['score'] +
            0.4 * correctness['score'] +
            0.2 * precision['score']
        )
        
        return {
            'faithfulness': faith,
            'answer_correctness': correctness,
            'context_precision': precision,
            'overall_score': round(overall, 4)
        }


def main():
    """테스트"""
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("=" * 70)
    print("RAG 시스템 핵심 평가 지표 테스트")
    print("=" * 70)
    
    # 테스트 데이터
    question = "AI 플랫폼의 기본 관리자 아이디와 비밀번호는?"
    answer = "기본 관리자 계정의 아이디는 KWATER이고 비밀번호도 KWATER입니다."
    ground_truth = "기본 관리자 계정의 아이디와 비밀번호는 모두 KWATER입니다."
    contexts = [
        "기본 관리자 계정: 아이디 KWATER, 비밀번호 KWATER",
        "AI 플랫폼 접속 주소는 http://waio-portal-vip:10011 입니다.",
        "시스템은 4가지 영역으로 구성됩니다."
    ]
    
    print(f"\n질문: {question}")
    print(f"답변: {answer}")
    print(f"정답: {ground_truth}")
    print(f"참고 자료 수: {len(contexts)}개")
    
    print("\n" + "=" * 70)
    
    # 평가 실행
    metrics = RAGCoreMetrics()
    results = metrics.evaluate_all(question, answer, ground_truth, contexts)
    
    print("\n📊 RAG 핵심 평가 결과:")
    print()
    print(f"1️⃣  Faithfulness (충실성):        {results['faithfulness']['score']*100:>6.1f}%")
    print(f"   - 자료 기반 문장: {results['faithfulness']['supported_claims']}/{results['faithfulness']['total_claims']}")
    print(f"   - 환각 방지 성공: {results['faithfulness']['support_ratio']*100:.1f}%")
    print()
    print(f"2️⃣  Answer Correctness (답변 정확도): {results['answer_correctness']['score']*100:>6.1f}%")
    print(f"   - 의미 유사도: {results['answer_correctness']['semantic_similarity']*100:.1f}%")
    print(f"   - 사실적 정확도: {results['answer_correctness']['factual_correctness']*100:.1f}%")
    print()
    print(f"3️⃣  Context Precision (문맥 정밀도):  {results['context_precision']['score']*100:>6.1f}%")
    print(f"   - 관련 자료: {results['context_precision']['relevant_contexts']}/{results['context_precision']['total_contexts']}")
    print(f"   - 검색 효율성: {results['context_precision']['precision']*100:.1f}%")
    print()
    print("=" * 70)
    print(f"🏆 종합 점수:                     {results['overall_score']*100:>6.1f}%")
    print("=" * 70)
    
    print("\n✅ 테스트 완료!")


if __name__ == "__main__":
    main()



