#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 평가 모듈 (Unified Evaluation Module)

모든 평가 지표를 한 번에 실행하고 비교할 수 있는 통합 인터페이스
다른 프로젝트에서도 쉽게 재사용 가능

사용 예시:
    from scripts.unified_evaluation import UnifiedEvaluator
    
    evaluator = UnifiedEvaluator()
    results = evaluator.evaluate_all(
        question="AI 플랫폼의 기본 관리자 아이디는?",
        prediction="기본 관리자 아이디는 KWATER입니다.",
        ground_truth="기본 관리자 아이디는 KWATER입니다.",
        contexts=["관리자 계정: KWATER", "시스템 접속 정보..."]
    )
    
    # 모든 지표가 포함된 종합 결과
    print(results)
"""

from typing import List, Dict, Any
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.academic_metrics import AcademicMetrics
from scripts.rag_core_metrics import RAGCoreMetrics
from scripts.enhanced_scoring import DomainSpecificScoring


class UnifiedEvaluator:
    """
    통합 평가자
    
    4가지 평가 체계를 한 번에 실행:
    1. 기본 Score (v5 방식) - 도메인 가중치
    2. 도메인 특화 평가 - 숫자/단위 정확도
    3. RAG 핵심 3대 지표 - Faithfulness, Correctness, Precision
    4. 학술 표준 지표 - F1, ROUGE-L, BLEU, Exact Match
    """
    
    def __init__(self):
        """초기화"""
        self.domain_scorer = DomainSpecificScoring()
        self.academic = AcademicMetrics()
        self.rag = RAGCoreMetrics()
    
    def evaluate_all(
        self,
        question: str,
        prediction: str,
        ground_truth: str,
        contexts: List[str] = None,
        keywords: List[str] = None
    ) -> Dict[str, Any]:
        """
        모든 평가 지표를 한 번에 실행
        
        Args:
            question: 질문
            prediction: 생성된 답변
            ground_truth: 정답
            contexts: 참고 자료 리스트 (RAG 평가에 필요)
            keywords: 필수 키워드 리스트 (선택)
            
        Returns:
            {
                'basic_score': Dict,           # 기본 Score (v5 방식)
                'domain_specific': Dict,       # 도메인 특화 평가
                'rag_metrics': Dict,          # RAG 핵심 3대 지표
                'academic_metrics': Dict,      # 학술 표준 지표
                'summary': Dict               # 주요 점수 요약
            }
        """
        contexts = contexts or []
        keywords = keywords or []
        
        # 1. 기본 Score (v5 방식)
        basic_score = self._evaluate_basic_v5(prediction, ground_truth, keywords)
        
        # 2. 도메인 특화 평가
        domain_specific = self.domain_scorer.score_answer_v5_style(
            prediction,
            ground_truth,
            keywords
        )
        
        # 3. RAG 핵심 3대 지표
        rag_metrics = {}
        if contexts:
            rag_metrics = self.rag.evaluate_all(
                question,
                prediction,
                ground_truth,
                contexts
            )
        
        # 4. 학술 표준 지표
        academic_metrics = self.academic.evaluate_all(prediction, ground_truth)
        
        # 주요 점수 요약
        summary = self._create_summary(
            basic_score,
            domain_specific,
            rag_metrics,
            academic_metrics
        )
        
        return {
            'basic_score': basic_score,
            'domain_specific': domain_specific,
            'rag_metrics': rag_metrics,
            'academic_metrics': academic_metrics,
            'summary': summary
        }
    
    def _evaluate_basic_v5(
        self,
        prediction: str,
        gold_answer: str,
        keywords: List[str]
    ) -> Dict[str, Any]:
        """
        기본 Score (v5 방식) 평가
        
        run_qa_benchmark.py의 score_answer 로직과 동일
        """
        import re
        
        def normalize_text(t: str) -> str:
            return t.strip().lower()
        
        def units_equivalent(u1: str, u2: str) -> bool:
            """단위 동의어 체크"""
            synonyms = [
                {"mg/l", "ppm"},
                {"㎎/l", "ppm", "mg/l"},
                {"℃", "°c", "도"},
            ]
            u1_lower = u1.lower().strip()
            u2_lower = u2.lower().strip()
            if u1_lower == u2_lower:
                return True
            for group in synonyms:
                if u1_lower in group and u2_lower in group:
                    return True
            return False
        
        p = normalize_text(prediction)
        g = normalize_text(gold_answer)
        
        # 정답이 "없음"인 경우 특별 처리
        if g in {"없음", "없다", "없습니다", "none", "no"}:
            score = 1.0 if p.startswith("문서에서 해당 정보를 확인할 수 없습니다") or p.startswith("문서에서 관련 정보를 찾을 수 없습니다") else 0.0
            return {
                'score': score,
                'type': 'negative_answer',
                'details': {}
            }
        
        # v5 로직: numeric + unit + keyword 가중치 적용
        keywords_set = set(re.findall(r"[\w\-/%°℃]+", g))
        nums = set(re.findall(r"\d+(?:[\.,]\d+)?", g))
        units = set(re.findall(r"[a-z%°℃/㎎]+", g, re.IGNORECASE))
        
        hit = 0.0
        total = 0.0
        
        numeric_hit = 0.0
        unit_hit = 0.0
        keyword_hit = 0.0
        
        # numeric에 높은 가중치 (1.5)
        if nums:
            total += 1.5
            if any(n in p for n in nums):
                hit += 1.5
                numeric_hit = 1.0
        
        # units에 가중치 (1.3)
        if units:
            total += 1.3
            uh = 0.0
            for u in units:
                if u.lower() in p:
                    uh = 1.3
                    break
            # unit synonym 매핑 시도
            if uh == 0.0:
                for u in units:
                    for v in ["mg/l", "ppm", "℃", "°c", "㎎/l"]:
                        if v in p and units_equivalent(u, v):
                            uh = 1.3
                            break
                    if uh > 0:
                        break
            hit += uh
            if uh > 0:
                unit_hit = 1.0
        
        # general keywords (1.0 가중치)
        kw = {k for k in keywords_set if k not in nums and k not in units and len(k) >= 2}
        if kw:
            total += 1.0
            if any(k in p for k in kw):
                hit += 1.0
                keyword_hit = 1.0
        
        final_score = (hit / total) if total > 0 else 0.0
        
        return {
            'score': round(final_score, 4),
            'type': 'weighted',
            'numeric_hit': numeric_hit,
            'unit_hit': unit_hit,
            'keyword_hit': keyword_hit,
            'details': {
                'nums': list(nums),
                'units': list(units),
                'keywords': list(kw)[:5]
            }
        }
    
    def _create_summary(
        self,
        basic: Dict,
        domain: Dict,
        rag: Dict,
        academic: Dict
    ) -> Dict[str, Any]:
        """주요 점수 요약 생성"""
        summary = {
            # 기본 점수
            'basic_v5_score': basic.get('score', 0.0),
            
            # 도메인 특화
            'domain_total_score': domain.get('total_score', 0.0),
            'numeric_accuracy': domain.get('numeric_score', 0.0),
            'unit_accuracy': domain.get('unit_score', 0.0),
            'keyword_accuracy': domain.get('keyword_score', 0.0),
            
            # 학술 지표
            'token_f1': academic.get('token_f1', {}).get('f1', 0.0),
            'rouge_l': academic.get('rouge_l', {}).get('f1', 0.0),
            'bleu_2': academic.get('bleu_2', 0.0),
            'exact_match': academic.get('exact_match', 0.0),
        }
        
        # RAG 지표 (있는 경우)
        if rag:
            summary.update({
                'faithfulness': rag.get('faithfulness', {}).get('score', 0.0),
                'answer_correctness': rag.get('answer_correctness', {}).get('score', 0.0),
                'context_precision': rag.get('context_precision', {}).get('score', 0.0),
                'rag_overall': rag.get('overall_score', 0.0),
            })
        
        return summary
    
    def evaluate_batch(
        self,
        qa_pairs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        여러 QA 쌍을 배치로 평가
        
        Args:
            qa_pairs: QA 쌍 리스트
                [
                    {
                        'question': str,
                        'prediction': str,
                        'ground_truth': str,
                        'contexts': List[str],  # optional
                        'keywords': List[str]   # optional
                    },
                    ...
                ]
        
        Returns:
            {
                'individual_results': List[Dict],  # 각 질문별 평가 결과
                'aggregated_stats': Dict           # 전체 통계
            }
        """
        results = []
        
        for item in qa_pairs:
            result = self.evaluate_all(
                question=item['question'],
                prediction=item['prediction'],
                ground_truth=item['ground_truth'],
                contexts=item.get('contexts', []),
                keywords=item.get('keywords', [])
            )
            results.append(result)
        
        # 전체 통계 계산
        stats = self._aggregate_stats(results)
        
        return {
            'individual_results': results,
            'aggregated_stats': stats
        }
    
    def _aggregate_stats(self, results: List[Dict]) -> Dict[str, Any]:
        """배치 결과의 평균 통계 계산"""
        if not results:
            return {}
        
        summaries = [r['summary'] for r in results]
        
        # 각 지표의 평균 계산
        stats = {}
        for key in summaries[0].keys():
            values = [s.get(key, 0.0) for s in summaries]
            stats[f'avg_{key}'] = sum(values) / len(values) if values else 0.0
            stats[f'min_{key}'] = min(values) if values else 0.0
            stats[f'max_{key}'] = max(values) if values else 0.0
        
        stats['total_evaluated'] = len(results)
        
        return stats
    
    def print_results(self, results: Dict[str, Any], detailed: bool = False):
        """
        평가 결과를 보기 좋게 출력
        
        Args:
            results: evaluate_all() 또는 evaluate_batch()의 결과
            detailed: 상세 정보 출력 여부
        """
        print("=" * 80)
        print("📊 통합 평가 결과")
        print("=" * 80)
        
        # 배치 결과인 경우
        if 'aggregated_stats' in results:
            stats = results['aggregated_stats']
            print(f"\n총 평가 질문 수: {stats['total_evaluated']}개\n")
            
            print("=" * 80)
            print("평균 점수 요약")
            print("=" * 80)
            
            # 주요 지표만 출력
            key_metrics = [
                ('avg_basic_v5_score', '기본 Score (v5 방식)'),
                ('avg_domain_total_score', '도메인 특화 종합'),
                ('avg_numeric_accuracy', '숫자 정확도'),
                ('avg_unit_accuracy', '단위 정확도'),
                ('avg_token_f1', 'Token F1 (SQuAD)'),
                ('avg_rouge_l', 'ROUGE-L'),
                ('avg_faithfulness', 'Faithfulness (충실성)'),
                ('avg_answer_correctness', 'Answer Correctness'),
                ('avg_context_precision', 'Context Precision'),
            ]
            
            for key, label in key_metrics:
                if key in stats:
                    print(f"{label:30s}: {stats[key]*100:6.1f}%")
            
            return
        
        # 단일 결과인 경우
        summary = results.get('summary', {})
        
        print("\n1️⃣  기본 Score (v5 방식)")
        print(f"   종합 점수: {summary.get('basic_v5_score', 0)*100:.1f}%")
        
        print("\n2️⃣  도메인 특화 평가")
        print(f"   종합 점수: {summary.get('domain_total_score', 0)*100:.1f}%")
        print(f"   숫자 정확도: {summary.get('numeric_accuracy', 0)*100:.1f}%")
        print(f"   단위 정확도: {summary.get('unit_accuracy', 0)*100:.1f}%")
        print(f"   키워드 정확도: {summary.get('keyword_accuracy', 0)*100:.1f}%")
        
        if summary.get('faithfulness') is not None:
            print("\n3️⃣  RAG 핵심 3대 지표")
            print(f"   Faithfulness (충실성): {summary.get('faithfulness', 0)*100:.1f}%")
            print(f"   Answer Correctness (정확도): {summary.get('answer_correctness', 0)*100:.1f}%")
            print(f"   Context Precision (정밀도): {summary.get('context_precision', 0)*100:.1f}%")
            print(f"   RAG 종합 점수: {summary.get('rag_overall', 0)*100:.1f}%")
        
        print("\n4️⃣  학술 표준 지표")
        print(f"   Token F1: {summary.get('token_f1', 0)*100:.1f}%")
        print(f"   ROUGE-L: {summary.get('rouge_l', 0)*100:.1f}%")
        print(f"   BLEU-2: {summary.get('bleu_2', 0)*100:.1f}%")
        print(f"   Exact Match: {summary.get('exact_match', 0)*100:.1f}%")
        
        if detailed:
            print("\n" + "=" * 80)
            print("상세 정보")
            print("=" * 80)
            
            import json
            print(json.dumps(results, indent=2, ensure_ascii=False))
        
        print("\n" + "=" * 80)


def main():
    """사용 예시 및 테스트"""
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("=" * 80)
    print("통합 평가 모듈 테스트")
    print("=" * 80)
    
    # 테스트 데이터
    test_cases = [
        {
            'question': 'AI 플랫폼의 기본 관리자 아이디와 비밀번호는?',
            'prediction': '기본 관리자 계정의 아이디는 KWATER이고 비밀번호도 KWATER입니다.',
            'ground_truth': '기본 관리자 계정의 아이디와 비밀번호는 모두 KWATER입니다.',
            'contexts': [
                '기본 관리자 계정: 아이디 KWATER, 비밀번호 KWATER',
                'AI 플랫폼 접속 주소는 http://waio-portal-vip:10011 입니다.',
            ],
            'keywords': ['KWATER', '관리자', '아이디', '비밀번호']
        },
        {
            'question': '수질 기준 온도는?',
            'prediction': '수질 기준 온도는 25℃입니다.',
            'ground_truth': '수질 기준 온도는 25℃입니다.',
            'contexts': [
                '수질 검사 기준: 온도 25℃, pH 7.0',
            ],
            'keywords': ['수질', '온도', '25', '℃']
        }
    ]
    
    # 통합 평가자 생성
    evaluator = UnifiedEvaluator()
    
    print("\n[테스트 1] 단일 평가")
    print("-" * 80)
    result = evaluator.evaluate_all(**test_cases[0])
    evaluator.print_results(result)
    
    print("\n\n[테스트 2] 배치 평가")
    print("-" * 80)
    batch_results = evaluator.evaluate_batch(test_cases)
    evaluator.print_results(batch_results)
    
    print("\n✅ 테스트 완료!")


if __name__ == "__main__":
    main()

