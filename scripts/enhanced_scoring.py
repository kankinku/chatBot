#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
도메인 특화 강화 평가 지표

숫자, 단위, 도메인 키워드를 강조하는 평가 방식
"""

import re
from typing import List, Dict, Any


class DomainSpecificScoring:
    """도메인 특화 평가 (v5 개선 버전)"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """텍스트 정규화"""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    @staticmethod
    def extract_numbers(text: str) -> set:
        """숫자 추출 (날짜, URL, 계정 포함)"""
        # 숫자 패턴: 정수, 소수, 날짜, IP, 포트 등
        numbers = set(re.findall(r'\d+(?:[.,]\d+)?', text))
        return numbers
    
    @staticmethod
    def extract_units(text: str) -> set:
        """단위 추출"""
        # 단위 패턴: %, ℃, mg/L, ppm 등
        units = set(re.findall(r'[%°℃㎎]+|(?:mg|ppm|rpm|kwh|kg)/[lL]?', text, re.IGNORECASE))
        return units
    
    @staticmethod
    def extract_keywords(text: str, exclude_nums: set, exclude_units: set) -> set:
        """일반 키워드 추출 (숫자/단위 제외)"""
        # 한글, 영문 2자 이상
        keywords = set(re.findall(r'[\w\-/]+', text))
        keywords = {k for k in keywords if len(k) >= 2 and k not in exclude_nums and k not in exclude_units}
        return keywords
    
    @staticmethod
    def units_equivalent(u1: str, u2: str) -> bool:
        """단위 동의어 체크"""
        synonyms = [
            {"mg/l", "ppm", "㎎/l"},
            {"℃", "°c", "도"},
            {"%", "percent"},
        ]
        u1_lower = u1.lower().strip()
        u2_lower = u2.lower().strip()
        if u1_lower == u2_lower:
            return True
        for group in synonyms:
            if u1_lower in group and u2_lower in group:
                return True
        return False
    
    @staticmethod
    def score_answer_v5_style(
        pred: str,
        gold: str,
        keywords: List[str] = None
    ) -> Dict[str, Any]:
        """
        v5 스타일 평가 (가중치 적용)
        
        Returns:
            {
                'total_score': float,
                'numeric_score': float,
                'unit_score': float,
                'keyword_score': float,
                'details': {...}
            }
        """
        pred_norm = DomainSpecificScoring.normalize_text(pred)
        gold_norm = DomainSpecificScoring.normalize_text(gold)
        
        # 정답이 "없음"인 경우
        if gold_norm in {"없음", "없다", "없습니다", "none", "no"}:
            is_correct = pred_norm.startswith("문서에서") and ("없" in pred_norm or "찾을 수" in pred_norm)
            return {
                'total_score': 1.0 if is_correct else 0.0,
                'numeric_score': 0.0,
                'unit_score': 0.0,
                'keyword_score': 1.0 if is_correct else 0.0,
                'details': {'type': 'negative_answer'}
            }
        
        # 숫자, 단위, 키워드 추출
        gold_nums = DomainSpecificScoring.extract_numbers(gold_norm)
        gold_units = DomainSpecificScoring.extract_units(gold_norm)
        gold_keywords = DomainSpecificScoring.extract_keywords(gold_norm, gold_nums, gold_units)
        
        pred_nums = DomainSpecificScoring.extract_numbers(pred_norm)
        pred_units = DomainSpecificScoring.extract_units(pred_norm)
        
        # 점수 계산 (v5 가중치 적용)
        hit = 0.0
        total = 0.0
        
        numeric_hit = 0.0
        unit_hit = 0.0
        keyword_hit = 0.0
        
        # 1. 숫자 (가중치 1.5)
        if gold_nums:
            total += 1.5
            # 숫자가 예측에 포함되어 있는지
            num_match = any(n in pred_norm for n in gold_nums)
            if num_match:
                hit += 1.5
                numeric_hit = 1.0
        
        # 2. 단위 (가중치 1.3)
        if gold_units:
            total += 1.3
            unit_found = False
            
            # 직접 매칭
            for u in gold_units:
                if u.lower() in pred_norm:
                    unit_found = True
                    break
            
            # 동의어 매칭
            if not unit_found:
                for gold_u in gold_units:
                    for pred_u in pred_units:
                        if DomainSpecificScoring.units_equivalent(gold_u, pred_u):
                            unit_found = True
                            break
                    if unit_found:
                        break
            
            if unit_found:
                hit += 1.3
                unit_hit = 1.0
        
        # 3. 일반 키워드 (가중치 1.0)
        if gold_keywords:
            total += 1.0
            # 키워드 매칭
            kw_match = any(k in pred_norm for k in gold_keywords)
            if kw_match:
                hit += 1.0
                keyword_hit = 1.0
        
        # 최종 점수
        total_score = (hit / total) if total > 0 else 0.0
        
        return {
            'total_score': round(total_score, 4),
            'numeric_score': numeric_hit,
            'unit_score': unit_hit,
            'keyword_score': keyword_hit,
            'details': {
                'gold_nums': list(gold_nums),
                'gold_units': list(gold_units),
                'gold_keywords': list(gold_keywords)[:5],  # 상위 5개만
                'pred_nums': list(pred_nums),
                'pred_units': list(pred_units),
            }
        }
    
    @staticmethod
    def score_numeric_accuracy(pred: str, gold: str) -> float:
        """
        숫자 정확도만 평가 (도메인 특화 강조)
        
        Returns:
            0.0 ~ 1.0 (숫자 매칭 비율)
        """
        pred_norm = DomainSpecificScoring.normalize_text(pred)
        gold_norm = DomainSpecificScoring.normalize_text(gold)
        
        gold_nums = DomainSpecificScoring.extract_numbers(gold_norm)
        
        if not gold_nums:
            return 1.0  # 숫자가 없으면 만점
        
        # 각 숫자가 예측에 포함되었는지 확인
        matched = sum(1 for n in gold_nums if n in pred_norm)
        return matched / len(gold_nums)
    
    @staticmethod
    def score_unit_accuracy(pred: str, gold: str) -> float:
        """
        단위 정확도만 평가
        
        Returns:
            0.0 ~ 1.0 (단위 매칭 비율)
        """
        pred_norm = DomainSpecificScoring.normalize_text(pred)
        gold_norm = DomainSpecificScoring.normalize_text(gold)
        
        gold_units = DomainSpecificScoring.extract_units(gold_norm)
        pred_units = DomainSpecificScoring.extract_units(pred_norm)
        
        if not gold_units:
            return 1.0  # 단위가 없으면 만점
        
        # 각 단위가 매칭되는지 확인 (동의어 포함)
        matched = 0
        for gold_u in gold_units:
            if gold_u.lower() in pred_norm:
                matched += 1
            else:
                # 동의어 체크
                for pred_u in pred_units:
                    if DomainSpecificScoring.units_equivalent(gold_u, pred_u):
                        matched += 1
                        break
        
        return matched / len(gold_units)


def main():
    """테스트"""
    print("=" * 70)
    print("도메인 특화 평가 테스트")
    print("=" * 70)
    
    # 테스트 케이스
    pred = "발주기관은 한국수자원공사입니다. 주소는 http://10.103.11.112:10011 입니다."
    gold = "발주기관은 한국수자원공사이며, 주소는 http://10.103.11.112:10011 또는 http://waio-portal-vip:10011입니다."
    
    print(f"\n예측: {pred}")
    print(f"정답: {gold}")
    print("\n" + "=" * 70)
    
    scorer = DomainSpecificScoring()
    result = scorer.score_answer_v5_style(pred, gold)
    
    print("\n📊 평가 결과:")
    print(f"  🏆 종합 점수: {result['total_score']*100:.1f}%")
    print(f"  🔢 숫자 정확도: {result['numeric_score']*100:.0f}%")
    print(f"  📏 단위 정확도: {result['unit_score']*100:.0f}%")
    print(f"  🔑 키워드 정확도: {result['keyword_score']*100:.0f}%")
    
    print("\n📋 상세 정보:")
    print(f"  정답 숫자: {result['details']['gold_nums']}")
    print(f"  예측 숫자: {result['details']['pred_nums']}")
    print(f"  정답 단위: {result['details']['gold_units']}")
    print(f"  예측 단위: {result['details']['pred_units']}")
    
    # 개별 평가
    numeric_acc = scorer.score_numeric_accuracy(pred, gold)
    unit_acc = scorer.score_unit_accuracy(pred, gold)
    
    print("\n🎯 도메인 특화 강조:")
    print(f"  숫자만 평가: {numeric_acc*100:.1f}%")
    print(f"  단위만 평가: {unit_acc*100:.1f}%")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

