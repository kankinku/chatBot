#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGAS 정식 평가 방법 (LLM 기반)

Es et al. (2023)의 정식 Faithfulness 평가 구현
"""

import sys
import re
import json
from typing import List, Dict, Any
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.generation.llm_client import OllamaClient
from config.model_config import LLMModelConfig


class RAGASMetrics:
    """RAGAS 정식 평가 지표 (LLM 기반)"""
    
    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: LLM 클라이언트 (없으면 기본 Ollama 사용)
        """
        if llm_client is None:
            llm_config = LLMModelConfig(
                host="localhost",
                port=11434,
                model_name="qwen2.5:7b-instruct-q4_K_M"
            )
            self.llm = OllamaClient(llm_config)
        else:
            self.llm = llm_client
    
    # ============================================================
    # 1. Faithfulness (정식 RAGAS 방법)
    # ============================================================
    
    def faithfulness_llm_based(
        self,
        question: str,
        answer: str,
        contexts: List[str]
    ) -> Dict[str, Any]:
        """
        Faithfulness (충실성) - RAGAS 정식 LLM 기반 평가
        
        단계:
        1. LLM으로 답변에서 진술(statements) 추출
        2. LLM으로 각 진술이 context에 의해 지지되는지 검증
        3. 지지된 진술 비율로 점수 계산
        
        Reference:
            Es et al. (2023). RAGAS: Automated Evaluation of 
            Retrieval Augmented Generation. arXiv:2309.15217
        """
        if not answer or not contexts:
            return {
                'score': 0.0,
                'statements': [],
                'verified': [],
                'supported_count': 0,
                'total_count': 0
            }
        
        # Step 1: 진술 추출 (Statement Extraction)
        statements = self._extract_statements(question, answer)
        
        if not statements:
            return {
                'score': 0.0,
                'statements': [],
                'verified': [],
                'supported_count': 0,
                'total_count': 0
            }
        
        # Step 2: 진술 검증 (Verification)
        verified_results = []
        supported_count = 0
        
        combined_context = '\n\n'.join(contexts[:5])  # 상위 5개 context만 사용
        
        for statement in statements:
            is_supported = self._verify_statement(statement, combined_context)
            verified_results.append({
                'statement': statement,
                'supported': is_supported
            })
            if is_supported:
                supported_count += 1
        
        # Step 3: 점수 계산
        total_count = len(statements)
        score = supported_count / total_count if total_count > 0 else 0.0
        
        return {
            'score': round(score, 4),
            'statements': statements,
            'verified': verified_results,
            'supported_count': supported_count,
            'total_count': total_count
        }
    
    def _extract_statements(self, question: str, answer: str) -> List[str]:
        """
        LLM을 사용하여 답변에서 진술 추출
        
        RAGAS 방식: 답변의 각 문장을 더 세밀한 진술로 분해
        """
        prompt = f"""다음 답변에서 검증 가능한 진술(statement)들을 추출하세요.

질문: {question}

답변: {answer}

지침:
- 답변의 각 문장을 독립적인 사실 진술로 분해하세요
- 하나의 문장에 여러 사실이 있으면 분리하세요
- 각 진술은 한 줄에 하나씩 나열하세요
- 진술만 나열하고 번호나 부가 설명은 생략하세요

진술 목록:"""

        try:
            response = self.llm.generate(prompt, timeout=10)
            
            # 응답을 줄 단위로 분리
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            # 번호나 불릿 제거
            statements = []
            for line in lines:
                # "1. ", "- ", "• " 등 제거
                cleaned = re.sub(r'^[\d\-\•\*\.]\s*', '', line)
                if len(cleaned) > 5:  # 최소 길이
                    statements.append(cleaned)
            
            return statements[:10]  # 최대 10개
            
        except Exception as e:
            # LLM 실패 시 간단한 문장 분리로 폴백
            return [s.strip() for s in re.split(r'[.!?]\s+', answer) if len(s.strip()) > 10]
    
    def _verify_statement(self, statement: str, context: str) -> bool:
        """
        LLM을 사용하여 진술이 context에 의해 지지되는지 검증
        
        RAGAS 방식: LLM이 Yes/No로 판단
        """
        prompt = f"""다음 진술이 제공된 맥락(문서)에 의해 지지되는지 판단하세요.

맥락:
{context[:1000]}

진술: {statement}

지침:
- 진술의 내용이 맥락에 명시되어 있거나 합리적으로 추론 가능하면 "Yes"
- 진술의 내용이 맥락에 없거나 모순되면 "No"
- 한 단어로만 답변하세요: Yes 또는 No

답변:"""

        try:
            response = self.llm.generate(prompt, timeout=10)
            response_lower = response.lower().strip()
            
            # Yes 변형들 확인
            if any(word in response_lower for word in ['yes', '예', '네', '맞', '지지']):
                return True
            else:
                return False
                
        except Exception as e:
            # LLM 실패 시 토큰 오버랩으로 폴백
            statement_tokens = set(re.findall(r'[\w]+', statement.lower()))
            context_tokens = set(re.findall(r'[\w]+', context.lower()))
            
            if statement_tokens:
                overlap = len(statement_tokens & context_tokens) / len(statement_tokens)
                return overlap >= 0.7
            return False


def main():
    """테스트"""
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("=" * 80)
    print("RAGAS 정식 Faithfulness 평가 (LLM 기반)")
    print("=" * 80)
    print()
    print("⚠️  주의: 이 방식은 각 진술마다 LLM을 2번 호출하므로 시간이 오래 걸립니다!")
    print("         (30개 질문 x 평균 3개 진술 x 2번 LLM 호출 = 약 180회)")
    print()
    
    # 테스트 데이터
    question = "AI 플랫폼의 기본 관리자 아이디와 비밀번호는?"
    answer = "기본 관리자 계정의 아이디는 KWATER이고 비밀번호도 KWATER입니다. 관리자 아이디와 비밀번호는 지사에 문의하여 얻을 수 있습니다."
    contexts = [
        "기본 관리자 계정: 아이디 KWATER, 비밀번호 KWATER",
        "AI 플랫폼 접속 주소는 http://waio-portal-vip:10011 입니다."
    ]
    
    print(f"질문: {question}")
    print(f"\n답변: {answer}")
    print(f"\n참고 자료 {len(contexts)}개")
    
    print("\n" + "=" * 80)
    print("평가 시작...")
    print("=" * 80)
    
    try:
        evaluator = RAGASMetrics()
        result = evaluator.faithfulness_llm_based(question, answer, contexts)
        
        print(f"\n[Step 1] LLM으로 진술 추출:")
        print(f"  총 {len(result['statements'])}개 진술 추출됨")
        for i, stmt in enumerate(result['statements'], 1):
            print(f"    {i}. {stmt}")
        
        print(f"\n[Step 2] LLM으로 각 진술 검증:")
        for item in result['verified']:
            status = "✅ 지지됨" if item['supported'] else "❌ 미지지"
            print(f"  {status}: {item['statement'][:60]}...")
        
        print(f"\n[Step 3] 점수 계산:")
        print(f"  지지된 진술: {result['supported_count']}개")
        print(f"  전체 진술:   {result['total_count']}개")
        print(f"  Faithfulness = {result['supported_count']}/{result['total_count']} = {result['score']*100:.1f}%")
        
        print("\n" + "=" * 80)
        print(f"📊 최종 Faithfulness 점수: {result['score']*100:.1f}%")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        print("\nLLM이 실행 중인지 확인하세요: curl http://localhost:11434/api/tags")


if __name__ == "__main__":
    main()



