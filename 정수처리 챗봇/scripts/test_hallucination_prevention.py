"""
환상 방지 시스템 테스트

챗봇이 주어진 정보 외의 데이터로 추론하는 것을 막는 시스템을 테스트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from core.llm.hallucination_prevention import create_hallucination_prevention
from core.llm.answer_generator import AnswerGenerator

def test_hallucination_detection():
    """환상 감지 테스트"""
    print("=== 환상 감지 테스트 ===")
    
    prevention = create_hallucination_prevention()
    
    # 테스트 케이스들
    test_cases = [
        {
            "name": "일반 지식 사용",
            "answer": "일반적으로 알려진 바와 같이 LSTM 모델은 시계열 예측에 효과적입니다.",
            "context": "N-beats 모델이 최종 선정되었습니다.",
            "question": "최종 선정된 모델은 무엇인가요?",
            "expected_hallucination": True
        },
        {
            "name": "추측성 답변",
            "answer": "아마도 XGB 모델이 사용되었을 것으로 추정됩니다.",
            "context": "N-beats 모델이 최종 선정되었습니다.",
            "question": "최종 선정된 모델은 무엇인가요?",
            "expected_hallucination": True
        },
        {
            "name": "수치 환상",
            "answer": "MAE는 0.68, MSE는 2.37입니다.",
            "context": "N-beats 모델이 최종 선정되었습니다.",
            "question": "모델 성능은 어떤가요?",
            "expected_hallucination": True
        },
        {
            "name": "정상 답변",
            "answer": "문서에 따르면 N-beats 모델이 최종 선정되었습니다.",
            "context": "N-beats 모델이 최종 선정되었습니다.",
            "question": "최종 선정된 모델은 무엇인가요?",
            "expected_hallucination": False
        },
        {
            "name": "컨텍스트 외 정보",
            "answer": "일반적인 정수처리에서는 PAC를 사용합니다.",
            "context": "N-beats 모델이 최종 선정되었습니다.",
            "question": "응집제는 무엇을 사용하나요?",
            "expected_hallucination": True
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n테스트 {i}: {test_case['name']}")
        print(f"답변: {test_case['answer']}")
        print(f"컨텍스트: {test_case['context']}")
        print(f"질문: {test_case['question']}")
        
        detection = prevention.detect_hallucination(
            test_case['answer'],
            test_case['context'],
            test_case['question']
        )
        
        print(f"환상 감지: {detection.is_hallucination}")
        print(f"환상 유형: {detection.hallucination_type}")
        print(f"신뢰도: {detection.confidence:.2f}")
        print(f"감지된 패턴: {detection.detected_patterns}")
        
        if detection.is_hallucination == test_case['expected_hallucination']:
            print("✅ 테스트 통과")
        else:
            print("❌ 테스트 실패")
        
        if detection.suggested_correction:
            print(f"수정 제안: {detection.suggested_correction}")

def test_answer_correction():
    """답변 수정 테스트"""
    print("\n=== 답변 수정 테스트 ===")
    
    prevention = create_hallucination_prevention()
    
    test_answer = "일반적으로 알려진 바와 같이 LSTM 모델이 사용됩니다."
    test_context = "N-beats 모델이 최종 선정되었습니다."
    test_question = "최종 선정된 모델은 무엇인가요?"
    
    print(f"원본 답변: {test_answer}")
    
    corrected_answer, was_corrected = prevention.validate_answer_strictly(
        test_answer, test_context, test_question
    )
    
    print(f"수정된 답변: {corrected_answer}")
    print(f"수정 여부: {was_corrected}")

def test_strict_prompt():
    """엄격한 프롬프트 테스트"""
    print("\n=== 엄격한 프롬프트 테스트 ===")
    
    prevention = create_hallucination_prevention()
    
    context = "N-beats 모델이 최종 선정되었습니다. MAE는 0.68입니다."
    question = "최종 선정된 모델은 무엇인가요?"
    
    strict_prompt = prevention.create_strict_prompt(context, question)
    
    print("엄격한 프롬프트:")
    print(strict_prompt)

def test_answer_generator_integration():
    """답변 생성기 통합 테스트"""
    print("\n=== 답변 생성기 통합 테스트 ===")
    
    try:
        # 답변 생성기 초기화
        generator = AnswerGenerator()
        
        # 테스트 컨텍스트
        from core.document.pdf_processor import TextChunk
        
        test_chunks = [
            TextChunk(
                content="N-beats 모델이 최종 선정되었습니다. MAE는 0.68, MSE는 2.37입니다.",
                page_number=1,
                chunk_id="test_chunk_1",
                metadata={"process_type": "AI모델"}
            )
        ]
        
        question = "최종 선정된 모델은 무엇인가요?"
        
        print(f"질문: {question}")
        print(f"컨텍스트: {test_chunks[0].content}")
        
        # 답변 생성
        answer = generator.generate_context_answer(question, test_chunks)
        
        print(f"생성된 답변: {answer.content}")
        print(f"신뢰도: {answer.confidence}")
        print(f"메타데이터: {answer.metadata}")
        
    except Exception as e:
        print(f"통합 테스트 실패: {e}")

def main():
    """메인 테스트 함수"""
    print("환상 방지 시스템 테스트 시작")
    print("=" * 50)
    
    try:
        test_hallucination_detection()
        test_answer_correction()
        test_strict_prompt()
        test_answer_generator_integration()
        
        print("\n" + "=" * 50)
        print("모든 테스트 완료!")
        
    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
