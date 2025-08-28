#!/usr/bin/env python3
"""
간단한 PDF QA 시스템 테스트 스크립트
"""

import sys
import os
import time
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).resolve().parents[1]))

from main import PDFQASystem

def test_basic_functionality():
    """기본 기능 테스트"""
    print("🧪 PDF QA 시스템 기본 기능 테스트")
    print("=" * 50)
    
    # 시스템 초기화
    print("1. 시스템 초기화 중...")
    system = PDFQASystem()
    
    if not system.initialize_components():
        print("❌ 시스템 초기화 실패")
        return False
    
    print("✅ 시스템 초기화 성공")
    
    # 시스템 상태 확인
    print("\n2. 시스템 상태 확인...")
    system.show_system_status()
    
    # PDF 목록 확인
    print("\n3. 저장된 PDF 목록 확인...")
    system.show_pdf_list()
    
    # 간단한 질문 테스트
    print("\n4. 간단한 질문 테스트...")
    test_questions = [
        "이 시스템은 무엇을 하는 시스템인가요?",
        "시스템의 주요 기능은 무엇인가요?",
        "사용자 관리 기능에 대해 설명해주세요."
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n질문 {i}: {question}")
        try:
            start_time = time.time()
            result = system.ask_question(question)
            response_time = time.time() - start_time
            
            print(f"답변: {result['answer']}")
            print(f"신뢰도: {result['confidence_score']:.2f}")
            print(f"처리시간: {response_time:.2f}초")
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
    
    # 시스템 정리
    print("\n5. 시스템 정리...")
    system.cleanup()
    print("✅ 테스트 완료")
    
    return True

def test_pdf_processing():
    """PDF 처리 테스트"""
    print("\n📄 PDF 처리 테스트")
    print("=" * 50)
    
    system = PDFQASystem()
    system.initialize_components()
    
    # 테스트용 PDF 파일 경로
    test_pdf = "./data/pdfs/misc/테스트용_회사정보.pdf"
    
    if os.path.exists(test_pdf):
        print(f"테스트 PDF 파일 발견: {test_pdf}")
        
        try:
            # PDF 처리
            print("PDF 처리 중...")
            result = system.process_pdf(test_pdf)
            print(f"✅ PDF 처리 완료: {result}")
            
            # 처리된 PDF로 질문 테스트
            print("\n처리된 PDF로 질문 테스트...")
            test_result = system.ask_question("회사명이 뭐야?")
            print(f"답변: {test_result['answer']}")
            
        except Exception as e:
            print(f"❌ PDF 처리 오류: {e}")
    else:
        print(f"❌ 테스트 PDF 파일을 찾을 수 없습니다: {test_pdf}")
    
    system.cleanup()

if __name__ == "__main__":
    print("🚀 PDF QA 시스템 테스트 시작")
    print("=" * 60)
    
    try:
        # 기본 기능 테스트
        test_basic_functionality()
        
        # PDF 처리 테스트
        test_pdf_processing()
        
        print("\n🎉 모든 테스트 완료!")
        
    except KeyboardInterrupt:
        print("\n\n👋 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


