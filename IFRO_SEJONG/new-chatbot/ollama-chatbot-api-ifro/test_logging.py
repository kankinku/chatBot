#!/usr/bin/env python3
"""
챗봇 로깅 시스템 테스트 스크립트
Docker Desktop에서 로그가 올바르게 표시되는지 확인하기 위한 테스트
"""

import requests
import json
import time
import sys
from datetime import datetime

def test_chatbot_logging():
    """챗봇 API를 호출하여 로깅 시스템을 테스트합니다."""
    
    # 챗봇 API 엔드포인트
    chatbot_url = "http://localhost:8008/api/ask"
    
    # 테스트 질문들
    test_questions = [
        "교통사고가 발생했을 때 어떻게 해야 하나요?",
        "신호등이 고장났을 때의 대응 방법은?",
        "도로에서 긴급차량이 지나갈 때 어떻게 해야 하나요?",
        "주차 위반 시 벌금은 얼마인가요?",
        "음주운전 처벌 기준은 무엇인가요?"
    ]
    
    print("🧪 챗봇 로깅 시스템 테스트 시작")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 테스트 {i}/5: {question}")
        
        try:
            # API 호출
            response = requests.post(
                chatbot_url,
                json={
                    "question": question,
                    "mode": "accuracy",
                    "k": "auto"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 응답 성공 | 신뢰도: {result.get('confidence', 0):.2f}")
                print(f"📄 답변 길이: {len(result.get('answer', ''))}자")
                print(f"📊 소스 수: {len(result.get('sources', []))}개")
            else:
                print(f"❌ API 오류: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 요청 실패: {str(e)}")
        
        # 다음 테스트 전 잠시 대기
        time.sleep(2)
    
    print("\n" + "=" * 50)
    print("🎉 테스트 완료!")
    print("\n📋 로그 확인 방법:")
    print("1. Docker Desktop에서 'chatbot-gpu' 컨테이너의 Logs 탭 확인")
    print("2. 또는 다음 명령어로 로그 파일 확인:")
    print("   tail -f ollama-chatbot-api-ifro/logs/chatbot_conversations.log")
    print("   tail -f ollama-chatbot-api-ifro/logs/qa_detailed.log")

def check_log_files():
    """로그 파일들이 생성되었는지 확인합니다."""
    import os
    from pathlib import Path
    
    log_dir = Path("logs")
    log_files = [
        "chatbot_conversations.log",
        "qa_detailed.log", 
        "conversations.jsonl"
    ]
    
    print("\n📁 로그 파일 상태 확인:")
    print("-" * 30)
    
    for log_file in log_files:
        file_path = log_dir / log_file
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✅ {log_file}: {size} bytes")
        else:
            print(f"❌ {log_file}: 파일 없음")

if __name__ == "__main__":
    print("🚀 챗봇 로깅 테스트 도구")
    print("Docker Desktop에서 로그 확인을 위한 테스트를 시작합니다.\n")
    
    # 챗봇 서비스 상태 확인
    try:
        health_response = requests.get("http://localhost:8008/healthz", timeout=5)
        if health_response.status_code == 200:
            print("✅ 챗봇 서비스가 실행 중입니다.")
        else:
            print("❌ 챗봇 서비스가 응답하지 않습니다.")
            sys.exit(1)
    except requests.exceptions.RequestException:
        print("❌ 챗봇 서비스에 연결할 수 없습니다.")
        print("   docker-compose -f docker-compose.gpu.yml up -d 명령으로 서비스를 시작하세요.")
        sys.exit(1)
    
    # 로깅 테스트 실행
    test_chatbot_logging()
    
    # 로그 파일 확인
    check_log_files()
    
    print("\n💡 추가 정보:")
    print("- Docker Desktop에서 실시간 로그를 확인하려면 'chatbot-gpu' 컨테이너의 Logs 탭을 사용하세요.")
    print("- 상세한 로그는 ollama-chatbot-api-ifro/logs/qa_detailed.log 파일에서 확인할 수 있습니다.")
