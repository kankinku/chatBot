#!/usr/bin/env python3
"""
챗봇 로그 확인 스크립트

간단한 명령어로 챗봇 로그를 확인할 수 있는 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from utils.log_viewer import ChatbotLogViewer

def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("""
🤖 챗봇 로그 확인 도구

사용법:
  python check_logs.py [명령어] [옵션]

명령어:
  summary     - 최근 세션 요약 (기본값)
  steps       - 단계별 상세 정보 포함
  performance - 성능 분석
  search [키워드] - 키워드 검색

옵션:
  --hours N   - 조회할 시간 범위 (기본값: 24시간)
  --log-dir   - 로그 디렉토리 (기본값: logs)

예시:
  python check_logs.py summary
  python check_logs.py steps --hours 48
  python check_logs.py performance
  python check_logs.py search "교통량"
        """)
        return
    
    command = sys.argv[1]
    
    # 옵션 파싱
    hours = 24
    log_dir = "logs"
    
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--hours" and i + 1 < len(sys.argv):
            hours = int(sys.argv[i + 1])
        elif arg == "--log-dir" and i + 1 < len(sys.argv):
            log_dir = sys.argv[i + 1]
    
    viewer = ChatbotLogViewer(log_dir)
    
    if command == "summary":
        viewer.print_recent_summary(hours, show_steps=False)
    elif command == "steps":
        viewer.print_recent_summary(hours, show_steps=True)
    elif command == "performance":
        viewer.analyze_performance(hours)
    elif command == "search":
        if len(sys.argv) < 3:
            print("검색할 키워드를 입력해주세요.")
            print("예시: python check_logs.py search '교통량'")
            return
        keyword = sys.argv[2]
        viewer.search_sessions(keyword, hours)
    else:
        print(f"알 수 없는 명령어: {command}")
        print("사용 가능한 명령어: summary, steps, performance, search")

if __name__ == "__main__":
    main()
