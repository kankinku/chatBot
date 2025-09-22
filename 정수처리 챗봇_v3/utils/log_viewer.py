#!/usr/bin/env python3
"""
챗봇 로그 뷰어

단계별 처리 로그를 확인하고 분석할 수 있는 도구
"""

import os
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class LogEntry:
    """로그 엔트리"""
    timestamp: str
    session_id: str
    step: str
    step_time: float
    details: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SessionSummary:
    """세션 요약"""
    session_id: str
    start_time: str
    end_time: str
    total_time: float
    steps: List[LogEntry]
    question: Optional[str] = None
    pipeline_type: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None

class ChatbotLogViewer:
    """챗봇 로그 뷰어"""
    
    def __init__(self, log_dir: str = "logs"):
        """로그 뷰어 초기화"""
        self.log_dir = Path(log_dir)
        self.step_log_path = self.log_dir / "step_processing.log"
        self.detailed_log_path = self.log_dir / "chatbot_detailed.log"
        
    def parse_step_log_line(self, line: str) -> Optional[LogEntry]:
        """단계별 로그 라인 파싱"""
        try:
            # 형식: 2024-01-01 12:00:00 | [session_xxx] 단계명 : 시간초 | 상세정보
            match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| \[(session_\d{8}_\d{4})\] (.+?) : ([\d.]+)초( \| (.+))?', line)
            if match:
                timestamp, session_id, step, step_time, _, details = match.groups()
                return LogEntry(
                    timestamp=timestamp,
                    session_id=session_id,
                    step=step,
                    step_time=float(step_time),
                    details=details
                )
        except Exception as e:
            print(f"로그 라인 파싱 오류: {e}")
        return None
    
    def get_recent_sessions(self, hours: int = 24) -> List[SessionSummary]:
        """최근 세션들 조회"""
        if not self.step_log_path.exists():
            print(f"단계별 로그 파일이 없습니다: {self.step_log_path}")
            return []
        
        sessions = defaultdict(list)
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with open(self.step_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                log_entry = self.parse_step_log_line(line)
                if log_entry:
                    # 시간 필터링
                    log_time = datetime.strptime(log_entry.timestamp, '%Y-%m-%d %H:%M:%S')
                    if log_time >= cutoff_time:
                        sessions[log_entry.session_id].append(log_entry)
        
        # 세션 요약 생성
        session_summaries = []
        for session_id, steps in sessions.items():
            if not steps:
                continue
                
            # 시간순 정렬
            steps.sort(key=lambda x: x.timestamp)
            
            # 시작/종료 시간
            start_time = steps[0].timestamp
            end_time = steps[-1].timestamp
            
            # 총 시간 계산
            start_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
            end_dt = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
            total_time = (end_dt - start_dt).total_seconds()
            
            # 성공/실패 판단
            success = True
            error_message = None
            for step in steps:
                if step.step == "오류":
                    success = False
                    error_message = step.details
                    break
            
            # 질문 추출 (시작 단계에서)
            question = None
            for step in steps:
                if step.step == "시작" and step.details:
                    question_match = re.search(r'질문: (.+?)\.\.\.', step.details)
                    if question_match:
                        question = question_match.group(1)
                    break
            
            # 파이프라인 타입 추출
            pipeline_type = None
            for step in steps:
                if "파이프라인 시작" in step.step:
                    pipeline_type = step.step.replace("파이프라인 시작", "").strip()
                    break
            
            session_summary = SessionSummary(
                session_id=session_id,
                start_time=start_time,
                end_time=end_time,
                total_time=total_time,
                steps=steps,
                question=question,
                pipeline_type=pipeline_type,
                success=success,
                error_message=error_message
            )
            session_summaries.append(session_summary)
        
        # 최신 순으로 정렬
        session_summaries.sort(key=lambda x: x.start_time, reverse=True)
        return session_summaries
    
    def print_session_summary(self, session: SessionSummary, show_steps: bool = False):
        """세션 요약 출력"""
        print(f"\n{'='*80}")
        print(f"세션 ID: {session.session_id}")
        print(f"시작 시간: {session.start_time}")
        print(f"종료 시간: {session.end_time}")
        print(f"총 처리 시간: {session.total_time:.3f}초")
        print(f"파이프라인: {session.pipeline_type or '알 수 없음'}")
        print(f"상태: {'성공' if session.success else '실패'}")
        
        if session.question:
            print(f"질문: {session.question}")
        
        if session.error_message:
            print(f"오류: {session.error_message}")
        
        if show_steps:
            print(f"\n단계별 처리:")
            for i, step in enumerate(session.steps, 1):
                status_icon = "OK" if step.step != "오류" else "ERR"
                print(f"  {i:2d}. {status_icon} {step.step}: {step.step_time:.3f}초")
                if step.details:
                    print(f"      └─ {step.details}")
        
        print(f"{'='*80}")
    
    def print_recent_summary(self, hours: int = 24, show_steps: bool = False):
        """최근 세션들 요약 출력"""
        sessions = self.get_recent_sessions(hours)
        
        if not sessions:
            print(f"최근 {hours}시간 동안의 세션이 없습니다.")
            return
        
        print(f"\n최근 {hours}시간 세션 요약 ({len(sessions)}개)")
        print(f"{'='*80}")
        
        # 통계 계산
        total_sessions = len(sessions)
        successful_sessions = sum(1 for s in sessions if s.success)
        failed_sessions = total_sessions - successful_sessions
        
        pipeline_counts = defaultdict(int)
        total_times = []
        
        for session in sessions:
            pipeline_counts[session.pipeline_type or "알 수 없음"] += 1
            total_times.append(session.total_time)
        
        # 기본 통계 출력
        print(f"총 세션: {total_sessions}")
        print(f"성공: {successful_sessions} ({successful_sessions/total_sessions*100:.1f}%)")
        print(f"실패: {failed_sessions} ({failed_sessions/total_sessions*100:.1f}%)")
        
        if total_times:
            avg_time = sum(total_times) / len(total_times)
            min_time = min(total_times)
            max_time = max(total_times)
            print(f"평균 처리 시간: {avg_time:.3f}초 (최소: {min_time:.3f}초, 최대: {max_time:.3f}초)")
        
        print(f"\n파이프라인별 분포:")
        for pipeline, count in pipeline_counts.items():
            print(f"  {pipeline}: {count}개 ({count/total_sessions*100:.1f}%)")
        
        # 최근 세션들 상세 출력
        print(f"\n📋 최근 세션들:")
        for session in sessions[:10]:  # 최근 10개만
            self.print_session_summary(session, show_steps)
    
    def analyze_performance(self, hours: int = 24):
        """성능 분석"""
        sessions = self.get_recent_sessions(hours)
        
        if not sessions:
            print(f"최근 {hours}시간 동안의 세션이 없습니다.")
            return
        
        print(f"\n성능 분석 (최근 {hours}시간)")
        print(f"{'='*80}")
        
        # 단계별 평균 시간 계산
        step_times = defaultdict(list)
        for session in sessions:
            for step in session.steps:
                step_times[step.step].append(step.step_time)
        
        print(f"단계별 평균 처리 시간:")
        for step, times in sorted(step_times.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True):
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            print(f"  {step}: {avg_time:.3f}초 (최소: {min_time:.3f}초, 최대: {max_time:.3f}초)")
        
        # 파이프라인별 성능
        pipeline_times = defaultdict(list)
        for session in sessions:
            if session.pipeline_type:
                pipeline_times[session.pipeline_type].append(session.total_time)
        
        print(f"\n파이프라인별 평균 처리 시간:")
        for pipeline, times in pipeline_times.items():
            avg_time = sum(times) / len(times)
            print(f"  {pipeline}: {avg_time:.3f}초 ({len(times)}개 세션)")
    
    def search_sessions(self, keyword: str, hours: int = 24):
        """키워드로 세션 검색"""
        sessions = self.get_recent_sessions(hours)
        
        matching_sessions = []
        for session in sessions:
            # 질문에서 검색
            if session.question and keyword.lower() in session.question.lower():
                matching_sessions.append(session)
                continue
            
            # 단계별 상세정보에서 검색
            for step in session.steps:
                if step.details and keyword.lower() in step.details.lower():
                    matching_sessions.append(session)
                    break
        
        if not matching_sessions:
            print(f"'{keyword}'와 관련된 세션을 찾을 수 없습니다.")
            return
        
        print(f"\n'{keyword}' 검색 결과 ({len(matching_sessions)}개)")
        print(f"{'='*80}")
        
        for session in matching_sessions:
            self.print_session_summary(session, show_steps=True)

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="챗봇 로그 뷰어")
    parser.add_argument("--hours", type=int, default=24, help="조회할 시간 범위 (시간)")
    parser.add_argument("--show-steps", action="store_true", help="단계별 상세 정보 표시")
    parser.add_argument("--performance", action="store_true", help="성능 분석")
    parser.add_argument("--search", type=str, help="키워드 검색")
    parser.add_argument("--log-dir", type=str, default="logs", help="로그 디렉토리")
    
    args = parser.parse_args()
    
    viewer = ChatbotLogViewer(args.log_dir)
    
    if args.search:
        viewer.search_sessions(args.search, args.hours)
    elif args.performance:
        viewer.analyze_performance(args.hours)
    else:
        viewer.print_recent_summary(args.hours, args.show_steps)

if __name__ == "__main__":
    main()
