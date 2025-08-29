"""
대화 이력 로그 관리 모듈

이 모듈은 대화 이력을 로그로 저장하고, 이전 답변을 빠르게 검색하여
LLM 사용 없이 답변할 수 있는 기능을 제공합니다.
"""

import json
import sqlite3
import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import re

logger = logging.getLogger(__name__)

@dataclass
class ConversationLog:
    """대화 로그 데이터 클래스"""
    id: str
    question: str
    answer: str
    question_hash: str
    question_type: str
    confidence_score: float
    timestamp: datetime
    pdf_id: Optional[str] = None
    used_chunks: Optional[List[str]] = None
    metadata: Optional[Dict] = None

class ConversationLogger:
    """
    대화 이력 로그 관리 클래스
    
    주요 기능:
    1. 대화 이력을 SQLite DB에 저장
    2. 질문 해시 기반 빠른 검색
    3. 유사 질문 검색
    4. 답변 품질 평가
    5. 로그 통계 및 관리
    """
    
    def __init__(self, db_path: str = "data/conversation_history.db"):
        """
        ConversationLogger 초기화
        
        Args:
            db_path: SQLite 데이터베이스 파일 경로
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 데이터베이스 초기화
        self._init_database()
        
        logger.info(f"ConversationLogger 초기화 완료: {db_path}")
    
    def _init_database(self):
        """데이터베이스 테이블 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 대화 로그 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_logs (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    question_hash TEXT NOT NULL,
                    question_type TEXT NOT NULL,
                    confidence_score REAL DEFAULT 0.0,
                    timestamp TEXT NOT NULL,
                    pdf_id TEXT,
                    used_chunks TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 질문 해시 인덱스 생성 (빠른 검색용)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_question_hash 
                ON conversation_logs(question_hash)
            """)
            
            # 질문 유형 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_question_type 
                ON conversation_logs(question_type)
            """)
            
            # 타임스탬프 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON conversation_logs(timestamp)
            """)
            
            conn.commit()
    
    def _generate_question_hash(self, question: str) -> str:
        """
        질문의 해시값 생성 (정규화된 형태)
        
        Args:
            question: 원본 질문
            
        Returns:
            질문 해시값
        """
        # 질문 정규화 (공백 제거, 소문자 변환, 특수문자 제거)
        normalized = re.sub(r'[^\w\s가-힣]', '', question.lower().strip())
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # 해시 생성
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def _calculate_similarity(self, question1: str, question2: str) -> float:
        """
        두 질문 간의 유사도 계산 (간단한 Jaccard 유사도)
        
        Args:
            question1: 첫 번째 질문
            question2: 두 번째 질문
            
        Returns:
            유사도 점수 (0-1)
        """
        # 단어 집합으로 분리
        words1 = set(re.findall(r'\w+', question1.lower()))
        words2 = set(re.findall(r'\w+', question2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard 유사도 계산
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def add_conversation(self, 
                        question: str, 
                        answer: str, 
                        question_type: str = "general",
                        confidence_score: float = 0.0,
                        pdf_id: Optional[str] = None,
                        used_chunks: Optional[List[str]] = None,
                        metadata: Optional[Dict] = None) -> str:
        """
        대화 로그 추가
        
        Args:
            question: 사용자 질문
            answer: 시스템 답변
            question_type: 질문 유형
            confidence_score: 답변 신뢰도
            pdf_id: PDF 문서 ID
            used_chunks: 사용된 청크 ID들
            metadata: 추가 메타데이터
            
        Returns:
            생성된 로그 ID
        """
        log_id = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(question) % 10000}"
        question_hash = self._generate_question_hash(question)
        timestamp = datetime.now().isoformat()
        
        log_entry = ConversationLog(
            id=log_id,
            question=question,
            answer=answer,
            question_hash=question_hash,
            question_type=question_type,
            confidence_score=confidence_score,
            timestamp=timestamp,
            pdf_id=pdf_id,
            used_chunks=used_chunks or [],
            metadata=metadata or {}
        )
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversation_logs 
                (id, question, answer, question_hash, question_type, confidence_score, 
                 timestamp, pdf_id, used_chunks, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                log_entry.id,
                log_entry.question,
                log_entry.answer,
                log_entry.question_hash,
                log_entry.question_type,
                log_entry.confidence_score,
                log_entry.timestamp,
                log_entry.pdf_id,
                json.dumps(log_entry.used_chunks),
                json.dumps(log_entry.metadata)
            ))
            conn.commit()
        
        logger.info(f"대화 로그 추가: {log_id}")
        return log_id
    
    def find_exact_match(self, question: str) -> Optional[ConversationLog]:
        """
        정확히 일치하는 질문 검색
        
        Args:
            question: 검색할 질문
            
        Returns:
            일치하는 대화 로그 또는 None
        """
        question_hash = self._generate_question_hash(question)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, question, answer, question_hash, question_type, 
                       confidence_score, timestamp, pdf_id, used_chunks, metadata
                FROM conversation_logs 
                WHERE question_hash = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (question_hash,))
            
            row = cursor.fetchone()
            if row:
                return ConversationLog(
                    id=row[0],
                    question=row[1],
                    answer=row[2],
                    question_hash=row[3],
                    question_type=row[4],
                    confidence_score=row[5],
                    timestamp=row[6],
                    pdf_id=row[7],
                    used_chunks=json.loads(row[8]) if row[8] else [],
                    metadata=json.loads(row[9]) if row[9] else {}
                )
        
        return None
    
    def find_similar_questions(self, question: str, threshold: float = 0.7, limit: int = 5) -> List[Tuple[ConversationLog, float]]:
        """
        유사한 질문 검색
        
        Args:
            question: 검색할 질문
            threshold: 유사도 임계값
            limit: 반환할 최대 결과 수
            
        Returns:
            (대화 로그, 유사도 점수) 튜플 리스트
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, question, answer, question_hash, question_type, 
                       confidence_score, timestamp, pdf_id, used_chunks, metadata
                FROM conversation_logs 
                ORDER BY timestamp DESC
                LIMIT 100
            """)
            
            similar_questions = []
            for row in cursor.fetchall():
                stored_question = row[1]
                similarity = self._calculate_similarity(question, stored_question)
                
                if similarity >= threshold:
                    log_entry = ConversationLog(
                        id=row[0],
                        question=row[1],
                        answer=row[2],
                        question_hash=row[3],
                        question_type=row[4],
                        confidence_score=row[5],
                        timestamp=row[6],
                        pdf_id=row[7],
                        used_chunks=json.loads(row[8]) if row[8] else [],
                        metadata=json.loads(row[9]) if row[9] else {}
                    )
                    similar_questions.append((log_entry, similarity))
            
            # 유사도 순으로 정렬하고 제한
            similar_questions.sort(key=lambda x: x[1], reverse=True)
            return similar_questions[:limit]
    
    def get_conversation_history(self, pdf_id: Optional[str] = None, limit: int = 20) -> List[ConversationLog]:
        """
        대화 이력 조회
        
        Args:
            pdf_id: 특정 PDF의 대화 이력만 조회 (None이면 전체)
            limit: 반환할 최대 결과 수
            
        Returns:
            대화 로그 리스트
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if pdf_id:
                cursor.execute("""
                    SELECT id, question, answer, question_hash, question_type, 
                           confidence_score, timestamp, pdf_id, used_chunks, metadata
                    FROM conversation_logs 
                    WHERE pdf_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (pdf_id, limit))
            else:
                cursor.execute("""
                    SELECT id, question, answer, question_hash, question_type, 
                           confidence_score, timestamp, pdf_id, used_chunks, metadata
                    FROM conversation_logs 
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
            
            logs = []
            for row in cursor.fetchall():
                log_entry = ConversationLog(
                    id=row[0],
                    question=row[1],
                    answer=row[2],
                    question_hash=row[3],
                    question_type=row[4],
                    confidence_score=row[5],
                    timestamp=row[6],
                    pdf_id=row[7],
                    used_chunks=json.loads(row[8]) if row[8] else [],
                    metadata=json.loads(row[9]) if row[9] else {}
                )
                logs.append(log_entry)
            
            return logs
    
    def clear_history(self, pdf_id: Optional[str] = None) -> int:
        """
        대화 이력 삭제
        
        Args:
            pdf_id: 특정 PDF의 대화 이력만 삭제 (None이면 전체)
            
        Returns:
            삭제된 로그 수
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if pdf_id:
                cursor.execute("DELETE FROM conversation_logs WHERE pdf_id = ?", (pdf_id,))
            else:
                cursor.execute("DELETE FROM conversation_logs")
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            logger.info(f"대화 이력 삭제: {deleted_count}개 (PDF ID: {pdf_id})")
            return deleted_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        대화 로그 통계 조회
        
        Returns:
            통계 정보 딕셔너리
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 전체 로그 수
            cursor.execute("SELECT COUNT(*) FROM conversation_logs")
            total_logs = cursor.fetchone()[0]
            
            # 오늘 로그 수
            cursor.execute("""
                SELECT COUNT(*) FROM conversation_logs 
                WHERE DATE(timestamp) = DATE('now')
            """)
            today_logs = cursor.fetchone()[0]
            
            # 질문 유형별 통계
            cursor.execute("""
                SELECT question_type, COUNT(*) 
                FROM conversation_logs 
                GROUP BY question_type
            """)
            type_stats = dict(cursor.fetchall())
            
            # 평균 신뢰도
            cursor.execute("SELECT AVG(confidence_score) FROM conversation_logs")
            avg_confidence = cursor.fetchone()[0] or 0.0
            
            return {
                "total_logs": total_logs,
                "today_logs": today_logs,
                "type_statistics": type_stats,
                "average_confidence": round(avg_confidence, 3)
            }

