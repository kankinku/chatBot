"""
PDF 전처리 및 데이터베이스 저장 모듈

PDF 파일들을 미리 처리하여 텍스트와 임베딩을 데이터베이스에 저장하여
실시간 질문-답변 속도를 크게 향상시킵니다.
"""

import os
import sqlite3
import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

from core.document.pdf_processor import PDFProcessor, TextChunk

logger = logging.getLogger(__name__)

class PDFDatabase:
    """PDF 내용을 저장하는 SQLite 데이터베이스"""
    
    def __init__(self, db_path: str = "./data/pdf_database.db"):
        """
        데이터베이스 초기화
        
        Args:
            db_path: 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self.ensure_db_directory()
        self.init_database()
    
    def ensure_db_directory(self):
        """데이터베이스 디렉토리 생성"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
    
    def init_database(self):
        """데이터베이스 테이블 생성"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS pdf_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL UNIQUE,
                    file_hash TEXT NOT NULL,
                    total_pages INTEGER,
                    total_chunks INTEGER,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS text_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pdf_file_id INTEGER,
                    chunk_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    page_number INTEGER,
                    embedding BLOB,
                    metadata TEXT,
                    FOREIGN KEY (pdf_file_id) REFERENCES pdf_files (id)
                )
            ''')
            
            # 인덱스 생성
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_pdf_files_hash 
                ON pdf_files(file_hash)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_chunks_pdf_file 
                ON text_chunks(pdf_file_id)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_chunks_page 
                ON text_chunks(page_number)
            ''')
            
        logger.info(f"PDF 데이터베이스 초기화 완료: {self.db_path}")
    
    def get_file_hash(self, filepath: str) -> str:
        """파일 해시 계산"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def is_file_processed(self, filepath: str) -> bool:
        """파일이 이미 처리되었는지 확인"""
        if not os.path.exists(filepath):
            return False
            
        file_hash = self.get_file_hash(filepath)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM pdf_files WHERE filepath = ? AND file_hash = ?",
                (filepath, file_hash)
            )
            return cursor.fetchone()[0] > 0
    
    def save_pdf_data(self, filepath: str, chunks: List[TextChunk], 
                      metadata: Dict) -> int:
        """PDF 데이터를 데이터베이스에 저장"""
        file_hash = self.get_file_hash(filepath)
        filename = os.path.basename(filepath)
        
        with sqlite3.connect(self.db_path) as conn:
            # PDF 파일 정보 저장
            cursor = conn.execute('''
                INSERT INTO pdf_files 
                (filename, filepath, file_hash, total_pages, total_chunks, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                filename, 
                filepath, 
                file_hash,
                metadata.get('pages', 0),
                len(chunks),
                json.dumps(metadata)
            ))
            
            pdf_file_id = cursor.lastrowid
            
            # 텍스트 청크들 저장
            for chunk in chunks:
                embedding_blob = pickle.dumps(chunk.embedding) if chunk.embedding is not None else None
                chunk_metadata = json.dumps(chunk.metadata) if chunk.metadata else None
                
                conn.execute('''
                    INSERT INTO text_chunks 
                    (pdf_file_id, chunk_id, content, page_number, embedding, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    pdf_file_id,
                    chunk.chunk_id,
                    chunk.content,
                    chunk.page_number,
                    embedding_blob,
                    chunk_metadata
                ))
            
            conn.commit()
            
        logger.info(f"PDF 데이터 저장 완료: {filename} ({len(chunks)}개 청크)")
        return pdf_file_id
    
    def load_all_chunks(self) -> List[TextChunk]:
        """모든 텍스트 청크 로드"""
        chunks = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT chunk_id, content, page_number, embedding, metadata
                FROM text_chunks
                ORDER BY pdf_file_id, page_number
            ''')
            
            for row in cursor.fetchall():
                chunk_id, content, page_number, embedding_blob, metadata_json = row
                
                # 임베딩 복원
                embedding = None
                if embedding_blob:
                    embedding = pickle.loads(embedding_blob)
                
                # 메타데이터 복원
                metadata = None
                if metadata_json:
                    metadata = json.loads(metadata_json)
                
                chunk = TextChunk(
                    content=content,
                    page_number=page_number,
                    chunk_id=chunk_id,
                    embedding=embedding,
                    metadata=metadata
                )
                chunks.append(chunk)
        
        logger.info(f"데이터베이스에서 {len(chunks)}개 청크 로드 완료")
        return chunks
    
    def get_pdf_files(self) -> List[Dict]:
        """처리된 PDF 파일 목록 조회"""
        files = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT filename, filepath, total_pages, total_chunks, processed_at
                FROM pdf_files
                ORDER BY processed_at DESC
            ''')
            
            for row in cursor.fetchall():
                files.append({
                    'filename': row[0],
                    'filepath': row[1],
                    'total_pages': row[2],
                    'total_chunks': row[3],
                    'processed_at': row[4]
                })
        
        return files
    
    def get_statistics(self) -> Dict:
        """전처리 통계 정보"""
        files = self.get_pdf_files()
        
        total_files = len(files)
        total_chunks = sum(f['total_chunks'] for f in files)
        total_pages = sum(f['total_pages'] for f in files)
        
        return {
            "total_files": total_files,
            "total_chunks": total_chunks,
            "total_pages": total_pages,
            "files": files
        }
    
    def clear_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM text_chunks")
            conn.execute("DELETE FROM pdf_files")
            conn.commit()
        
        logger.info("데이터베이스 초기화 완료")

class PDFPreprocessor:
    """PDF 전처리 시스템 (FastPDFPreprocessor의 별칭)"""
    
    def __init__(self, embedding_model: str = "jhgan/ko-sroberta-multitask"):
        """
        전처리기 초기화
        
        Args:
            embedding_model: 임베딩 모델명
        """
        self._processor = FastPDFPreprocessor(embedding_model)
    
    def preprocess_pdf(self, pdf_path: str, force_reprocess: bool = False) -> bool:
        """PDF 파일 전처리"""
        return self._processor.preprocess_pdf(pdf_path, force_reprocess)
    
    def preprocess_directory(self, directory_path: str, force_reprocess: bool = False) -> Dict[str, int]:
        """디렉토리 내 모든 PDF 파일 전처리"""
        return self._processor.preprocess_directory(directory_path, force_reprocess)
    
    def load_preprocessed_data(self) -> List[TextChunk]:
        """전처리된 데이터 로드"""
        return self._processor.load_preprocessed_data()


class FastPDFPreprocessor:
    """고속 PDF 전처리 시스템"""
    
    def __init__(self, embedding_model: str = "jhgan/ko-sroberta-multitask"):
        """
        전처리기 초기화
        
        Args:
            embedding_model: 임베딩 모델명
        """
        self.pdf_processor = PDFProcessor(embedding_model=embedding_model)
        self.database = PDFDatabase()
        
        logger.info("FastPDFPreprocessor 초기화 완료")
    
    def preprocess_pdf(self, pdf_path: str, force_reprocess: bool = False) -> bool:
        """
        PDF 파일 전처리
        
        Args:
            pdf_path: PDF 파일 경로
            force_reprocess: 강제 재처리 여부
            
        Returns:
            처리 성공 여부
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
            return False
        
        # 이미 처리된 파일인지 확인
        if not force_reprocess and self.database.is_file_processed(pdf_path):
            logger.info(f"이미 처리된 파일입니다: {os.path.basename(pdf_path)}")
            return True
        
        try:
            logger.info(f"PDF 전처리 시작: {os.path.basename(pdf_path)}")
            
            # PDF 처리
            chunks, metadata = self.pdf_processor.process_pdf(pdf_path)
            
            # 데이터베이스에 저장
            self.database.save_pdf_data(pdf_path, chunks, metadata)
            
            logger.info(f"PDF 전처리 완료: {os.path.basename(pdf_path)}")
            return True
            
        except Exception as e:
            logger.error(f"PDF 전처리 실패: {e}")
            return False
    
    def preprocess_directory(self, directory_path: str, 
                           force_reprocess: bool = False) -> Dict[str, int]:
        """
        디렉토리 내 모든 PDF 파일 전처리
        
        Args:
            directory_path: 디렉토리 경로
            force_reprocess: 강제 재처리 여부
            
        Returns:
            처리 결과 통계
        """
        stats = {"success": 0, "failed": 0, "skipped": 0}
        
        pdf_files = list(Path(directory_path).rglob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"PDF 파일을 찾을 수 없습니다: {directory_path}")
            return stats
        
        logger.info(f"{len(pdf_files)}개 PDF 파일 전처리 시작")
        
        for pdf_file in pdf_files:
            try:
                if self.preprocess_pdf(str(pdf_file), force_reprocess):
                    stats["success"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as e:
                logger.error(f"파일 처리 실패 {pdf_file}: {e}")
                stats["failed"] += 1
        
        logger.info(f"전처리 완료: 성공 {stats['success']}, "
                   f"실패 {stats['failed']}, 건너뜀 {stats['skipped']}")
        
        return stats
    
    def load_preprocessed_data(self) -> List[TextChunk]:
        """전처리된 데이터 로드"""
        return self.database.load_all_chunks()
    
    def get_statistics(self) -> Dict:
        """전처리 통계 정보"""
        files = self.database.get_pdf_files()
        
        total_files = len(files)
        total_chunks = sum(f['total_chunks'] for f in files)
        total_pages = sum(f['total_pages'] for f in files)
        
        return {
            "total_files": total_files,
            "total_chunks": total_chunks,
            "total_pages": total_pages,
            "files": files
        }

def preprocess_pdfs_command():
    """PDF 전처리 명령어 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF 파일 전처리")
    parser.add_argument("--path", required=True, help="PDF 파일 또는 디렉토리 경로")
    parser.add_argument("--force", action="store_true", help="강제 재처리")
    
    args = parser.parse_args()
    
    preprocessor = FastPDFPreprocessor()
    
    if os.path.isfile(args.path):
        # 단일 파일 처리
        success = preprocessor.preprocess_pdf(args.path, args.force)
        print(f"처리 완료: {'성공' if success else '실패'}")
    elif os.path.isdir(args.path):
        # 디렉토리 처리
        stats = preprocessor.preprocess_directory(args.path, args.force)
        print(f"처리 완료: 성공 {stats['success']}, 실패 {stats['failed']}, 건너뜀 {stats['skipped']}")
    else:
        print(f"경로를 찾을 수 없습니다: {args.path}")

if __name__ == "__main__":
    preprocess_pdfs_command()
