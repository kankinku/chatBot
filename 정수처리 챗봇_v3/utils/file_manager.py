"""
파일 관리 유틸리티

PDF 파일 저장, 조회, 관리를 위한 헬퍼 함수들을 제공합니다.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
import json
from datetime import datetime

class PDFFileManager:
    """PDF 파일 관리 클래스"""
    
    def __init__(self, base_data_dir: str = "./data"):
        """
        파일 매니저 초기화
        
        Args:
            base_data_dir: 기본 데이터 디렉토리
        """
        self.base_dir = Path(base_data_dir)
        
        # 디렉토리 구조 생성
        self.pdf_dir = self.base_dir / "pdfs"
        self.vector_store_dir = self.base_dir / "vector_store"
        self.conversation_dir = self.base_dir / "conversation_history"
        self.temp_dir = self.base_dir / "temp"
        
        # 디렉토리 생성
        for directory in [self.pdf_dir, self.vector_store_dir, 
                         self.conversation_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_pdf(self, source_path: str, category: Optional[str] = None, 
                 custom_name: Optional[str] = None) -> Dict[str, str]:
        """
        PDF 파일을 관리 폴더에 저장
        
        Args:
            source_path: 원본 PDF 파일 경로
            category: 카테고리 폴더명 (선택사항)
            custom_name: 사용자 지정 파일명 (선택사항)
            
        Returns:
            저장된 파일 정보
        """
        source_path = Path(source_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {source_path}")
        
        if not source_path.suffix.lower() == '.pdf':
            raise ValueError("PDF 파일만 저장할 수 있습니다.")
        
        # 저장 경로 결정
        if category:
            target_dir = self.pdf_dir / category
            target_dir.mkdir(exist_ok=True)
        else:
            target_dir = self.pdf_dir
        
        # 파일명 결정
        if custom_name:
            if not custom_name.endswith('.pdf'):
                custom_name += '.pdf'
            target_filename = custom_name
        else:
            target_filename = source_path.name
        
        target_path = target_dir / target_filename
        
        # 중복 파일명 처리
        counter = 1
        original_target = target_path
        while target_path.exists():
            name_without_ext = original_target.stem
            target_path = target_dir / f"{name_without_ext}_{counter}.pdf"
            counter += 1
        
        # 파일 복사
        shutil.copy2(source_path, target_path)
        
        # 메타데이터 생성
        file_info = {
            "original_path": str(source_path),
            "saved_path": str(target_path),
            "category": category or "uncategorized",
            "filename": target_path.name,
            "size_bytes": target_path.stat().st_size,
            "created_at": datetime.now().isoformat(),
            "file_hash": self._calculate_file_hash(target_path)
        }
        
        return file_info
    
    def list_pdfs(self, category: Optional[str] = None) -> List[Dict[str, str]]:
        """
        저장된 PDF 파일 목록 조회
        
        Args:
            category: 특정 카테고리만 조회 (선택사항)
            
        Returns:
            PDF 파일 정보 리스트
        """
        pdf_files = []
        
        if category:
            search_dir = self.pdf_dir / category
            if not search_dir.exists():
                return []
        else:
            search_dir = self.pdf_dir
        
        # PDF 파일 검색 (재귀적)
        for pdf_path in search_dir.rglob("*.pdf"):
            relative_path = pdf_path.relative_to(self.pdf_dir)
            category_name = relative_path.parent.name if relative_path.parent != Path('.') else "uncategorized"
            
            file_info = {
                "filename": pdf_path.name,
                "path": str(pdf_path),
                "relative_path": str(relative_path),
                "category": category_name,
                "size_bytes": pdf_path.stat().st_size,
                "size_mb": round(pdf_path.stat().st_size / 1024 / 1024, 2),
                "modified_at": datetime.fromtimestamp(pdf_path.stat().st_mtime).isoformat()
            }
            
            pdf_files.append(file_info)
        
        return sorted(pdf_files, key=lambda x: x['modified_at'], reverse=True)
    
    def get_pdf_path(self, filename: str, category: Optional[str] = None) -> Optional[str]:
        """
        PDF 파일 경로 조회
        
        Args:
            filename: 파일명
            category: 카테고리 (선택사항)
            
        Returns:
            파일 경로 또는 None
        """
        if category:
            pdf_path = self.pdf_dir / category / filename
        else:
            # 전체 폴더에서 검색
            for pdf_path in self.pdf_dir.rglob(filename):
                return str(pdf_path)
            return None
        
        return str(pdf_path) if pdf_path.exists() else None
    
    def delete_pdf(self, filename: str, category: Optional[str] = None) -> bool:
        """
        PDF 파일 삭제
        
        Args:
            filename: 파일명
            category: 카테고리 (선택사항)
            
        Returns:
            삭제 성공 여부
        """
        pdf_path = self.get_pdf_path(filename, category)
        
        if pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)
            return True
        
        return False
    
    def create_category(self, category_name: str) -> str:
        """
        새 카테고리 폴더 생성
        
        Args:
            category_name: 카테고리 이름
            
        Returns:
            생성된 폴더 경로
        """
        category_dir = self.pdf_dir / category_name
        category_dir.mkdir(exist_ok=True)
        return str(category_dir)
    
    def get_categories(self) -> List[str]:
        """
        사용 가능한 카테고리 목록 조회
        
        Returns:
            카테고리 이름 리스트
        """
        categories = []
        
        for item in self.pdf_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                categories.append(item.name)
        
        return sorted(categories)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """파일 해시 계산"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_storage_info(self) -> Dict[str, any]:
        """저장소 정보 조회"""
        total_files = len(list(self.pdf_dir.rglob("*.pdf")))
        total_size = sum(f.stat().st_size for f in self.pdf_dir.rglob("*.pdf"))
        
        return {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "categories": self.get_categories(),
            "pdf_directory": str(self.pdf_dir)
        }

# 편의 함수들
def setup_pdf_storage() -> PDFFileManager:
    """PDF 저장소 설정 및 초기화"""
    manager = PDFFileManager()
    
    # 기본 카테고리 생성
    default_categories = ["academic", "manuals", "reports", "misc"]
    for category in default_categories:
        manager.create_category(category)
    
    return manager

def quick_save_pdf(pdf_path: str, category: str = "misc") -> str:
    """PDF 파일을 빠르게 저장"""
    manager = setup_pdf_storage()
    result = manager.save_pdf(pdf_path, category)
    return result["saved_path"]

if __name__ == "__main__":
    # 테스트 코드
    manager = setup_pdf_storage()
    
    print("PDF 파일 매니저 테스트")
    print(f"저장소 정보: {manager.get_storage_info()}")
    print(f"사용 가능한 카테고리: {manager.get_categories()}")
    
    # 저장된 PDF 목록
    pdfs = manager.list_pdfs()
    print(f"저장된 PDF 파일: {len(pdfs)}개")
    for pdf in pdfs[:5]:  # 최대 5개만 표시
        print(f"  - {pdf['filename']} ({pdf['category']}) - {pdf['size_mb']}MB")
