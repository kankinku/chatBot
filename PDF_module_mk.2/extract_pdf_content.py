#!/usr/bin/env python3
"""
PDF 파일 내용 추출 스크립트
data 폴더의 PDF 파일들의 내용을 확인하여 테스트 질문을 만들기 위한 도구
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from core.pdf_processor import PDFProcessor

def extract_pdf_content(pdf_path: str) -> str:
    """PDF 파일에서 텍스트 추출"""
    try:
        processor = PDFProcessor()
        full_text, metadata = processor.extract_text_from_pdf(pdf_path)
        return full_text, metadata
    except Exception as e:
        print(f"PDF 추출 실패 {pdf_path}: {e}")
        return "", {}

def main():
    """메인 함수"""
    data_folder = "./data"
    
    if not os.path.exists(data_folder):
        print("data 폴더가 존재하지 않습니다.")
        return
    
    pdf_files = []
    for file in os.listdir(data_folder):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(data_folder, file))
    
    if not pdf_files:
        print("data 폴더에 PDF 파일이 없습니다.")
        return
    
    print(f"발견된 PDF 파일: {len(pdf_files)}개")
    print("=" * 80)
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"\n📄 파일: {filename}")
        print("-" * 60)
        
        # PDF 내용 추출
        text, metadata = extract_pdf_content(pdf_path)
        
        if text:
            print(f"페이지 수: {metadata.get('pages', 'N/A')}")
            print(f"추출 방법: {', '.join(metadata.get('extraction_method', []))}")
            print(f"텍스트 길이: {len(text)} 문자")
            
            # 처음 1000자만 출력
            preview = text[:1000].replace('\n', ' ').strip()
            print(f"\n📝 내용 미리보기:")
            print(preview)
            
            if len(text) > 1000:
                print("... (더 많은 내용이 있습니다)")
        else:
            print("❌ 텍스트 추출 실패")
        
        print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
