#!/usr/bin/env python3
"""
JSONL 형식의 코퍼스를 txt 파일로 변환하는 스크립트
"""

import json
import os
from pathlib import Path

def convert_corpus_to_txt(jsonl_path: str, output_path: str = None):
    """
    JSONL 형식의 코퍼스를 txt 파일로 변환
    
    Args:
        jsonl_path: 입력 JSONL 파일 경로
        output_path: 출력 txt 파일 경로 (기본값: corpus.txt)
    """
    if output_path is None:
        output_path = "corpus.txt"
    
    # 출력 디렉토리 생성
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    texts = []
    
    print(f"JSONL 파일 읽는 중: {jsonl_path}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                if 'text' in data:
                    texts.append(data['text'])
                else:
                    print(f"경고: {line_num}번째 줄에 'text' 필드가 없습니다.")
            except json.JSONDecodeError as e:
                print(f"오류: {line_num}번째 줄 JSON 파싱 실패: {e}")
                continue
    
    print(f"총 {len(texts)}개의 텍스트 청크를 찾았습니다.")
    
    # txt 파일로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, text in enumerate(texts):
            f.write(f"=== 청크 {i+1} ===\n")
            f.write(text)
            f.write("\n\n")
    
    print(f"txt 파일 생성 완료: {output_path}")
    print(f"파일 크기: {os.path.getsize(output_path)} bytes")

def main():
    """메인 함수"""
    # 기본 경로 설정
    jsonl_path = "data/corpus_v1.jsonl"
    output_path = "data/corpus.txt"
    
    # 파일 존재 확인
    if not os.path.exists(jsonl_path):
        print(f"오류: {jsonl_path} 파일을 찾을 수 없습니다.")
        return
    
    # 변환 실행
    convert_corpus_to_txt(jsonl_path, output_path)

if __name__ == "__main__":
    main()
