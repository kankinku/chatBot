#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 corpus를 v6 형식으로 변환하는 스크립트
"""

import json
import sys

def convert_corpus(input_path: str, output_path: str):
    """
    v2 corpus를 v6 형식으로 변환
    
    v2 format: {doc_id, filename, page, start, length, text, extra}
    v6 format: {doc_id, filename, page, start_offset, length, text}
    """
    with open(input_path, "r", encoding="utf-8") as f_in:
        with open(output_path, "w", encoding="utf-8") as f_out:
            count = 0
            for line in f_in:
                try:
                    data = json.loads(line.strip())
                    
                    # v6 형식으로 변환
                    v6_chunk = {
                        "doc_id": data.get("doc_id", "unknown"),
                        "filename": data.get("filename", "unknown.pdf"),
                        "page": data.get("page") or 0,  # None일 경우 0
                        "start_offset": data.get("start", 0),  # "start" → "start_offset"
                        "length": data.get("length", len(data.get("text", ""))),
                        "text": data.get("text", ""),
                    }
                    
                    f_out.write(json.dumps(v6_chunk, ensure_ascii=False) + "\n")
                    count += 1
                
                except Exception as e:
                    print(f"Error converting line: {e}", file=sys.stderr)
                    continue
            
            print(f"Converted {count} chunks")

if __name__ == "__main__":
    input_file = "../../정수처리 챗봇_v2/data/corpus_v1.jsonl"
    output_file = "../data/corpus.jsonl"
    
    print(f"Converting {input_file} → {output_file}")
    convert_corpus(input_file, output_file)
    print("Done!")

