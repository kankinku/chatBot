#!/usr/bin/env python3
"""
간단한 형태의 QA 결과 리포트 생성 스크립트
번호, 질문, 정답, 답변 형태로 저장
"""

import argparse
import json
import sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Generate simple QA report")
    ap.add_argument("--input", required=True, help="Input JSON report file")
    ap.add_argument("--output", required=True, help="Output simple JSON file")
    args = ap.parse_args()

    # 입력 파일 읽기
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 간단한 형태로 변환
    simple_results = []
    for row in data["rows"]:
        simple_results.append({
            "번호": row["id"],
            "질문": row["question"],
            "정답": row["gold"],
            "답변": row["pred"]
        })

    # 출력 파일 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(simple_results, f, ensure_ascii=False, indent=2)

    print(f"간단한 결과 파일이 생성되었습니다: {output_path}")
    print(f"총 {len(simple_results)}개 질문 결과")

if __name__ == "__main__":
    main()
