#!/bin/bash

echo ""
echo "=========================================="
echo "🇰🇷 한국어 PDF 질문답변 시스템"
echo "=========================================="
echo ""

# 가상환경 활성화
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "⚠️  가상환경을 찾을 수 없습니다."
    echo "   python -m venv venv 명령어로 가상환경을 생성해주세요."
    exit 1
fi

# Python 스크립트 실행
python korean_qa.py

echo ""
echo "👋 프로그램을 종료합니다."
