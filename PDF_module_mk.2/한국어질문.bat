@echo off
chcp 65001 > nul
echo.
echo ==========================================
echo 🇰🇷 한국어 PDF 질문답변 시스템
echo ==========================================
echo.

REM 가상환경 활성화
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo ⚠️  가상환경을 찾을 수 없습니다.
    echo    python -m venv venv 명령어로 가상환경을 생성해주세요.
    pause
    exit /b 1
)

REM Python 스크립트 실행
python korean_qa.py

pause
