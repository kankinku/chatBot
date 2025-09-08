#!/bin/bash
set -e

echo "범용 RAG 시스템 서버 시작 중..."

# 환경 변수 설정
export PYTHONPATH=/app
export MODEL_TYPE=${MODEL_TYPE:-local}
export MODEL_NAME=${MODEL_NAME:-beomi/KoAlpaca-Polyglot-1.3B}
export EMBEDDING_MODEL=${EMBEDDING_MODEL:-jhgan/ko-sroberta-multitask}
export RESET_CHUNKS_ON_START=${RESET_CHUNKS_ON_START:-false}
export AUTO_UPLOAD_PDFS=${AUTO_UPLOAD_PDFS:-true}

# 메모리 최적화 환경 변수 추가
export MAX_MODEL_MEMORY_GB=${MAX_MODEL_MEMORY_GB:-8.0}
export MEMORY_WARNING_THRESHOLD=${MEMORY_WARNING_THRESHOLD:-75.0}
export MEMORY_CRITICAL_THRESHOLD=${MEMORY_CRITICAL_THRESHOLD:-85.0}
export PDF_MAX_MEMORY_GB=${PDF_MAX_MEMORY_GB:-2.0}

# 로그 디렉토리 생성
mkdir -p /app/logs

echo "환경 설정:"
echo "  - MODEL_TYPE: $MODEL_TYPE"
echo "  - MODEL_NAME: $MODEL_NAME"
echo "  - EMBEDDING_MODEL: $EMBEDDING_MODEL"
echo "  - PYTHONPATH: $PYTHONPATH"
echo "  - RESET_CHUNKS_ON_START: ${RESET_CHUNKS_ON_START:-false}"
echo "  - AUTO_UPLOAD_PDFS: ${AUTO_UPLOAD_PDFS:-false}"
echo "  - MAX_MODEL_MEMORY_GB: $MAX_MODEL_MEMORY_GB"
echo "  - MEMORY_WARNING_THRESHOLD: $MEMORY_WARNING_THRESHOLD"
echo "  - MEMORY_CRITICAL_THRESHOLD: $MEMORY_CRITICAL_THRESHOLD"
echo "  - PDF_MAX_MEMORY_GB: $PDF_MAX_MEMORY_GB"

# 1단계: 의존성 확인
echo "1단계: 의존성 확인 중..."
python -c "
import sys
required_packages = ['sentence_transformers', 'torch', 'transformers', 'numpy', 'sklearn', 'psutil']
missing_packages = []

for package in required_packages:
    try:
        __import__(package.replace('-', '_'))
        print(f'[OK] {package}')
    except ImportError:
        print(f'[FAIL] {package} (설치 필요)')
        missing_packages.append(package)

if missing_packages:
    print(f'[ERROR] 누락된 패키지: {missing_packages}')
    sys.exit(1)
else:
    print('[SUCCESS] 모든 의존성이 설치되어 있습니다.')
"

if [ $? -ne 0 ]; then
    echo "[ERROR] 의존성 확인 실패. 컨테이너를 종료합니다."
    exit 1
fi

# 2단계: 모델 자동 다운로드 (메모리 최적화 적용)
echo "2단계: AI 모델 자동 다운로드 중..."
python - <<'PY'
import os
import sys
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model(model_id, model_name, cache_dir="./models"):
    """모델 자동 다운로드 함수 (메모리 최적화 적용)"""
    try:
        print(f"[DOWNLOAD] {model_name} 다운로드 중: {model_id}")
        
        if "tokenizer" in model_name.lower():
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            print(f"[SUCCESS] {model_name} 토크나이저 다운로드 완료")
        elif "sbert" in model_name.lower() or "sentence" in model_name.lower():
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_id, cache_folder=cache_dir)
            print(f"[SUCCESS] {model_name} SBERT 모델 다운로드 완료")
        elif "sqlcoder" in model_id.lower() or "sqlcoder" in model_name.lower():
            # SQLCoder는 Llama 기반 모델이므로 AutoModelForCausalLM 사용
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype="auto"
            )
            print(f"[SUCCESS] {model_name} SQLCoder 모델 다운로드 완료")
        else:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print(f"[SUCCESS] {model_name} 모델 다운로드 완료")
        
        return True
    except Exception as e:
        print(f"[ERROR] {model_name} 다운로드 실패: {e}")
        return False

# 다운로드할 모델 목록
models_to_download = [
    {
        "id": "beomi/KoAlpaca-Polyglot-1.3B",
        "name": "KoAlpaca-Polyglot-1.3B (답변 생성)",
        "type": "causal"
    },
    {
        "id": "jhgan/ko-sroberta-multitask", 
        "name": "Ko-SRoBERTa (임베딩/SBERT)",
        "type": "sbert"
    },
    {
        "id": "defog/sqlcoder-7b-2",
        "name": "SQLCoder (SQL 생성)",
        "type": "sqlcoder"
    }
]

print("AI 모델 자동 다운로드를 시작합니다...")
print("=" * 60)

success_count = 0
total_count = len(models_to_download)

for i, model_info in enumerate(models_to_download, 1):
    print(f"\n{i}/{total_count}. {model_info['name']}")
    if download_model(model_info['id'], model_info['name']):
        success_count += 1
    else:
        print(f"[WARNING] {model_info['name']} 다운로드 실패, 계속 진행합니다...")

print(f"\n" + "=" * 60)
print(f"[RESULT] 모델 다운로드 결과: {success_count}/{total_count} 성공")
print("=" * 60)

if success_count == 0:
    print("[ERROR] 모든 모델 다운로드에 실패했습니다.")
    sys.exit(1)
elif success_count < total_count:
    print("[WARNING] 일부 모델 다운로드에 실패했지만 계속 진행합니다.")
else:
    print("[SUCCESS] 모든 모델 다운로드 완료!")
PY

if [ $? -ne 0 ]; then
    echo "[ERROR] 모델 다운로드 실패. 컨테이너를 종료합니다."
    exit 1
fi

# 3단계: 핵심 모듈 초기화 확인 (메모리 최적화 포함)
echo "3단계: 핵심 모듈 초기화 확인 중..."
python -c "
import sys
sys.path.append('/app')

try:
    from core.query.query_router import QueryRouter
    from core.database.sql_element_extractor import SQLElementExtractor
    from core.llm.answer_generator import AnswerGenerator
    from core.utils.memory_optimizer import memory_optimizer, model_memory_manager
    print('[SUCCESS] 핵심 모듈 초기화 완료')
    print('[INFO] 메모리 최적화 시스템 준비 완료')
except Exception as e:
    print(f'[WARNING] 핵심 모듈 초기화 실패: {e}')
    print('계속 진행합니다...')
"

if [ $? -ne 0 ]; then
    echo "[WARNING] 핵심 모듈 초기화 실패, 계속 진행합니다..."
fi

# 4단계: 청크 초기화 (필요시)
echo "4단계: 청크 초기화 확인..."
if [ "$RESET_CHUNKS_ON_START" = "true" ]; then
    echo "청크 초기화 모드가 활성화되었습니다."
    echo "기존 청크를 초기화하고 새로운 모델로 재생성합니다..."
    
    python -c "
import sys
sys.path.append('/app')

try:
    from core.document.vector_store import HybridVectorStore
    from core.document.pdf_processor import PDFProcessor
    from pathlib import Path
    
    print('청크 초기화 시작...')
    
    # 벡터 저장소 초기화
    vector_store = HybridVectorStore()
    pdf_processor = PDFProcessor()
    
    # 기존 청크 수 확인
    total_chunks = vector_store.get_total_chunks()
    print(f'기존 청크 수: {total_chunks}')
    
    # 벡터 저장소 초기화
    vector_store.clear()
    print('벡터 저장소 초기화 완료')
    
    # PDF 파일 찾기
    pdf_files = []
    pdf_dir = Path('/app/data/pdfs')
    if pdf_dir.exists():
        pdf_files.extend([str(f) for f in pdf_dir.glob('*.pdf')])
    
    data_dir = Path('/app/data')
    if data_dir.exists():
        pdf_files.extend([str(f) for f in data_dir.glob('*.pdf')])
    
    print(f'발견된 PDF 파일: {len(pdf_files)}개')
    
    if pdf_files:
        total_new_chunks = 0
        for pdf_file in pdf_files:
            try:
                print(f'처리 중: {pdf_file}')
                chunks = pdf_processor.process_pdf(pdf_file)
                if chunks:
                    vector_store.add_chunks(chunks)
                    total_new_chunks += len(chunks)
                    print(f'  → {len(chunks)}개 청크 생성 완료')
                else:
                    print(f'  → 청크 생성 실패')
            except Exception as e:
                print(f'  → 오류: {e}')
                continue
        
        print(f'청크 재생성 완료! 총 {total_new_chunks}개 청크 생성')
    else:
        print('PDF 파일이 없습니다.')
    
    print('[SUCCESS] 청크 초기화 완료')
    
except Exception as e:
    print(f'[ERROR] 청크 초기화 실패: {e}')
    print('계속 진행합니다...')
"
else
    echo "청크 초기화 모드가 비활성화되어 있습니다. 기존 청크를 유지합니다."
fi

# 5단계: 모델 로딩 테스트 (메모리 최적화 적용)
echo "5단계: 모델 로딩 테스트 중..."
python - <<'PY'
import sys
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# 메모리 최적화 설정 적용
max_model_memory_gb = float(os.getenv('MAX_MODEL_MEMORY_GB', '8.0'))
memory_warning_threshold = float(os.getenv('MEMORY_WARNING_THRESHOLD', '75.0'))
memory_critical_threshold = float(os.getenv('MEMORY_CRITICAL_THRESHOLD', '85.0'))

print(f"메모리 최적화 설정:")
print(f"  - 최대 모델 메모리: {max_model_memory_gb}GB")
print(f"  - 메모리 경고 임계치: {memory_warning_threshold}%")
print(f"  - 메모리 위험 임계치: {memory_critical_threshold}%")

print("\n다운로드된 모델 로딩 테스트 중...")

# KoAlpaca-Polyglot-1.3B 테스트
try:
    tokenizer = AutoTokenizer.from_pretrained('beomi/KoAlpaca-Polyglot-1.3B', cache_dir='./models')
    model = AutoModelForCausalLM.from_pretrained('beomi/KoAlpaca-Polyglot-1.3B', cache_dir='./models', low_cpu_mem_usage=True)
    print('[SUCCESS] KoAlpaca-Polyglot-1.3B 모델 로딩 성공')
except Exception as e:
    print(f'[ERROR] KoAlpaca-Polyglot-1.3B 모델 로딩 실패: {e}')

# SBERT 모델 테스트
try:
    sbert_model = SentenceTransformer('jhgan/ko-sroberta-multitask', cache_folder='./models')
    print('[SUCCESS] SBERT 모델 로딩 성공')
except Exception as e:
    print(f'[ERROR] SBERT 모델 로딩 실패: {e}')

# SQLCoder 토크나이저 테스트
try:
    sql_tokenizer = AutoTokenizer.from_pretrained('defog/sqlcoder-7b-2', cache_dir='./models', trust_remote_code=True)
    print('[SUCCESS] SQLCoder 토크나이저 로딩 성공')
except Exception as e:
    print(f'[ERROR] SQLCoder 토크나이저 로딩 실패: {e}')

print('[SUCCESS] 모델 로딩 테스트 완료!')
PY

# 6단계: 메모리 모니터링 시작
echo "6단계: 메모리 모니터링 시스템 시작..."
python -c "
import sys
sys.path.append('/app')

try:
    from core.utils.memory_optimizer import start_memory_monitoring
    start_memory_monitoring()
    print('[SUCCESS] 메모리 모니터링 시스템 시작 완료')
except Exception as e:
    print(f'[WARNING] 메모리 모니터링 시작 실패: {e}')
    print('계속 진행합니다...')
"

# 7단계: 서버 시작(+옵션 스모크 테스트)
echo "7단계: 챗봇 서버 시작 중..."
echo "============================================================"
echo "[SUCCESS] 모든 초기화 단계가 완료되었습니다!"
echo "메모리 최적화 시스템이 활성화되었습니다."
echo "챗봇 서버를 시작합니다..."
echo "============================================================"

API_HOST=${API_HOST:-127.0.0.1}
API_PORT=${API_PORT:-8008}
API_BASE="http://$API_HOST:$API_PORT"

if [ "${SMOKE_TEST:-false}" = "true" ]; then
  echo "[INFO] SMOKE_TEST=true - 서버 백그라운드 실행 후 엔드포인트 점검을 수행합니다."
  python run_server.py &
  SERVER_PID=$!
  echo "[INFO] 서버 PID: $SERVER_PID"

  # 준비 대기
  for i in $(seq 1 30); do
    if curl -fsS "$API_BASE/healthz" >/dev/null 2>&1; then
      echo "[READY] healthz OK"
      break
    fi
    sleep 1
  done

  echo "=== 공개 엔드포인트 점검 ==="
  curl -fsS "$API_BASE/" && echo
  curl -fsS "$API_BASE/health" && echo
  curl -fsS "$API_BASE/healthz" && echo
  curl -fsS "$API_BASE/readyz" && echo
  curl -fsS "$API_BASE/metrics" >/dev/null && echo "metrics ok"
  curl -fsS "$API_BASE/ci/selftest" && echo

  if [ -n "$ADMIN_API_KEY" ]; then
    H=( -H "X-API-Key: $ADMIN_API_KEY" )
    echo "=== 설정/정책 점검 ==="
    curl -fsS "${H[@]}" "$API_BASE/config/info" && echo
    curl -fsS "${H[@]}" -X POST "$API_BASE/release/gate" && echo

    echo "=== 레이트리밋 리포트 ==="
    curl -fsS "${H[@]}" "$API_BASE/ratelimit/report" && echo

    echo "=== 벡터/저널 ==="
    curl -fsS "$API_BASE/vector-store-stats" && echo
    curl -fsS "${H[@]}" "$API_BASE/vector/journal" && echo
    curl -fsS "${H[@]}" -X POST "$API_BASE/vector/journal/replay" && echo

    echo "=== 품질 루프 ==="
    curl -fsS "${H[@]}" "$API_BASE/quality/goldens" && echo
    curl -fsS "${H[@]}" -H 'Content-Type: application/json' -X POST "$API_BASE/quality/offline-metrics?k=5" -d '[]' && echo
    curl -fsS "${H[@]}" "$API_BASE/quality/change-matrix" && echo

    echo "=== 카탈로그/승인 ==="
    curl -fsS "${H[@]}" -H 'Content-Type: application/json' -X POST "$API_BASE/catalog/requests" -d '{"id":"sample-dataset","version":"1.0","owner":"ops","desc":"테스트"}' && echo
    curl -fsS "${H[@]}" "$API_BASE/catalog/requests" && echo

    echo "=== 운영/보안 스텁 ==="
    curl -fsS "${H[@]}" "$API_BASE/ops/regression-summary" && echo
    curl -fsS "${H[@]}" -X POST "$API_BASE/ops/schedule/run?name=daily-compaction" && echo
    curl -fsS "${H[@]}" -X POST "$API_BASE/ops/security/scan" && echo
  fi

  echo "=== 질문 파이프라인(공개) 샘플 ==="
  curl -fsS -H 'Content-Type: application/json' -X POST "$API_BASE/ask" -d '{"question":"정수처리 공정 기본 원리 알려줘","pdf_id":"","user_id":"smoke","use_conversation_context":true,"max_chunks":5}' && echo

  echo "[INFO] 스모크 테스트 완료. 포그라운드로 전환합니다."
  wait $SERVER_PID
else
  # 서버 실행(포그라운드)
  exec python run_server.py
fi
