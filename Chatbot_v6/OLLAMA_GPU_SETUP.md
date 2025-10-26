# Ollama GPU 설정 가이드

## 현재 환경
- GPU: NVIDIA GeForce RTX 5060 Ti (16GB)
- CUDA Version: 12.9
- Driver Version: 576.88

## Ollama GPU 사용 확인

### 1. Ollama가 GPU를 사용하는지 확인
```powershell
# Ollama 실행 중 nvidia-smi로 확인
nvidia-smi

# Ollama 프로세스가 GPU를 사용하고 있는지 확인
# "ollama" 프로세스가 GPU 메모리를 사용하고 있어야 함
```

### 2. GPU 사용 강제 설정 (Windows)

Ollama는 기본적으로 GPU를 자동 감지하지만, 명시적으로 설정하려면:

#### 방법 1: 환경변수 설정 (PowerShell - 현재 세션)
```powershell
$env:CUDA_VISIBLE_DEVICES="0"
$env:OLLAMA_NUM_GPU="1"
ollama serve
```

#### 방법 2: 시스템 환경변수 설정 (영구적)
1. Windows 설정 → 시스템 → 정보 → 고급 시스템 설정
2. 환경 변수 클릭
3. 사용자 변수 또는 시스템 변수에 추가:
   - `CUDA_VISIBLE_DEVICES` = `0`
   - `OLLAMA_NUM_GPU` = `1`

#### 방법 3: Ollama 서비스 재시작
```powershell
# Ollama 프로세스 종료
taskkill /F /IM ollama.exe 2>$null

# GPU 환경변수와 함께 재시작
$env:CUDA_VISIBLE_DEVICES="0"
ollama serve
```

### 3. GPU 사용 확인 방법

```powershell
# 1. Ollama 모델 실행 중 nvidia-smi 확인
nvidia-smi -l 1  # 1초마다 갱신

# 2. Ollama 로그에서 GPU 사용 확인
# Ollama 시작 시 "CUDA" 또는 "GPU" 관련 메시지 확인
```

### 4. 성능 테스트

```powershell
# Python 스크립트로 응답 시간 측정
cd C:\Users\hanji\Documents\github\chatBot\Chatbot_v6
python -c "
import time
import urllib.request
import json

url = 'http://localhost:11434/api/generate'
data = {
    'model': 'qwen2.5:3b-instruct-q4_K_M',
    'prompt': '간단한 질문: 1+1은?',
    'stream': False,
    'options': {
        'num_ctx': 4096,
        'num_predict': 128
    }
}

start = time.time()
req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
response = urllib.request.urlopen(req, timeout=60)
elapsed = time.time() - start

print(f'응답 시간: {elapsed:.2f}초')
print(f'GPU 사용 시 예상: 2-5초')
print(f'CPU 사용 시 예상: 20-40초')
"
```

## 프로젝트 설정 업데이트 완료

### 1. 컨텍스트 윈도우 축소
- **변경 전**: `DEFAULT_LLM_NUM_CTX = 8192`
- **변경 후**: `DEFAULT_LLM_NUM_CTX = 4096`
- **위치**: `config/constants.py:107`
- **효과**: 처리 시간 약 30-40% 단축 예상

### 2. 리랭킹 대상 축소
- **변경 전**: `rerank_top_n: 50`
- **변경 후**: `rerank_top_n: 20`
- **위치**: `config/default.yaml:30`
- **효과**: 리랭킹 시간 약 50-60% 단축

### 3. GPU 설정 확인사항
- 프로젝트는 이미 GPU 사용 설정됨:
  - `use_gpu: true` (config/default.yaml:28)
  - `force_gpu_embedding: true` (config/default.yaml:34)
  - `force_gpu_faiss: true` (config/default.yaml:35)
  - `DEFAULT_EMBEDDING_DEVICE: "cuda"` (config/constants.py:100)

## 예상 성능 개선

### 변경 전 (40초)
- LLM 생성: ~35초 (CPU 사용 시)
- 리랭킹: ~2초
- 검색: ~2초
- 기타: ~1초

### 변경 후 (GPU 사용 시)
- LLM 생성: **~3초** (GPU 사용 + 컨텍스트 축소)
- 리랭킹: **~1초** (대상 축소)
- 검색: ~2초
- 기타: ~1초
- **총 예상: 7초 이내**

## 다음 단계

1. **Ollama GPU 사용 확인**:
   ```powershell
   # 벤치마크 실행 중 다른 터미널에서
   nvidia-smi -l 1
   ```

2. **벤치마크 재실행**:
   ```powershell
   python scripts/evaluate_qa_unified.py
   ```

3. **로그에서 시간 확인**:
   - 각 질문의 "시간: XXXms" 확인
   - GPU 사용 시: 3000-5000ms 예상
   - CPU 사용 시: 여전히 30000-40000ms

## 문제 해결

### Ollama가 여전히 느린 경우

1. **Ollama 재시작**:
   ```powershell
   taskkill /F /IM ollama.exe
   $env:CUDA_VISIBLE_DEVICES="0"
   ollama serve
   ```

2. **모델 재로드**:
   ```powershell
   # 모델 언로드 후 재로드
   curl -X POST http://localhost:11434/api/generate -d "{\"model\":\"qwen2.5:3b-instruct-q4_K_M\",\"keep_alive\":0}"
   ```

3. **CUDA 설치 확인**:
   ```powershell
   nvcc --version  # CUDA 컴파일러 버전 확인
   ```

### GPU 메모리 부족 시

컨텍스트를 더 줄일 수 있습니다:
```python
# config/constants.py
DEFAULT_LLM_NUM_CTX: Final[int] = 2048  # 4096 → 2048
```


