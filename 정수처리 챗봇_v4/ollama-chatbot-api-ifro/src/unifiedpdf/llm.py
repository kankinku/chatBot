from __future__ import annotations

import json
import urllib.request
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from .config import PipelineConfig
from .types import RetrievedSpan
from .timeouts import LLM_TIMEOUT_S, run_with_timeout
import re


_LOG_PATH = os.path.join("logs", "llm_errors.log")
os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
logging.basicConfig(level=logging.ERROR)
_logger = logging.getLogger("unifiedpdf.llm")
if not _logger.handlers:
    _fh = logging.FileHandler(_LOG_PATH, encoding="utf-8")
    _fh.setLevel(logging.ERROR)
    _logger.addHandler(_fh)


def _extract_keywords_from_question(question: str, domain_dict: dict = None, qtype: str = "general") -> List[str]:
    """Domain Dictionary를 활용한 키워드 추출"""
    keywords = []
    
    # Domain Dictionary가 없으면 기본 방식 사용
    if domain_dict is None:
        # 기본 키워드 추출 (하위 호환성)
        tech_terms = re.findall(r'[가-힣]+(?:공정|모델|알고리즘|시스템|플랫폼|입력변수|수질|인자)', question)
        keywords.extend(tech_terms)
        
        nouns = re.findall(r'[가-힣]{2,}', question)
        keywords.extend([noun for noun in nouns if len(noun) >= 2])
        
        english_terms = re.findall(r'[A-Za-z]+', question)
        keywords.extend(english_terms)
        
        return list(set(keywords))
    
    # 1. 질문 유형별 특화 키워드 우선 추출
    if qtype in domain_dict:
        type_keywords = domain_dict[qtype]
        for kw in type_keywords:
            if kw.lower() in question.lower():
                keywords.append(kw)
    
    # 2. 일반 키워드에서 매칭
    general_keywords = domain_dict.get("keywords", [])
    for kw in general_keywords:
        if kw.lower() in question.lower():
            keywords.append(kw)
    
    # 3. 동의어 확장
    synonyms = domain_dict.get("synonyms", {})
    for main_term, synonym_list in synonyms.items():
        if main_term.lower() in question.lower():
            keywords.append(main_term)
            keywords.extend(synonym_list)
    
    # 4. 단위 정보도 키워드로 활용
    units = domain_dict.get("units", [])
    for unit in units:
        if unit in question:
            keywords.append(unit)
    
    # 5. 기본 명사 추출 (도메인 키워드에 없는 경우)
    nouns = re.findall(r'[가-힣]{2,}', question)
    for noun in nouns:
        if len(noun) >= 2 and noun not in keywords:
            keywords.append(noun)
    
    # 6. 영어 용어 추출
    english_terms = re.findall(r'[A-Za-z]+', question)
    keywords.extend(english_terms)
    
    # 중복 제거 및 정리
    keywords = list(set(keywords))
    return keywords


def _select_best_contexts(contexts: List[RetrievedSpan], question: str, max_contexts: int = 4, domain_dict: dict = None, qtype: str = "general") -> List[RetrievedSpan]:
    """Domain Dictionary를 활용한 키워드 매칭 기반 최적의 컨텍스트 선택"""
    if len(contexts) <= max_contexts:
        return contexts
    
    keywords = _extract_keywords_from_question(question, domain_dict, qtype)
    if not keywords:
        return contexts[:max_contexts]
    
    # 각 컨텍스트에 대해 키워드 매칭 점수 계산
    scored_contexts = []
    for context in contexts:
        text_lower = context.chunk.text.lower()
        keyword_matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        
        # 도메인 특화 키워드에 더 높은 가중치 부여
        domain_keyword_matches = 0
        high_priority_matches = 0
        
        if domain_dict:
            # 질문 유형별 키워드 매칭
            if qtype in domain_dict:
                type_keywords = domain_dict[qtype]
                domain_keyword_matches += sum(1 for kw in type_keywords if kw.lower() in text_lower)
            
            # 일반 도메인 키워드 매칭
            general_keywords = domain_dict.get("keywords", [])
            domain_keyword_matches += sum(1 for kw in general_keywords if kw.lower() in text_lower)
            
            # 고우선순위 키워드 매칭 (URL, 수치, 계정 정보 등)
            # 고우선순위 키워드들 (domain_dictionary에서 로드)
            high_priority_keywords = domain_dict.get("high_priority_keywords", []) if domain_dict else []
            high_priority_matches = sum(1 for kw in high_priority_keywords if kw.lower() in text_lower)
        
        # 점수 = 원래 점수 + 키워드 매칭 보너스 + 도메인 키워드 보너스 + 고우선순위 보너스
        basic_bonus = keyword_matches * 0.1  # 기본 키워드 매칭당 0.1점
        domain_bonus = domain_keyword_matches * 0.2  # 도메인 키워드 매칭당 0.2점
        priority_bonus = high_priority_matches * 0.3  # 고우선순위 키워드 매칭당 0.3점
        total_score = context.score + basic_bonus + domain_bonus + priority_bonus
        
        scored_contexts.append((total_score, context))
    
    # 점수 순으로 정렬하고 상위 선택
    scored_contexts.sort(key=lambda x: x[0], reverse=True)
    return [ctx for _, ctx in scored_contexts[:max_contexts]]


def _format_prompt(question: str, contexts: List[RetrievedSpan], qtype: str = "general") -> str:
    # 자연스러운 챗봇 스타일 프롬프트
    parts = [
        "안녕하세요! 정수장 시스템에 대해 궁금한 점이 있으시군요.",
        "",
        "제가 문서를 확인해서 친근하게 답변드릴게요.",
        "",
        "문서 내용:",
    ]
    
    # 문서를 최대 2개만 사용하고 각각 150자로 제한
    for i, s in enumerate(contexts[:2], start=1):
        text = s.chunk.text[:150] + "..." if len(s.chunk.text) > 150 else s.chunk.text
        parts.append(f"{i}. {text}")
    
    parts.extend([
        "",
        f"질문: {question}",
        "",
        "답변:"
    ])
    
    return "\n".join(parts)


def _format_prompt_v2(question: str, contexts: List[RetrievedSpan], qtype: str = "general", domain_dict: dict = None) -> str:
    """Domain Dictionary를 활용한 개선된 문서 중심 답변 프롬프트"""
    
    # 질문 유형별 맞춤형 프롬프트 전략 (개선된 버전)
    if qtype in ["technical_spec", "definition"]:
        # 기술사양이나 정의형 질문은 더 상세한 정보 필요
        max_contexts = 6  # 4 → 6개로 증가
        text_length = 800  # 500 → 800자로 증가
        instruction_emphasis = "기술적 세부사항과 정확한 용어를 포함하여"
    elif qtype in ["system_info", "numeric"]:
        # 시스템 정보나 수치형 질문은 정확한 데이터 중요
        max_contexts = 5  # 3 → 5개로 증가
        text_length = 600  # 400 → 600자로 증가
        instruction_emphasis = "정확한 수치, URL, 계정 정보를 그대로"
    else:
        # 일반적인 질문
        max_contexts = 4  # 3 → 4개로 증가
        text_length = 500  # 400 → 500자로 증가
        instruction_emphasis = "문서 내용을 바탕으로"
    
    # Domain Dictionary를 활용한 키워드 기반 최적의 컨텍스트 선택
    selected_contexts = _select_best_contexts(contexts, question, max_contexts, domain_dict, qtype)
    
    parts = [
        "당신은 고산 정수장 시스템 사용자 설명서 전문 챗봇입니다.",
        "",
        "중요: 반드시 제공된 문서 내용만을 기반으로 답변하세요. 외부 지식이나 일반적인 정보는 사용하지 마세요.",
        "",
        "문서 내용:",
    ]
    
    # 더 많은 문서와 더 긴 텍스트 사용 (chunk overlap 포함)
    for i, context in enumerate(selected_contexts, start=1):
        full_text = context.chunk.text
        
        # Chunk overlap을 고려한 텍스트 선택
        if len(full_text) > text_length:
            # 문장 경계를 고려한 텍스트 자르기
            truncated_text = full_text[:text_length]
            
            # 마지막 문장이 완전하지 않으면 이전 문장까지로 조정
            last_period = truncated_text.rfind('.')
            last_newline = truncated_text.rfind('\n')
            last_break = max(last_period, last_newline)
            
            if last_break > text_length * 0.8:  # 80% 이상이면 문장 경계에서 자르기
                truncated_text = truncated_text[:last_break + 1]
            
            text = truncated_text + "..."
        else:
            text = full_text
            
        parts.append(f"[문서 {i}] {text}")
    
    parts.extend([
        "",
        f"질문: {question}",
        "",
        "지침:",
        f"- 위 문서 내용에서만 {instruction_emphasis} 답변을 찾아주세요",
        "- 문서에 없는 정보는 '문서에서 확인할 수 없습니다'라고 답변하세요",
        "- 정확한 수치, 날짜, URL, 계정 정보를 그대로 사용하세요",
        "- 추측이나 일반적인 정보는 사용하지 마세요",
        "- 답변은 한국어를 사용해주세요(url이나 전문 용어는 예외)",
        "- 관련 키워드나 용어가 문서에 있다면 반드시 포함하세요",
        "- 문서 내용을 바탕으로 직접적인 답변을 제공하세요. 추론이나 해석은 피하세요",
        "- 답변은 간결하고 명확하게 작성하세요 (최대 3-4문장)",
        "- 문서 내용을 그대로 복사하지 말고 요약하여 자연스럽게 답변하세요",
        "- 핵심 정보만 포함하고 불필요한 세부사항은 생략하세요",
        "- 출처 표시나 문서 번호는 포함하지 마세요",
        "- 이모지나 특수 기호는 사용하지 마세요",
        "- 자연스러운 대화체로 답변하세요",
        "",
        "답변:"
    ])
    
    return "\n".join(parts)


def _format_prompt_recovery(question: str, contexts: List[RetrievedSpan], qtype: str = "general", domain_dict: dict = None) -> str:
    """Domain Dictionary를 활용한 개선된 엄격한 문서 중심 복구 프롬프트"""
    
    # 복구 시에는 더 많은 컨텍스트와 더 긴 텍스트 사용
    max_contexts = 5
    text_length = 600
    
    # Domain Dictionary를 활용한 키워드 기반 최적의 컨텍스트 선택
    selected_contexts = _select_best_contexts(contexts, question, max_contexts, domain_dict, qtype)
    
    parts = [
        "고산 정수장 시스템 사용자 설명서 전문 챗봇입니다.",
        "",
        "경고: 오직 제공된 문서 내용만 사용하세요. 외부 지식은 절대 사용하지 마세요.",
        "",
        "문서 내용:",
    ]
    
    # 더 많은 문서와 더 긴 텍스트 사용
    for i, context in enumerate(selected_contexts, start=1):
        text = context.chunk.text[:text_length] + "..." if len(context.chunk.text) > text_length else context.chunk.text
        parts.append(f"[문서 {i}] {text}")
    
    parts.extend([
        "",
        f"질문: {question}",
        "",
        "엄격한 지침:",
        "- 문서에 명시된 정보만 답변하세요",
        "- 문서에 없는 내용은 '문서에서 확인할 수 없습니다'라고만 답변하세요",
        "- 수치, 날짜, URL, 계정 정보는 문서의 정확한 값을 사용하세요",
        "- 일반적인 추측이나 외부 지식은 사용하지 마세요",
        "- 관련 키워드나 전문 용어가 문서에 있다면 반드시 포함하세요",
        "- 여러 문서에서 관련 정보를 종합하여 답변하세요",
        "- 답변은 간결하고 명확하게 작성하세요 (최대 3-4문장)",
        "- 문서 내용을 그대로 복사하지 말고 요약하여 자연스럽게 답변하세요",
        "- 핵심 정보만 포함하고 불필요한 세부사항은 생략하세요",
        "- 출처 표시나 문서 번호는 포함하지 마세요",
        "- 이모지나 특수 기호는 사용하지 마세요",
        "- 자연스러운 대화체로 답변하세요",
        "",
        "답변:"
    ])
    
    return "\n".join(parts)


def ollama_generate(prompt: str, model_name: str, timeout_s: Optional[int] = None) -> str:
    # Docker 환경에서는 ollama 서비스명 사용, 로컬에서는 localhost 사용
    ollama_host = os.getenv('OLLAMA_HOST', 'ollama')
    url = f"http://{ollama_host}:11434/api/generate"
    data = {
        "model": model_name, 
        "prompt": prompt, 
        "stream": False,
        "keep_alive": "24h",  # 모델을 메모리에 유지하여 응답 속도 향상
        "options": {
            "temperature": 0.0,  # llama3.1 모델에 적합한 매우 낮은 온도
            "top_p": 0.9,        # 높은 집중도
            "top_k": 40,         # 어휘 다양성 조절
            "repeat_penalty": 1.1,  # 반복 방지
            "num_ctx": 8192,     # llama3.1 모델에 적합한 컨텍스트 길이
            "num_predict": 512,  # 더 긴 답변 가능
        }
    }
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s or LLM_TIMEOUT_S) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="ignore"))
            return body.get("response", "")
    except Exception as e:
        _logger.error("Ollama request failed: %s", e)
        return ""


def _post_process_answer(text: str) -> str:
    """간단한 답변 후처리 (길이 제한 포함)"""
    if not text:
        return ""
    
    # 기본 정리
    text = text.strip()
    
    # [답변] 형식 제거
    if text.startswith("[답변]"):
        text = text[4:].strip()
    
    # 길이 제한 (최대 500자)
    if len(text) > 500:
        # 문장 단위로 자르기
        sentences = text.split('.')
        truncated_text = ""
        for sentence in sentences:
            if len(truncated_text + sentence + '.') <= 500:
                truncated_text += sentence + '.'
            else:
                break
        text = truncated_text.rstrip('.') + '.'
    
    # 개행문자 정리
    text = text.replace("\\n", " ")
    text = text.replace("\n", " ")
    
    # 연속된 공백 정리
    import re
    text = re.sub(r'\s+', ' ', text)
    
    # 출처 표시 제거
    text = re.sub(r'\(문서\s*\d+\)', '', text)
    text = re.sub(r'문서\s*\d+에서\s*확인할\s*수\s*있습니다', '', text)
    text = re.sub(r'문서\s*\d+의\s*.*?부분에서', '', text)
    
    # 이모지 및 특수 기호 제거
    text = re.sub(r'❍|●|○|◆|◇|■|□|▲|△|▼|▽', '', text)
    text = re.sub(r'[^\w\s가-힣.,!?()\-]', '', text)
    
    # 연속된 공백 다시 정리
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def generate_answer(question: str, contexts: List[RetrievedSpan], cfg: PipelineConfig, qtype: str = "general", recovery: bool = False) -> str:
    # Domain Dictionary 로드
    domain_dict = None
    try:
        import json
        domain_dict_path = cfg.domain.domain_dict_path
        if domain_dict_path and Path(domain_dict_path).exists():
            with open(domain_dict_path, 'r', encoding='utf-8') as f:
                domain_dict = json.load(f)
    except Exception as e:
        _logger.warning(f"Domain dictionary 로드 실패: {e}")
    
    # Use improved prompt format (and stricter recovery prompt if requested)
    if recovery:
        prompt = _format_prompt_recovery(question, contexts, qtype=qtype, domain_dict=domain_dict)
    else:
        prompt = _format_prompt_v2(question, contexts, qtype=qtype, domain_dict=domain_dict)

    def _call():
        return ollama_generate(prompt, cfg.model_name, timeout_s=LLM_TIMEOUT_S)

    # Retries with backoff
    tries = max(0, int(getattr(cfg, "llm_retries", 0))) + 1
    backoff = int(getattr(cfg, "llm_retry_backoff_ms", 300)) / 1000.0
    text = ""
    for t in range(tries):
        text = run_with_timeout(_call, timeout_s=LLM_TIMEOUT_S + 2, default="")  # small cushion
        if text.strip():
            break
        time.sleep(backoff * (t + 1))
    
    # 답변 품질 검사 및 재시도
    if text.strip() and _is_answer_insufficient(text.strip()):
        # 확장된 컨텍스트로 재시도
        recovery_prompt = _format_prompt_recovery(question, contexts, qtype=qtype, domain_dict=domain_dict)
        try:
            recovery_text = run_with_timeout(
                lambda: ollama_generate(recovery_prompt, cfg.model_name, timeout_s=LLM_TIMEOUT_S),
                timeout_s=LLM_TIMEOUT_S + 2, 
                default=""
            )
            if recovery_text.strip() and not _is_answer_insufficient(recovery_text.strip()):
                return _post_process_answer(recovery_text)
        except Exception:
            pass
    
    final_answer = _post_process_answer(text) or _extractive_fallback_answer(question, contexts, domain_dict)
    
    # 답변 실패 케이스 로깅
    if _is_answer_insufficient(final_answer):
        _log_failed_answer(question, contexts, final_answer, qtype)
    
    return final_answer

def _is_answer_insufficient(answer: str) -> bool:
    """답변이 불충분한지 판단"""
    insufficient_indicators = [
        "문서에서 확인할 수 없습니다",
        "문서에 명시되어 있지 않습니다",
        "문서에서 찾을 수 없습니다",
        "문서에 해당 정보가 없습니다",
        "확인할 수 없습니다"
    ]
    
    answer_lower = answer.lower()
    return any(indicator in answer_lower for indicator in insufficient_indicators)

def _extractive_fallback_answer(question: str, contexts: List[RetrievedSpan], domain_dict: dict) -> str:
    """추출적 답변 모드 (문서에서 직접 추출) + 정답 후보 highlight"""
    if not contexts:
        return "문서에서 관련 정보를 찾을 수 없습니다."
    
    # 질문에서 키워드 추출
    keywords = _extract_keywords_from_question(question, domain_dict)
    
    # 가장 관련성 높은 컨텍스트 선택
    best_contexts = _select_best_contexts(contexts, question, 1, domain_dict)
    if not best_contexts:
        return "문서에서 관련 정보를 찾을 수 없습니다."
    
    best_context = best_contexts[0]
    
    # 정답 후보 highlight 기능
    highlighted_info = _extract_highlighted_info(best_context.chunk.text, question, keywords)
    if highlighted_info:
        return highlighted_info
    
    # 키워드가 포함된 문장들을 추출
    sentences = best_context.chunk.text.split('.')
    relevant_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(kw.lower() in sentence_lower for kw in keywords):
            relevant_sentences.append(sentence.strip())
    
    if relevant_sentences:
        return ". ".join(relevant_sentences[:3]) + "."  # 최대 3개 문장
    else:
        # 키워드 매칭이 없으면 첫 번째 문장 반환
        first_sentence = sentences[0].strip() if sentences else best_context.chunk.text[:200]
        return first_sentence + "..."

def _extract_highlighted_info(text: str, question: str, keywords: List[str]) -> str:
    """정답 후보 highlight 기능 (수치, URL, 변수명 우선 추출)"""
    import re
    
    question_lower = question.lower()
    
    # URL 패턴 추출
    url_pattern = r'https?://[^\s]+|waio-portal-vip[^\s]*|10\.\d+\.\d+\.\d+[^\s]*'
    urls = re.findall(url_pattern, text)
    if urls and any(term in question_lower for term in ["url", "주소", "접속", "대시보드"]):
        return f"URL: {', '.join(urls)}"
    
    # 수치 패턴 추출 (포트 번호, 수치 등)
    numeric_pattern = r'\d+\.?\d*'
    numbers = re.findall(numeric_pattern, text)
    if numbers and any(term in question_lower for term in ["포트", "번호", "수치", "값"]):
        # 포트 번호 우선 (10011 등)
        port_numbers = [n for n in numbers if len(n) >= 4 and n.startswith('10')]
        if port_numbers:
            return f"포트 번호: {', '.join(port_numbers)}"
    
    # 입력변수 관련 추출
    if any(term in question_lower for term in ["입력변수", "수질", "인자"]):
        # 수질 인자 키워드 매칭
        water_quality_terms = ["원수", "탁도", "pH", "온도", "알칼리도", "전기전도도", "응집제", "주입률", "NaOH", "활성탄"]
        found_terms = [term for term in water_quality_terms if term.lower() in text.lower()]
        if found_terms:
            return f"수질 인자: {', '.join(found_terms)}"
    
    # 발행일 및 컨소시엄 정보 추출
    if any(term in question_lower for term in ["발행일", "컨소시엄", "작성"]):
        # 날짜 패턴 추출
        date_pattern = r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일|\d{4}\.\d{1,2}\.\d{1,2}|\d{4}-\d{1,2}-\d{1,2}'
        dates = re.findall(date_pattern, text)
        if dates:
            return f"발행일: {', '.join(dates)}"
        
        # 컨소시엄 정보 추출
        consortium_pattern = r'[가-힣]+컨소시엄|에셈블'
        consortiums = re.findall(consortium_pattern, text)
        if consortiums:
            return f"컨소시엄: {', '.join(consortiums)}"
    
    # 시스템 목적 관련 추출
    if any(term in question_lower for term in ["목적", "주요", "시스템"]):
        # 목적 관련 키워드 매칭
        purpose_keywords = ["자율운영", "AI", "활용성", "안정적", "정보제공", "시운전", "성능검증"]
        found_purpose = [kw for kw in purpose_keywords if kw.lower() in text.lower()]
        if found_purpose:
            return f"시스템 목적: {', '.join(found_purpose)}"
    
    # 혼화응집 공정 목표 추출
    if any(term in question_lower for term in ["혼화응집", "목표", "산출"]):
        # 혼화응집 관련 키워드 매칭
        mixing_keywords = ["혼화", "응집", "목표", "산출", "예측", "제어"]
        found_mixing = [kw for kw in mixing_keywords if kw.lower() in text.lower()]
        if found_mixing:
            return f"혼화응집 목표: {', '.join(found_mixing)}"
    
    # EMS 자동/부분 모드 차이점 추출
    if any(term in question_lower for term in ["ems", "자동모드", "부분모드", "차이점"]):
        # EMS 관련 키워드 매칭
        ems_keywords = ["자동모드", "부분모드", "ems", "송수펌프", "제어"]
        found_ems = [kw for kw in ems_keywords if kw.lower() in text.lower()]
        if found_ems:
            return f"EMS 모드: {', '.join(found_ems)}"
    
    # PMS 펌프 결함 정보 추출
    if any(term in question_lower for term in ["pms", "펌프", "결함", "진단"]):
        # PMS 관련 키워드 매칭
        pms_keywords = ["pms", "펌프", "결함", "진단", "모터", "진동", "온도"]
        found_pms = [kw for kw in pms_keywords if kw.lower() in text.lower()]
        if found_pms:
            return f"PMS 결함 정보: {', '.join(found_pms)}"
    
    # 계정 정보 추출
    if any(term in question_lower for term in ["계정", "사용자", "로그인"]):
        account_pattern = r'[가-힣]+\([^)]+\)|[A-Za-z]+@[A-Za-z]+\.[A-Za-z]+'
        accounts = re.findall(account_pattern, text)
        if accounts:
            return f"계정 정보: {', '.join(accounts)}"
    
    return ""

def _log_failed_answer(question: str, contexts: List[RetrievedSpan], answer: str, qtype: str):
    """답변 불능 케이스 로깅"""
    import json
    from datetime import datetime
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "question_type": qtype,
        "context_count": len(contexts),
        "context_scores": [ctx.score for ctx in contexts[:3]],  # 상위 3개 점수
        "answer": answer,
        "failure_reason": "insufficient_answer" if _is_answer_insufficient(answer) else "no_answer"
    }
    
    try:
        log_file = "logs/failed_answers.jsonl"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        _logger.warning(f"답변 실패 로깅 실패: {e}")
