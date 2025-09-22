from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum


class QuestionType(Enum):
    DEFINITION = "definition"  # 정의형 질문
    PROCEDURAL = "procedural"  # 절차형 질문
    NUMERIC = "numeric"  # 수치형 질문
    COMPARATIVE = "comparative"  # 비교형 질문
    PROBLEM_SOLVING = "problem"  # 문제해결형 질문
    TECHNICAL_SPEC = "technical_spec"  # 기술사양형 질문
    SYSTEM_INFO = "system_info"  # 시스템정보형 질문
    UNKNOWN = "unknown"  # 분류 불가


class QuestionClassifier:
    def __init__(self, domain_dict_path: Optional[str] = None):
        self.domain_dict_path = domain_dict_path or "data/domain_dictionary.json"
        self.keywords = self._load_keywords()
        self.question_patterns = self._create_question_patterns()
    
    def _load_keywords(self) -> Dict[str, List[str]]:
        """도메인 사전에서 키워드 로드"""
        try:
            dict_path = Path(self.domain_dict_path)
            if dict_path.exists():
                with dict_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    return {
                        "definition": data.get("definition", []),
                        "procedural": data.get("procedural", []),
                        "comparative": data.get("comparative", []),
                        "problem": data.get("problem", []),
                        "technical_spec": data.get("technical_spec", []),
                        "system_info": data.get("system_info", []),
                    }
        except Exception:
            pass
        return {}
    
    def _create_question_patterns(self) -> Dict[QuestionType, List[str]]:
        """질문 유형별 패턴 생성"""
        return {
            QuestionType.DEFINITION: [
                r"무엇(?:인가요|입니까|이에요|예요)",
                r"정의(?:는|이|가)",
                r"의미(?:는|가)",
                r"개념(?:은|이)",
                r"설명(?:해주세요|해주시면)",
                r"란\s*(?:무엇|뭐)",
                r"이\s*(?:무엇|뭐)",
            ],
            QuestionType.PROCEDURAL: [
                r"어떻게\s*(?:하나요|하나|할까요|할까)",
                r"방법(?:은|이|가)",
                r"절차(?:는|가)",
                r"순서(?:는|가)",
                r"실행(?:하는|하는|할)",
                r"시작(?:하는|하는|할)",
                r"중지(?:하는|하는|할)",
                r"재시작(?:하는|하는|할)",
                r"복구(?:하는|하는|할)",
                r"설정(?:하는|하는|할)",
                r"접속(?:하는|하는|할)",
                r"로그인(?:하는|하는|할)",
            ],
            QuestionType.NUMERIC: [
                r"얼마(?:인가요|입니까|예요|이에요)",
                r"값(?:은|이|가)",
                r"수치(?:는|가)",
                r"계수(?:는|가)",
                r"비율(?:은|이|가)",
                r"퍼센트(?:는|가)",
                r"몇\s*(?:개|명|번|회|차)",
                r"몇\s*(?:시|분|초|일|월|년)",
                r"상관계수(?:는|가)",
                r"R²(?:는|가)",
                r"발행일(?:은|이|가)",
                r"작성일(?:은|이|가)",
            ],
            QuestionType.COMPARATIVE: [
                r"차이(?:점|는|가)",
                r"비교(?:하면|하면|하면)",
                r"vs\s*(?:는|는|는)",
                r"더\s*(?:좋|나쁘|크|작|높|낮)",
                r"장점(?:은|이|가)",
                r"단점(?:은|이|가)",
                r"우수(?:한|한|한)",
                r"열등(?:한|한|한)",
                r"효율(?:은|이|가)",
                r"성능(?:은|이|가)",
            ],
            QuestionType.PROBLEM_SOLVING: [
                r"문제(?:는|가|이)",
                r"오류(?:는|가|이)",
                r"이상(?:은|이|가)",
                r"고장(?:은|이|가)",
                r"원인(?:은|이|가)",
                r"대응(?:은|이|가)",
                r"대책(?:은|이|가)",
                r"해결(?:방법|책|방안)",
                r"증상(?:은|이|가)",
                r"장애(?:는|가|이)",
                r"트러블(?:은|이|가)",
                r"이슈(?:는|가|이)",
                r"에러(?:는|가|이)",
                r"실패(?:는|가|이)",
                r"불량(?:은|이|가)",
                r"결함(?:은|이|가)",
            ],
            QuestionType.TECHNICAL_SPEC: [
                r"모델(?:은|이|가)",
                r"알고리즘(?:은|이|가)",
                r"성능(?:은|이|가)",
                r"지표(?:는|가|이)",
                r"설정값(?:은|이|가)",
                r"고려사항(?:은|이|가)",
                r"매개변수(?:는|가|이)",
                r"하이퍼파라미터(?:는|가|이)",
                r"최적화(?:는|가|이)",
                r"N-beats(?:는|가|이)",
                r"LSTM(?:은|이|가)",
                r"GRU(?:는|가|이)",
                r"GBR(?:은|이|가)",
            ],
            QuestionType.SYSTEM_INFO: [
                r"URL(?:은|이|가)",
                r"계정(?:은|이|가)",
                r"비밀번호(?:는|가|이)",
                r"로그인(?:은|이|가)",
                r"접속주소(?:는|가|이)",
                r"IP(?:는|가|이)",
                r"포트(?:는|가|이)",
                r"서버(?:는|가|이)",
                r"호스트(?:는|가|이)",
                r"도메인(?:은|이|가)",
                r"네트워크(?:는|가|이)",
                r"연결(?:은|이|가)",
                r"통신(?:은|이|가)",
                r"프로토콜(?:은|이|가)",
            ]
        }
    
    def classify_question(self, question: str) -> Tuple[QuestionType, float, Dict[str, float]]:
        """질문을 분류하고 신뢰도 점수 반환"""
        question_lower = question.lower().strip()
        
        # 패턴 매칭 점수 계산
        pattern_scores = {}
        for qtype, patterns in self.question_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, question_lower))
                score += matches * 0.3  # 패턴 매칭 가중치
            
            # 키워드 매칭 점수 추가
            if qtype.value in self.keywords:
                keyword_score = 0.0
                for keyword in self.keywords[qtype.value]:
                    if keyword.lower() in question_lower:
                        keyword_score += 0.1
                score += keyword_score
            
            pattern_scores[qtype.value] = score
        
        # 최고 점수 질문 유형 선택
        if pattern_scores:
            best_type = max(pattern_scores.items(), key=lambda x: x[1])
            if best_type[1] > 0.1:  # 최소 임계값
                return QuestionType(best_type[0]), best_type[1], pattern_scores
        
        return QuestionType.UNKNOWN, 0.0, pattern_scores
    
    def get_answer_template(self, question_type: QuestionType) -> str:
        """질문 유형별 답변 템플릿 반환"""
        templates = {
            QuestionType.DEFINITION: "문서에 따르면 {answer}입니다.",
            QuestionType.PROCEDURAL: "다음과 같은 절차로 진행하시면 됩니다: {answer}",
            QuestionType.NUMERIC: "확인 결과 {answer}입니다.",
            QuestionType.COMPARATIVE: "비교 분석 결과는 다음과 같습니다: {answer}",
            QuestionType.PROBLEM_SOLVING: "문제 해결 방법은 다음과 같습니다: {answer}",
            QuestionType.TECHNICAL_SPEC: "기술 사양은 다음과 같습니다: {answer}",
            QuestionType.SYSTEM_INFO: "시스템 정보는 다음과 같습니다: {answer}",
            QuestionType.UNKNOWN: "{answer}"
        }
        return templates.get(question_type, "{answer}")
    
    def enhance_query_for_type(self, query: str, question_type: QuestionType) -> str:
        """질문 유형에 따라 쿼리 강화"""
        enhanced_query = query
        
        if question_type == QuestionType.NUMERIC:
            # 수치 관련 키워드 강화
            numeric_keywords = ["수치", "값", "계수", "비율", "퍼센트", "상관계수", "R²", "발행일", "작성일"]
            for keyword in numeric_keywords:
                if keyword in query:
                    enhanced_query += f" {keyword} {keyword}"  # 키워드 반복으로 강조
        
        elif question_type == QuestionType.SYSTEM_INFO:
            # 시스템 정보 관련 키워드 강화
            system_keywords = ["URL", "계정", "비밀번호", "로그인", "접속", "IP", "포트", "서버"]
            for keyword in system_keywords:
                if keyword in query:
                    enhanced_query += f" {keyword} {keyword}"
        
        elif question_type == QuestionType.TECHNICAL_SPEC:
            # 기술 사양 관련 키워드 강화
            tech_keywords = ["모델", "알고리즘", "성능", "지표", "설정값", "N-beats", "LSTM", "GRU"]
            for keyword in tech_keywords:
                if keyword in query:
                    enhanced_query += f" {keyword} {keyword}"
        
        return enhanced_query
