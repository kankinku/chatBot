"""
PDF 키워드 추출 및 캐시 관리 모듈

PDF 전처리 과정에서 자주 사용된 단어들을 추출하여 파이프라인 설정에 추가하는 기능
"""

import re
import logging
from typing import List, Dict, Set, Counter
from collections import Counter
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class PDFKeywordExtractor:
    """PDF 키워드 추출 및 캐시 관리 클래스"""
    
    def __init__(self, cache_threshold: int = 5):
        """
        키워드 추출기 초기화
        
        Args:
            cache_threshold: 키워드 추가 임계값 (최소 등장 횟수)
        """
        self.cache_threshold = cache_threshold
        self.keyword_cache = Counter()  # 메모리 기반 캐시
        self.extracted_keywords = set()  # 이미 추출된 키워드들
        
        # 한국어 불용어 목록
        self.stop_words = {
            '이', '그', '저', '것', '수', '등', '및', '또는', '그리고', '하지만',
            '그러나', '따라서', '그래서', '때문에', '위해서', '통해서', '의해서',
            '에서', '으로', '에게', '에서', '까지', '부터', '까지', '사이',
            '안에', '밖에', '위에', '아래에', '앞에', '뒤에', '옆에',
            '있다', '없다', '하다', '되다', '이다', '아니다', '있다가', '없다가',
            '하겠다', '되겠다', '이겠다', '아니겠다', '하고', '되고', '이고', '아니고',
            '하며', '되며', '이며', '아니며', '하거나', '되거나', '이거나', '아니거나',
            '하지만', '되지만', '이지만', '아니지만', '하므로', '되므로', '이므로', '아니므로',
            '하여', '되어', '이어', '아니어', '하니', '되니', '이니', '아니니',
            '하면', '되면', '이면', '아니면', '하던', '되던', '이던', '아니던',
            '하든', '되든', '이든', '아니든', '하든지', '되든지', '이든지', '아니든지',
            '하자', '되자', '이자', '아니자', '하셨다', '되셨다', '이셨다', '아니셨다',
            '하신다', '되신다', '이신다', '아니신다', '하시는', '되시는', '이시는', '아니시는',
            '하시고', '되시고', '이시고', '아니시고', '하시며', '되시며', '이시며', '아니시며',
            '하시거나', '되시거나', '이시거나', '아니시거나', '하시지만', '되시지만', '이시지만', '아니시지만',
            '하시므로', '되시므로', '이시므로', '아니시므로', '하셔서', '되셔서', '이셔서', '아니셔서',
            '하시니', '되시니', '이시니', '아니시니', '하시면', '되시면', '이시면', '아니시면',
            '하시던', '되시던', '이시던', '아니시던', '하시든', '되시든', '이시든', '아니시든',
            '하시든지', '되시든지', '이시든지', '아니시든지', '하시자', '되시자', '이시자', '아니시자',
            '하셨다', '되셨다', '이셨다', '아니셨다', '하신다', '되신다', '이신다', '아니신다',
            '하시는', '되시는', '이시는', '아니시는', '하시고', '되시고', '이시고', '아니시고',
            '하시며', '되시며', '이시며', '아니시며', '하시거나', '되시거나', '이시거나', '아니시거나',
            '하시지만', '되시지만', '이시지만', '아니시지만', '하시므로', '되시므로', '이시므로', '아니시므로',
            '하셔서', '되셔서', '이셔서', '아니셔서', '하시니', '되시니', '이시니', '아니시니',
            '하시면', '되시면', '이시면', '아니시면', '하시던', '되시던', '이시던', '아니시던',
            '하시든', '되시든', '이시든', '아니시든', '하시든지', '되시든지', '이시든지', '아니시든지',
            '하시자', '되시자', '이시자', '아니시자'
        }
        
        logger.info(f"[SUCCESS] PDF 키워드 추출기 초기화 (임계값: {cache_threshold})")
    
    def extract_keywords_from_text(self, text: str) -> List[str]:
        """
        텍스트에서 키워드 추출 (개선된 버전)
        
        Args:
            text: 추출할 텍스트
            
        Returns:
            추출된 키워드 리스트 (최대 20개)
        """
        # 텍스트 전처리
        text = self._preprocess_text(text)
        
        # 단어 분할 (한국어 + 영어)
        words = self._tokenize_text(text)
        
        # 불용어 제거 및 필터링
        valid_keywords = []
        for word in words:
            if self._is_valid_keyword(word):
                valid_keywords.append(word)
        
        # 키워드 빈도수 업데이트
        self.keyword_cache.update(valid_keywords)
        
        # 중요도 기반 키워드 선택 (최대 20개)
        keywords = self._select_important_keywords(valid_keywords, text)
        
        # 로그 출력 (개선된 버전)
        if keywords:
            logger.debug(f"[SUCCESS] 키워드 추출 완료: {len(keywords)}개 (전체 후보: {len(valid_keywords)}개)")
        else:
            logger.debug("[SUCCESS] 키워드 추출 완료: 0개")
        return keywords
    
    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        # 특수문자 제거 (한글, 영어, 숫자, 공백만 유지)
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        
        # 연속된 공백 정리
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _tokenize_text(self, text: str) -> List[str]:
        """텍스트를 단어로 분할 (복합명사 고려)"""
        words = []
        
        # 1. 복합명사 추출 (공백이 있는 복합명사도 포함)
        compound_words = self._extract_compound_nouns(text)
        words.extend(compound_words)
        
        # 2. 일반 단어 토큰화
        korean_pattern = r'[가-힣]+|[a-zA-Z]+|\d+'
        tokens = re.findall(korean_pattern, text)
        
        for token in tokens:
            # 기본 길이 필터링 (2글자 이상)
            if len(token) >= 2:
                words.append(token.lower())
        
        return words
    
    def _extract_compound_nouns(self, text: str) -> List[str]:
        """복합명사 추출"""
        compound_words = []
        
        # 정수처리 도메인 특화 복합명사 패턴
        compound_patterns = [
            # 공정 관련 (공백 포함/미포함 모두)
            r'착수\s*공정',
            r'약품\s*공정',
            r'혼화\s*응집\s*공정',
            r'혼화\s*공정',
            r'응집\s*공정',
            r'침전\s*공정',
            r'여과\s*공정',
            r'소독\s*공정',
            r'슬러지\s*처리\s*공정',
            r'슬러지\s*공정',
            
            # 시스템 관련
            r'ems\s*시스템',
            r'pms\s*시스템',
            r'스마트\s*정수장',
            r'ai\s*모델',
            r'머신러닝\s*모델',
            r'딥러닝\s*모델',
            
            # 장비/설비 관련
            r'교반기\s*회전속도',
            r'펌프\s*세부\s*현황',
            r'밸브\s*개도율',
            r'수위\s*목표값',
            r'잔류염소\s*농도',
            r'원수\s*탁도',
            r'응집제\s*주입률',
            r'슬러지\s*발생량',
            
            # 성능 지표 관련
            r'성능\s*지표',
            r'정확도\s*지표',
            r'효율\s*지표',
            r'mae\s*평균\s*절대\s*오차',
            r'mse\s*평균\s*제곱\s*오차',
            r'rmse\s*평균\s*제곱근\s*오차',
            r'r²\s*결정계수',
            r'r2\s*결정계수',
        ]
        
        for pattern in compound_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # 공백 제거하여 표준화
                normalized = re.sub(r'\s+', '', match)
                if len(normalized) >= 2:
                    compound_words.append(normalized.lower())
        
        return compound_words
    
    def _is_valid_keyword(self, word: str) -> bool:
        """유효한 키워드인지 확인 (개선된 버전)"""
        # 불용어 제거
        if word in self.stop_words:
            return False
        
        # 너무 짧은 단어 제거
        if len(word) < 2:
            return False
        
        # 숫자만으로 구성된 단어 제거
        if word.isdigit():
            return False
        
        # 영어 단어는 3글자 이상만 유효
        if word.isascii() and len(word) < 3:
            return False
        
        # 너무 긴 단어 제거 (20글자 이상)
        if len(word) > 20:
            return False
        
        # 반복 문자 패턴 제거 (예: "aaaa", "1111")
        if len(set(word)) == 1 and len(word) > 2:
            return False
        
        # 특수 패턴 제거 (예: "abcabc", "123123")
        if len(word) >= 4 and word[:len(word)//2] == word[len(word)//2:]:
            return False
        
        return True
    
    def _select_important_keywords(self, keywords: List[str], original_text: str) -> List[str]:
        """
        중요도 기반 키워드 선택 (최대 30개)
        
        Args:
            keywords: 후보 키워드 리스트
            original_text: 원본 텍스트
            
        Returns:
            선택된 중요 키워드 리스트
        """
        if not keywords:
            return []
        
        # 키워드 빈도 계산
        keyword_freq = Counter(keywords)
        
        # 중요도 점수 계산
        keyword_scores = {}
        for keyword, freq in keyword_freq.items():
            score = 0
            
            # 1. 빈도 점수 (0-40점)
            score += min(freq * 2, 40)
            
            # 2. 길이 점수 (0-20점) - 적당한 길이의 단어 선호
            if 3 <= len(keyword) <= 6:
                score += 20
            elif len(keyword) == 2 or len(keyword) == 7:
                score += 15
            elif len(keyword) == 8:
                score += 10
            
            # 3. 전문용어 점수 (0-20점)
            technical_terms = ['시스템', '모델', '데이터', '공정', '처리', '운영', '제어', '분석', '예측', '알고리즘', 
                             '성능', '효율', '최적화', '관리', '모니터링', '설정', '구성', '기능', '서비스', '플랫폼']
            if keyword in technical_terms:
                score += 20
            
            # 4. 도메인 특화 용어 점수 (0-20점)
            domain_terms = ['정수장', '정수', '원수', '유입', '유출', '수위', '탁도', '염소', '응집', '침전', 
                           '소독', '밸브', '펌프', '센서', '알람', '스마트', 'ai', '인공지능', '머신러닝']
            if keyword in domain_terms:
                score += 20
            
            keyword_scores[keyword] = score
        
        # 점수 순으로 정렬하여 상위 30개 선택 (20개에서 30개로 증가)
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        selected_keywords = [kw for kw, score in sorted_keywords[:30] if score > 0]
        
        return selected_keywords
    
    def get_frequent_keywords(self, min_frequency: int = None) -> List[str]:
        """
        자주 사용된 키워드 반환
        
        Args:
            min_frequency: 최소 빈도수 (기본값: cache_threshold)
            
        Returns:
            자주 사용된 키워드 리스트
        """
        if min_frequency is None:
            min_frequency = self.cache_threshold
        
        frequent_keywords = []
        for word, count in self.keyword_cache.items():
            if count >= min_frequency and word not in self.extracted_keywords:
                frequent_keywords.append(word)
        
        # 빈도수 순으로 정렬
        frequent_keywords.sort(key=lambda x: self.keyword_cache[x], reverse=True)
        
        return frequent_keywords
    
    def add_keywords_to_pipeline(self, pipeline_config_path: str = "config/pipelines/pdf_pipeline.json"):
        """
        추출된 키워드를 파이프라인 설정에 추가
        
        Args:
            pipeline_config_path: 파이프라인 설정 파일 경로
        """
        try:
            # 자주 사용된 키워드 가져오기
            frequent_keywords = self.get_frequent_keywords()
            
            if not frequent_keywords:
                logger.info("추가할 키워드가 없습니다.")
                return
            
            # 기존 설정 파일 읽기
            config_path = Path(pipeline_config_path)
            if not config_path.exists():
                logger.warning(f"파이프라인 설정 파일이 존재하지 않습니다: {config_path}")
                return
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 기존 키워드 목록
            existing_keywords = set(config.get("keywords", []))
            
            # 새로운 키워드 추가
            new_keywords = []
            for keyword in frequent_keywords:
                if keyword not in existing_keywords:
                    new_keywords.append(keyword)
                    existing_keywords.add(keyword)
            
            if new_keywords:
                # 설정 파일 업데이트
                config["keywords"] = list(existing_keywords)
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                
                # 추출된 키워드로 마킹
                self.extracted_keywords.update(new_keywords)
                
                # 파이프라인 추가 로그도 요약
                preview = new_keywords[:10]
                more = max(0, len(new_keywords) - 10)
                logger.info(f"파이프라인 설정에 {len(new_keywords)}개 키워드 추가: {preview}{' +'+str(more)+'개' if more else ''}")
            else:
                logger.info("추가할 새로운 키워드가 없습니다.")
                
        except Exception as e:
            logger.error(f"파이프라인 설정 업데이트 실패: {e}")
    
    def get_cache_stats(self) -> Dict:
        """캐시 통계 정보 반환"""
        return {
            "total_keywords": len(self.keyword_cache),
            "frequent_keywords": len(self.get_frequent_keywords()),
            "extracted_keywords": len(self.extracted_keywords),
            "cache_threshold": self.cache_threshold,
            "top_keywords": dict(self.keyword_cache.most_common(10))
        }
    
    def clear_cache(self):
        """캐시 초기화"""
        self.keyword_cache.clear()
        self.extracted_keywords.clear()
        logger.info("키워드 캐시 초기화 완료")
    
    def save_cache_to_file(self, cache_file: str = "data/pdf_keyword_cache.json"):
        """캐시를 파일로 저장 (선택사항)"""
        try:
            cache_data = {
                "keyword_cache": dict(self.keyword_cache),
                "extracted_keywords": list(self.extracted_keywords),
                "cache_threshold": self.cache_threshold
            }
            
            cache_path = Path(cache_file)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"키워드 캐시 저장 완료: {cache_path}")
            
        except Exception as e:
            logger.error(f"캐시 저장 실패: {e}")
    
    def load_cache_from_file(self, cache_file: str = "data/pdf_keyword_cache.json"):
        """파일에서 캐시 로드 (선택사항)"""
        try:
            cache_path = Path(cache_file)
            if not cache_path.exists():
                logger.info("캐시 파일이 존재하지 않습니다.")
                return
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            self.keyword_cache = Counter(cache_data.get("keyword_cache", {}))
            self.extracted_keywords = set(cache_data.get("extracted_keywords", []))
            self.cache_threshold = cache_data.get("cache_threshold", self.cache_threshold)
            
            logger.info(f"키워드 캐시 로드 완료: {len(self.keyword_cache)}개 키워드")
            
        except Exception as e:
            logger.error(f"캐시 로드 실패: {e}")
