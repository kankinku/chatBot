"""
이동 처리 모듈

지도 이동 요청을 처리하고 프론트엔드로 이동 정보를 전달
"""

import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
import json
import re

logger = logging.getLogger(__name__)

@dataclass
class MovementRequest:
    """이동 요청 데이터"""
    destination: str
    original_question: str
    confidence: float
    metadata: Optional[Dict] = None

@dataclass
class MovementResponse:
    """이동 응답 데이터"""
    success: bool
    message: str
    destination: str
    coordinates: Optional[Dict[str, float]] = None
    metadata: Optional[Dict] = None

class MovementHandler:
    """이동 처리기"""
    
    def __init__(self):
        """이동 처리기 초기화"""
        # 주요 지역별 좌표 정보 (세종시 동/읍)
        self.location_coordinates = {
            "한솔동": {"lat": 36.4800, "lng": 127.2889},
            "새롬동": {"lat": 36.4867, "lng": 127.2811},
            "도담동": {"lat": 36.4767, "lng": 127.2811},
            "아름동": {"lat": 36.4833, "lng": 127.2750},
            "종촌동": {"lat": 36.4889, "lng": 127.2750},
            "고운동": {"lat": 36.4756, "lng": 127.2689},
            "소담동": {"lat": 36.4811, "lng": 127.2689},
            "보람동": {"lat": 36.4867, "lng": 127.2628},
            "대평동": {"lat": 36.4922, "lng": 127.2628},
            "다정동": {"lat": 36.4978, "lng": 127.2567},
            "어진동": {"lat": 36.5033, "lng": 127.2567},
            "조치원읍": {"lat": 36.6033, "lng": 127.2989},
            "연기면": {"lat": 36.5589, "lng": 127.3250},
            "연동면": {"lat": 36.5589, "lng": 127.2989},
            "부강면": {"lat": 36.5144, "lng": 127.3250},
            "금남면": {"lat": 36.5144, "lng": 127.2989},
            "장군면": {"lat": 36.4699, "lng": 127.3250},
            "연서면": {"lat": 36.4699, "lng": 127.2989},
            "전의면": {"lat": 36.4254, "lng": 127.3250},
            "전동면": {"lat": 36.4254, "lng": 127.2989},
            "소정면": {"lat": 36.3809, "lng": 127.3250},
            "반곡동": {"lat": 36.5089, "lng": 127.2750},
            "가람동": {"lat": 36.5144, "lng": 127.2689},
            "한별동": {"lat": 36.5199, "lng": 127.2628},
            "새아름동": {"lat": 36.5254, "lng": 127.2567}
        }
        
        # 교차로 이름 매핑 (세종특별자치시 형식 -> 지역명)
        self.intersection_mapping = {
            # 조치원읍 교차로들
            "세종특별자치시조치원읍": "조치원읍",
            "세종특별자치시조치원읍(1)": "조치원읍",
            "세종특별자치시조치원읍(2)": "조치원읍",
            "세종특별자치시조치원읍(3)": "조치원읍",
            "세종특별자치시조치원읍(4)": "조치원읍",
            "세종특별자치시조치원읍(5)": "조치원읍",
            "세종특별자치시조치원읍(6)": "조치원읍",
            "세종특별자치시조치원읍(7)": "조치원읍",
            "세종특별자치시조치원읍(8)": "조치원읍",
            "세종특별자치시조치원읍(9)": "조치원읍",
            "세종특별자치시조치원읍(10)": "조치원읍",
            
            # 연기면 교차로들
            "세종특별자치시연기면": "연기면",
            "세종특별자치시연기면(1)": "연기면",
            "세종특별자치시연기면(2)": "연기면",
            "세종특별자치시연기면(3)": "연기면",
            "세종특별자치시연기면(4)": "연기면",
            "세종특별자치시연기면(5)": "연기면",
            
            # 연동면 교차로들
            "세종특별자치시연동면": "연동면",
            "세종특별자치시연동면(1)": "연동면",
            "세종특별자치시연동면(2)": "연동면",
            "세종특별자치시연동면(3)": "연동면",
            "세종특별자치시연동면(4)": "연동면",
            "세종특별자치시연동면(5)": "연동면",
            "세종특별자치시연동면(6)": "연동면",
            "세종특별자치시연동면(7)": "연동면",
            "세종특별자치시연동면(8)": "연동면",
            "세종특별자치시연동면(9)": "연동면",
            "세종특별자치시연동면(10)": "연동면",
            
            # 부강면 교차로들
            "세종특별자치시부강면": "부강면",
            "세종특별자치시부강면(1)": "부강면",
            "세종특별자치시부강면(2)": "부강면",
            "세종특별자치시부강면(3)": "부강면",
            "세종특별자치시부강면(4)": "부강면",
            "세종특별자치시부강면(5)": "부강면",
            "세종특별자치시부강면(6)": "부강면",
            "세종특별자치시부강면(7)": "부강면",
            "세종특별자치시부강면(8)": "부강면",
            "세종특별자치시부강면(9)": "부강면",
            "세종특별자치시부강면(10)": "부강면",
            
            # 금남면 교차로들
            "세종특별자치시금남면": "금남면",
            "세종특별자치시금남면(1)": "금남면",
            "세종특별자치시금남면(2)": "금남면",
            "세종특별자치시금남면(3)": "금남면",
            "세종특별자치시금남면(4)": "금남면",
            "세종특별자치시금남면(5)": "금남면",
            "세종특별자치시금남면(6)": "금남면",
            "세종특별자치시금남면(7)": "금남면",
            "세종특별자치시금남면(8)": "금남면",
            "세종특별자치시금남면(9)": "금남면",
            "세종특별자치시금남면(10)": "금남면",
            
            # 장군면 교차로들
            "세종특별자치시장군면": "장군면",
            "세종특별자치시장군면(1)": "장군면",
            "세종특별자치시장군면(2)": "장군면",
            "세종특별자치시장군면(3)": "장군면",
            "세종특별자치시장군면(4)": "장군면",
            "세종특별자치시장군면(5)": "장군면",
            "세종특별자치시장군면(6)": "장군면",
            "세종특별자치시장군면(7)": "장군면",
            "세종특별자치시장군면(8)": "장군면",
            "세종특별자치시장군면(9)": "장군면",
            "세종특별자치시장군면(10)": "장군면",
            
            # 연서면 교차로들
            "세종특별자치시연서면": "연서면",
            "세종특별자치시연서면(1)": "연서면",
            "세종특별자치시연서면(2)": "연서면",
            "세종특별자치시연서면(3)": "연서면",
            "세종특별자치시연서면(4)": "연서면",
            "세종특별자치시연서면(5)": "연서면",
            "세종특별자치시연서면(6)": "연서면",
            "세종특별자치시연서면(7)": "연서면",
            "세종특별자치시연서면(8)": "연서면",
            "세종특별자치시연서면(9)": "연서면",
            "세종특별자치시연서면(10)": "연서면",
            
            # 전의면 교차로들
            "세종특별자치시전의면": "전의면",
            "세종특별자치시전의면(1)": "전의면",
            "세종특별자치시전의면(2)": "전의면",
            "세종특별자치시전의면(3)": "전의면",
            "세종특별자치시전의면(4)": "전의면",
            "세종특별자치시전의면(5)": "전의면",
            "세종특별자치시전의면(6)": "전의면",
            "세종특별자치시전의면(7)": "전의면",
            "세종특별자치시전의면(8)": "전의면",
            "세종특별자치시전의면(9)": "전의면",
            "세종특별자치시전의면(10)": "전의면",
            
            # 전동면 교차로들
            "세종특별자치시전동면": "전동면",
            "세종특별자치시전동면(1)": "전동면",
            "세종특별자치시전동면(2)": "전동면",
            "세종특별자치시전동면(3)": "전동면",
            "세종특별자치시전동면(4)": "전동면",
            "세종특별자치시전동면(5)": "전동면",
            "세종특별자치시전동면(6)": "전동면",
            "세종특별자치시전동면(7)": "전동면",
            "세종특별자치시전동면(8)": "전동면",
            "세종특별자치시전동면(9)": "전동면",
            "세종특별자치시전동면(10)": "전동면",
            
            # 소정면 교차로들
            "세종특별자치시소정면": "소정면",
            "세종특별자치시소정면(1)": "소정면",
            "세종특별자치시소정면(2)": "소정면",
            "세종특별자치시소정면(3)": "소정면",
            "세종특별자치시소정면(4)": "소정면",
            "세종특별자치시소정면(5)": "소정면",
            "세종특별자치시소정면(6)": "소정면",
            "세종특별자치시소정면(7)": "소정면",
            "세종특별자치시소정면(8)": "소정면",
            "세종특별자치시소정면(9)": "소정면",
            "세종특별자치시소정면(10)": "소정면",
            
            # 동 지역 교차로들 (한솔동, 새롬동, 도담동 등)
            "세종특별자치시한솔동": "한솔동",
            "세종특별자치시새롬동": "새롬동",
            "세종특별자치시도담동": "도담동",
            "세종특별자치시아름동": "아름동",
            "세종특별자치시종촌동": "종촌동",
            "세종특별자치시고운동": "고운동",
            "세종특별자치시소담동": "소담동",
            "세종특별자치시보람동": "보람동",
            "세종특별자치시대평동": "대평동",
            "세종특별자치시다정동": "다정동",
            "세종특별자치시어진동": "어진동",
            "세종특별자치시반곡동": "반곡동",
            "세종특별자치시가람동": "가람동",
            "세종특별자치시한별동": "한별동",
            "세종특별자치시새아름동": "새아름동"
        }
        
        # 지역별 교차로 그룹화
        self.region_intersections = self._group_intersections_by_region()
        
        logger.info("이동 처리기 초기화 완료")
    
    def _group_intersections_by_region(self) -> Dict[str, List[str]]:
        """교차로를 지역별로 그룹화"""
        region_groups = {}
        
        for intersection_name, region in self.intersection_mapping.items():
            if region not in region_groups:
                region_groups[region] = []
            region_groups[region].append(intersection_name)
        
        return region_groups
    
    def process_movement_request(self, question: str, destination: str, confidence: float) -> MovementResponse:
        """
        이동 요청 처리
        
        Args:
            question: 원본 질문
            destination: 추출된 목적지
            confidence: 분류 신뢰도
            
        Returns:
            이동 응답
        """
        try:
            # 목적지 정규화
            normalized_destination = self._normalize_destination(destination)
            
            # 좌표 정보 확인
            coordinates = self.location_coordinates.get(normalized_destination)
            
            if coordinates:
                # 성공적인 이동 응답
                response = MovementResponse(
                    success=True,
                    message=f"{normalized_destination}로 이동 완료했습니다.",
                    destination=normalized_destination,
                    coordinates=coordinates,
                    metadata={
                        "original_question": question,
                        "confidence": confidence,
                        "normalized_destination": normalized_destination
                    }
                )
                
                logger.info(f"이동 요청 처리 완료: {normalized_destination} -> {coordinates}")
                return response
            else:
                # 좌표 정보가 없는 경우
                response = MovementResponse(
                    success=False,
                    message=f"{normalized_destination}의 위치 정보를 찾을 수 없습니다.",
                    destination=normalized_destination,
                    metadata={
                        "original_question": question,
                        "confidence": confidence,
                        "error": "location_not_found"
                    }
                )
                
                logger.warning(f"위치 정보 없음: {normalized_destination}")
                return response
                
        except Exception as e:
            logger.error(f"이동 요청 처리 실패: {e}")
            return MovementResponse(
                success=False,
                message="이동 요청 처리 중 오류가 발생했습니다.",
                destination=destination,
                metadata={
                    "original_question": question,
                    "confidence": confidence,
                    "error": str(e)
                }
            )
    
    def _normalize_destination(self, destination: str) -> str:
        """목적지 정규화"""
        # 공백 제거
        normalized = destination.strip()
        
        # 불필요한 접미사 제거
        suffixes_to_remove = ['으', '로', '으로', '에', '에서']
        for suffix in suffixes_to_remove:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
        
        # 세종시 동/읍 단위 정규화
        if normalized.endswith('동') or normalized.endswith('읍') or normalized.endswith('면'):
            return normalized
        elif normalized in ['한솔', '새롬', '도담', '아름', '종촌', '고운', '소담', '보람', '대평', '다정', '어진', '반곡', '가람', '한별', '새아름']:
            return normalized + '동'
        elif normalized in ['조치원']:
            return normalized + '읍'
        elif normalized in ['연기', '연동', '부강', '금남', '장군', '연서', '전의', '전동', '소정']:
            return normalized + '면'
        
        return normalized
    
    def get_intersection_by_region(self, region_name: str) -> List[str]:
        """지역명으로 교차로 목록 조회"""
        normalized_region = self._normalize_destination(region_name)
        return self.region_intersections.get(normalized_region, [])
    
    def get_region_by_intersection(self, intersection_name: str) -> Optional[str]:
        """교차로명으로 지역 조회"""
        return self.intersection_mapping.get(intersection_name)
    
    def search_intersections(self, query: str) -> List[str]:
        """교차로 검색 (부분 일치)"""
        results = []
        query_lower = query.lower()
        
        for intersection_name in self.intersection_mapping.keys():
            if query_lower in intersection_name.lower():
                results.append(intersection_name)
        
        return results
    
    def get_available_locations(self) -> Dict[str, Dict[str, float]]:
        """사용 가능한 위치 목록 반환"""
        return self.location_coordinates.copy()
    
    def get_available_regions(self) -> List[str]:
        """사용 가능한 지역 목록 반환"""
        return list(self.location_coordinates.keys())
    
    def get_intersection_mapping(self) -> Dict[str, str]:
        """교차로 매핑 정보 반환"""
        return self.intersection_mapping.copy()
    
    def add_location(self, name: str, lat: float, lng: float):
        """새로운 위치 추가"""
        self.location_coordinates[name] = {"lat": lat, "lng": lng}
        logger.info(f"새로운 위치 추가: {name} ({lat}, {lng})")
    
    def add_intersection_mapping(self, intersection_name: str, region: str):
        """새로운 교차로 매핑 추가"""
        self.intersection_mapping[intersection_name] = region
        if region not in self.region_intersections:
            self.region_intersections[region] = []
        self.region_intersections[region].append(intersection_name)
        logger.info(f"새로운 교차로 매핑 추가: {intersection_name} -> {region}")
    
    def remove_location(self, name: str) -> bool:
        """위치 제거"""
        if name in self.location_coordinates:
            del self.location_coordinates[name]
            logger.info(f"위치 제거: {name}")
            return True
        return False

if __name__ == "__main__":
    # 테스트 코드
    handler = MovementHandler()
    
    test_cases = [
        "조치원읍으로 이동해주세요",
        "연동면 지도로 가주세요",
        "한솔동으로 가주세요",
        "세종특별자치시조치원읍(1) 교차로 정보",
        "존재하지 않는 곳으로 이동"
    ]
    
    for test_question in test_cases:
        # 목적지 추출 (실제로는 QuestionAnalyzer에서 수행)
        destination = test_question.replace("로 이동해주세요", "").replace("지도로 가주세요", "").replace("로 가주세요", "").replace(" 교차로 정보", "").strip()
        
        result = handler.process_movement_request(test_question, destination, 0.8)
        print(f"\n질문: {test_question}")
        print(f"목적지: {destination}")
        print(f"결과: {result.message}")
        print(f"성공: {result.success}")
        if result.coordinates:
            print(f"좌표: {result.coordinates}")
        
        # 교차로 매핑 테스트
        if "세종특별자치시" in destination:
            region = handler.get_region_by_intersection(destination)
            if region:
                print(f"매핑된 지역: {region}")
                intersections = handler.get_intersection_by_region(region)
                print(f"해당 지역 교차로 수: {len(intersections)}")
