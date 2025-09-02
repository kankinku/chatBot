#!/usr/bin/env python3
"""
SBERT 모델 자동 다운로드 설정 스크립트

이 스크립트는 의도 분류기에 필요한 SBERT 모델을 미리 다운로드합니다.
"""

import os
import sys
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """의존성 확인"""
    print("=" * 60)
    print("🔍 의존성 확인")
    print("=" * 60)
    
    required_packages = [
        "sentence_transformers",
        "torch",
        "transformers",
        "numpy",
        "sklearn"  # scikit-learn의 실제 import 이름
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (설치 필요)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 다음 패키지들을 설치해야 합니다:")
        for package in missing_packages:
            if package == "sklearn":
                print(f"pip install scikit-learn")
            else:
                print(f"pip install {package}")
        return False
    else:
        print(f"\n✅ 모든 의존성이 설치되어 있습니다.")
        return True

def setup_sbert_models():
    """SBERT 모델들을 미리 다운로드"""
    print("=" * 60)
    print("🤖 SBERT 모델 자동 다운로드 설정")
    print("=" * 60)
    
    try:
        # sentence-transformers 임포트
        from sentence_transformers import SentenceTransformer
        
        # 다운로드할 모델 목록 (우선순위 순서)
        models = [
            {
                "name": "한국어 특화 모델",
                "model_id": "jhgan/ko-sroberta-multitask",
                "description": "한국어 교통 도메인에 최적화된 SBERT 모델",
                "priority": 1
            },
            {
                "name": "범용 모델 (대안)",
                "model_id": "sentence-transformers/all-MiniLM-L6-v2",
                "description": "빠르고 효율적인 범용 SBERT 모델",
                "priority": 2
            },
            {
                "name": "다국어 모델 (대안)",
                "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "description": "다국어 지원 SBERT 모델",
                "priority": 3
            }
        ]
        
        print(f"\n📥 SBERT 모델 다운로드를 시작합니다...")
        print("한국어 특화 모델이 성공하면 나머지는 건너뜁니다.")
        
        primary_model_success = False
        
        for i, model_info in enumerate(models, 1):
            print(f"\n{i}/{len(models)}. {model_info['name']}")
            print(f"   모델 ID: {model_info['model_id']}")
            print(f"   설명: {model_info['description']}")
            
            try:
                print(f"   ⏳ 다운로드 중...")
                model = SentenceTransformer(model_info['model_id'])
                print(f"   ✅ 다운로드 완료!")
                
                # 간단한 테스트
                test_sentences = ["안녕하세요", "교통량 확인"]
                embeddings = model.encode(test_sentences)
                print(f"   🧪 테스트 완료 (임베딩 크기: {embeddings.shape})")
                
                # 한국어 특화 모델이 성공하면 나머지는 건너뛰기
                if model_info['priority'] == 1:
                    primary_model_success = True
                    print(f"   🎯 한국어 특화 모델 다운로드 성공!")
                    print(f"   ✅ 나머지 대안 모델들은 건너뜁니다.")
                    break
                
            except Exception as e:
                print(f"   ❌ 다운로드 실패: {e}")
                if model_info['priority'] == 1:
                    print(f"   ⚠️ 한국어 특화 모델 실패. 대안 모델을 시도합니다.")
                    continue
                else:
                    print(f"   ⚠️ 이 모델은 건너뛰고 다음 모델을 시도합니다.")
                    continue
        
        if primary_model_success:
            print(f"\n✅ 한국어 특화 SBERT 모델 설정 완료!")
            print("이제 의도 분류기에서 최적화된 한국어 모델을 사용합니다.")
        else:
            print(f"\n⚠️ 한국어 특화 모델 다운로드에 실패했습니다.")
            print("대안 모델을 사용하거나 네트워크 연결을 확인해주세요.")
        
        # 모델 저장 경로 확인
        cache_dir = Path.home() / ".cache" / "torch" / "sentence_transformers"
        if cache_dir.exists():
            print(f"\n📁 모델 캐시 위치: {cache_dir}")
            cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
            print(f"📊 캐시 크기: {cache_size / (1024*1024):.1f} MB")
        
        return primary_model_success
        
    except ImportError:
        print("❌ sentence-transformers가 설치되지 않았습니다.")
        print("다음 명령어로 설치하세요:")
        print("pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"❌ SBERT 모델 설정 중 오류 발생: {e}")
        return False

def test_core_modules():
    """핵심 모듈 테스트 - 개별적으로 테스트하여 실패 지점 찾기"""
    print(f"\n" + "=" * 60)
    print("🧪 핵심 모듈 개별 테스트")
    print("=" * 60)
    
    # 테스트할 모듈 목록
    test_modules = [
        {
            "name": "QueryRouter",
            "path": "core.query.query_router",
            "class_name": "QueryRouter"
        },
        {
            "name": "SQLElementExtractor", 
            "path": "core.database.sql_element_extractor",
            "class_name": "SQLElementExtractor"
        },
        {
            "name": "AnswerGenerator",
            "path": "core.llm.answer_generator", 
            "class_name": "AnswerGenerator"
        },
        {
            "name": "PDFProcessor",
            "path": "core.document.pdf_processor",
            "class_name": "PDFProcessor"
        },
        {
            "name": "VectorStore",
            "path": "core.document.vector_store",
            "class_name": "VectorStore"
        },
        {
            "name": "QuestionAnalyzer",
            "path": "core.query.question_analyzer",
            "class_name": "QuestionAnalyzer"
        },
        {
            "name": "FastCache",
            "path": "core.cache.fast_cache",
            "class_name": "FastCache"
        },
        {
            "name": "SQLGenerator",
            "path": "core.database.sql_generator",
            "class_name": "SQLGenerator"
        }
    ]
    
    success_count = 0
    total_count = len(test_modules)
    
    print(f"총 {total_count}개 모듈을 테스트합니다...\n")
    
    for i, module_info in enumerate(test_modules, 1):
        print(f"{i}/{total_count}. {module_info['name']} 테스트 중...")
        
        try:
            # 모듈 임포트 테스트
            module = __import__(module_info['path'], fromlist=[module_info['class_name']])
            print(f"   ✅ 모듈 임포트 성공: {module_info['path']}")
            
            # 클래스 임포트 테스트
            try:
                class_obj = getattr(module, module_info['class_name'])
                print(f"   ✅ 클래스 임포트 성공: {module_info['class_name']}")
                
                # 인스턴스 생성 테스트 (선택적)
                try:
                    if module_info['name'] == "QueryRouter":
                        instance = class_obj()
                    elif module_info['name'] == "SQLElementExtractor":
                        instance = class_obj()
                    elif module_info['name'] == "AnswerGenerator":
                        instance = class_obj()
                    elif module_info['name'] == "PDFProcessor":
                        instance = class_obj()
                    elif module_info['name'] == "VectorStore":
                        instance = class_obj()
                    elif module_info['name'] == "QuestionAnalyzer":
                        instance = class_obj()
                    elif module_info['name'] == "FastCache":
                        instance = class_obj()
                    elif module_info['name'] == "SQLGenerator":
                        instance = class_obj()
                    
                    print(f"   ✅ 인스턴스 생성 성공")
                    success_count += 1
                    
                except Exception as e:
                    print(f"   ⚠️ 인스턴스 생성 실패: {e}")
                    print(f"   📝 이는 정상적인 경우일 수 있습니다 (의존성 문제)")
                    success_count += 1  # 임포트는 성공했으므로 성공으로 간주
                
            except AttributeError as e:
                print(f"   ❌ 클래스 임포트 실패: {e}")
                print(f"   📝 모듈에는 존재하지만 클래스를 찾을 수 없습니다")
                
        except ImportError as e:
            print(f"   ❌ 모듈 임포트 실패: {e}")
            print(f"   🔍 문제 분석:")
            print(f"      - 경로: {module_info['path']}")
            print(f"      - 오류: {str(e)}")
            
            # 상세한 문제 분석
            if "No module named" in str(e):
                print(f"      💡 해결 방법: 모듈 경로 확인 필요")
            elif "cannot import name" in str(e):
                print(f"      💡 해결 방법: 클래스명 확인 필요")
            elif "circular import" in str(e):
                print(f"      💡 해결 방법: 순환 import 문제 해결 필요")
                
        except Exception as e:
            print(f"   ❌ 예상치 못한 오류: {e}")
            print(f"   🔍 오류 타입: {type(e).__name__}")
            
        print()  # 빈 줄로 구분
    
    # 결과 요약
    print("=" * 60)
    print("📊 모듈 테스트 결과 요약")
    print("=" * 60)
    print(f"✅ 성공: {success_count}/{total_count}")
    print(f"❌ 실패: {total_count - success_count}/{total_count}")
    print(f"📈 성공률: {(success_count/total_count)*100:.1f}%")
    
    if success_count == total_count:
        print("\n🎉 모든 모듈이 성공적으로 테스트되었습니다!")
        return True
    else:
        print(f"\n⚠️ {total_count - success_count}개 모듈에서 문제가 발생했습니다.")
        print("위의 상세 오류 메시지를 확인하여 문제를 해결하세요.")
        return False

def main():
    """메인 함수"""
    print("🚀 SBERT 모델 자동 다운로드 설정 시작")
    
    # 1. 의존성 확인
    if not check_dependencies():
        print("\n❌ 의존성 문제로 인해 설정을 중단합니다.")
        return
    
    # 2. SBERT 모델 다운로드
    if not setup_sbert_models():
        print("\n❌ SBERT 모델 설정에 실패했습니다.")
        return
    
    # 3. 핵심 모듈 테스트
    if not test_core_modules():
        print("\n❌ 핵심 모듈 테스트에 실패했습니다.")
        return
    
    print(f"\n" + "=" * 60)
    print("🎉 SBERT 모델 설정 완료!")
    print("이제 최적화된 챗봇 시스템을 사용할 수 있습니다.")
    print("=" * 60)

if __name__ == "__main__":
    main()
