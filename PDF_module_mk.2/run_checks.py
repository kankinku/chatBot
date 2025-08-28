#!/usr/bin/env python3
"""
단일 점검 실행기 (루트 위치)

사용 예시:
  python run_checks.py simple
  python run_checks.py data
  python run_checks.py qa --iterations 2 --save
  python run_checks.py keyword
  python run_checks.py keyword-adv
  python run_checks.py keyword-real

프로젝트 루트에서 실행하세요.
"""

import argparse
import importlib
import sys
from pathlib import Path


def _ensure_project_root_on_sys_path() -> None:
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def run_simple() -> None:
    module = importlib.import_module('checks.simple_test')
    if hasattr(module, 'test_basic_functionality'):
        module.test_basic_functionality()
    if hasattr(module, 'test_pdf_processing'):
        module.test_pdf_processing()


def run_data() -> None:
    module = importlib.import_module('checks.test_data')
    if hasattr(module, 'test_database') and hasattr(module, 'test_vector_store') and hasattr(module, 'test_full_qa'):
        db, _ = module.test_database()
        module.test_vector_store(db)
        module.test_full_qa()


def run_qa(iterations: int = 1, save: bool = False) -> None:
    module = importlib.import_module('checks.test_qa_script')
    if hasattr(module, 'PDFQATestSuite'):
        suite = module.PDFQATestSuite()
        results = suite.run_full_test_suite(iterations=iterations)
        suite.print_summary()
        if save:
            import json
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'test_results_{timestamp}.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 테스트 결과 저장: {filename}")


def run_keyword_basic() -> None:
    module = importlib.import_module('checks.test_keyword_enhancement')
    if hasattr(module, 'main'):
        module.main()


def run_keyword_advanced() -> None:
    module = importlib.import_module('checks.test_enhanced_keyword_recognition')
    if hasattr(module, 'main'):
        module.main()


def run_keyword_real() -> None:
    module = importlib.import_module('checks.test_real_pdf_keywords')
    if hasattr(module, 'main'):
        module.main()


def main() -> None:
    _ensure_project_root_on_sys_path()

    parser = argparse.ArgumentParser(description='점검/테스트 오케스트레이터')
    sub = parser.add_subparsers(dest='cmd', required=True)

    sub.add_parser('simple', help='기본 기능 및 PDF 처리 점검')
    sub.add_parser('data', help='DB/벡터저장소/전체 QA 점검')

    qa_p = sub.add_parser('qa', help='실제 질문 세트 기반 품질 점검')
    qa_p.add_argument('--iterations', '-i', type=int, default=1)
    qa_p.add_argument('--save', '-s', action='store_true')

    sub.add_parser('keyword', help='키워드 향상/검색/생성기 개선 포인트 점검')
    sub.add_parser('keyword-adv', help='도메인별 향상 효과/통계/성능 비교(실전형)')
    sub.add_parser('keyword-real', help='실제 PDF 기반 인식률/품질/성능 테스트')

    args = parser.parse_args()

    if args.cmd == 'simple':
        run_simple()
    elif args.cmd == 'data':
        run_data()
    elif args.cmd == 'qa':
        run_qa(iterations=args.iterations, save=args.save)
    elif args.cmd == 'keyword':
        run_keyword_basic()
    elif args.cmd == 'keyword-adv':
        run_keyword_advanced()
    elif args.cmd == 'keyword-real':
        run_keyword_real()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
