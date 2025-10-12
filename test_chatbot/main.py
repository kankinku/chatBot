"""
RAG 챗봇 메인 실행 파일
"""
import os
import sys
from pdf_to_vectordb import build_vectordb
from rag_query import RAGSystem


def main():
    print("=" * 80)
    print("RAG 챗봇 시스템")
    print("=" * 80)
    print()
    
    # 벡터 DB 존재 여부 확인
    if not os.path.exists("./vectordb"):
        print("벡터 DB가 존재하지 않습니다. PDF 파일을 처리하여 벡터 DB를 생성합니다.\n")
        
        # data 폴더 확인
        if not os.path.exists("./data"):
            print("오류: data 폴더가 존재하지 않습니다.")
            print("data 폴더를 생성하고 PDF 파일을 추가한 후 다시 실행하세요.")
            return
        
        # PDF 파일 확인
        pdf_files = [f for f in os.listdir("./data") if f.endswith('.pdf')]
        if not pdf_files:
            print("오류: data 폴더에 PDF 파일이 없습니다.")
            print("data 폴더에 PDF 파일을 추가한 후 다시 실행하세요.")
            return
        
        # 벡터 DB 구축
        build_vectordb()
        print()
    else:
        print("기존 벡터 DB를 사용합니다.\n")
    
    # RAG 시스템 초기화
    try:
        rag = RAGSystem()
    except Exception as e:
        print(f"오류: RAG 시스템 초기화 실패 - {str(e)}")
        return
    
    # 대화형 루프
    print("=" * 80)
    print("질문을 입력하세요 (종료하려면 'quit' 또는 'exit' 입력)")
    print("=" * 80)
    print()
    
    while True:
        try:
            user_input = input("\n질문: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '종료']:
                print("챗봇을 종료합니다.")
                break
            
            if not user_input:
                continue
            
            print()
            answer, docs, metas = rag.query(user_input, top_k=3)
            
            print("\n" + "=" * 80)
            print("답변:")
            print("-" * 80)
            print(answer)
            print("=" * 80)
            
        except KeyboardInterrupt:
            print("\n\n챗봇을 종료합니다.")
            break
        except Exception as e:
            print(f"\n오류 발생: {str(e)}")


if __name__ == "__main__":
    main()

