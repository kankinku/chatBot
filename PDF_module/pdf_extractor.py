import os
from PyPDF2 import PdfReader

def create_target_folder():
    """
    target 폴더가 없으면 생성하는 함수
    """
    target_folder = "target"
    output_folder = "output"
    
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"'{target_folder}' 폴더가 생성되었습니다.")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"'{output_folder}' 폴더가 생성되었습니다.")
    
    return target_folder, output_folder

def extract_text_from_pdf(pdf_path):
    """
    PDF 파일에서 텍스트를 추출하는 함수
    
    Args:
        pdf_path (str): PDF 파일 경로
    
    Returns:
        str: 추출된 텍스트
    """
    try:
        # PDF 파일이 존재하는지 확인
        if not os.path.exists(pdf_path):
            return f"오류: PDF 파일을 찾을 수 없습니다: {pdf_path}"
        
        # PDF 파일 열기
        reader = PdfReader(pdf_path)
        text = ""
        
        # 모든 페이지의 텍스트 추출
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        return text
    
    except Exception as e:
        return f"오류 발생: {str(e)}"

def save_text_to_file(text, output_path):
    """
    추출된 텍스트를 파일로 저장하는 함수
    
    Args:
        text (str): 저장할 텍스트
        output_path (str): 저장할 파일 경로
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"텍스트가 성공적으로 저장되었습니다: {output_path}")
    except Exception as e:
        print(f"파일 저장 중 오류 발생: {str(e)}")

def process_pdf_files():
    """
    target 폴더 내의 모든 PDF 파일을 처리하는 함수
    """
    target_folder, output_folder = create_target_folder()
    
    # target 폴더 내의 모든 파일 검사
    pdf_files = [f for f in os.listdir(target_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"'{target_folder}' 폴더에 PDF 파일이 없습니다.")
        return
    
    print(f"\n총 {len(pdf_files)}개의 PDF 파일을 처리합니다...\n")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(target_folder, pdf_file)
        print(f"\n{pdf_file} 처리 중...")
        
        # 텍스트 추출
        extracted_text = extract_text_from_pdf(pdf_path)
        
        if not extracted_text.startswith("오류"):
            # 결과를 저장할 파일 이름 생성 (원본 파일명.txt)
            output_filename = os.path.splitext(pdf_file)[0] + ".txt"
            output_path = os.path.join(output_folder, output_filename)
            save_text_to_file(extracted_text, output_path)
            
            # 추출된 텍스트의 일부 미리보기 출력
            preview_length = min(200, len(extracted_text))
            print("\n추출된 텍스트 미리보기:")
            print("-" * 50)
            print(extracted_text[:preview_length] + "...")
            print("-" * 50)
        else:
            print(extracted_text)

def main():
    process_pdf_files()

if __name__ == "__main__":
    main()
