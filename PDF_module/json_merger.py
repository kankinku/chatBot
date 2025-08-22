import os
import json
from pathlib import Path

def merge_json_files(input_dir, output_file):
    """
    지정된 디렉토리의 모든 JSON 파일을 하나의 파일로 통합
    
    Args:
        input_dir (str): JSON 파일들이 있는 디렉토리 경로
        output_file (str): 통합된 JSON을 저장할 파일 경로
    """
    try:
        # 결과를 저장할 딕셔너리
        merged_data = {}
        
        # 입력 디렉토리가 존재하는지 확인
        if not os.path.exists(input_dir):
            print(f"디렉토리를 찾을 수 없습니다: {input_dir}")
            return
        
        # 디렉토리 내의 모든 JSON 파일 처리
        json_files = list(Path(input_dir).glob('*.json'))
        if not json_files:
            print(f"JSON 파일을 찾을 수 없습니다: {input_dir}")
            return
            
        print(f"발견된 JSON 파일 수: {len(json_files)}")
        
        # 각 JSON 파일 처리
        for json_path in json_files:
            try:
                # 파일명에서 확장자를 제외한 부분을 키로 사용
                key = json_path.stem
                print(f"처리 중: {json_path.name}")
                
                # JSON 파일 읽기
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 데이터를 통합 딕셔너리에 추가
                merged_data[key] = data
                
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류 ({json_path.name}): {str(e)}")
                continue
            except Exception as e:
                print(f"파일 처리 중 오류 발생 ({json_path.name}): {str(e)}")
                continue
        
        # 결과가 비어있는지 확인
        if not merged_data:
            print("통합할 데이터가 없습니다.")
            return
            
        # 통합된 데이터를 파일로 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=4)
            
        print(f"JSON 파일이 성공적으로 통합되었습니다: {output_file}")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    # 실행 예시
    input_directory = "./json"  # JSON 파일들이 있는 디렉토리
    output_file = "merged_manual.json"  # 통합된 JSON을 저장할 파일
    
    # JSON 파일 통합 실행
    merge_json_files(input_directory, output_file)
