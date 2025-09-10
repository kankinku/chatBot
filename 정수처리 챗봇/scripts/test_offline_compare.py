import sys
from pathlib import Path
import json
import random
import shutil

# Ensure project root on sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# chroma_db 디렉토리 강제 삭제
chroma_db_path = _ROOT / "chroma_db"
if chroma_db_path.exists():
    print(f"chroma_db 디렉토리를 삭제합니다: {chroma_db_path}")
    shutil.rmtree(chroma_db_path)
    print("chroma_db 디렉토리가 성공적으로 삭제되었습니다.")
else:
    print("chroma_db 디렉토리가 존재하지 않습니다.")


def main(sample_k: int = 15):
    from main import LocalChatbot

    qa_path = Path("data/tests/gosan_qa.json")
    if not qa_path.exists():
        # create from script if missing
        try:
            from scripts.gosan_qa_dataset import ensure_json
            qa_path = ensure_json(qa_path)
        except Exception:
            print("ERROR: gosan_qa.json not found and could not be generated.")
            sys.exit(2)

    with qa_path.open("r", encoding="utf-8") as f:
        qa_items = json.load(f)

    if not isinstance(qa_items, list) or len(qa_items) == 0:
        print("ERROR: gosan_qa.json has no items")
        sys.exit(2)

    # 샘플 추출 (매번 다른 랜덤 결과를 위해 시드 제거)
    samples = random.sample(qa_items, k=min(sample_k, len(qa_items)))

    # 운영 파이프라인과 동일한 흐름을 위해 LocalChatbot 사용
    chatbot = LocalChatbot()

    results = []
    for idx, item in enumerate(samples, start=1):
        q = item.get("question") or ""
        expected = item.get("answer") or ""
        resp = chatbot.process_question(q)
        generated = resp.get("answer", "")
        results.append({
            "no": idx,
            "question": q,
            "expected": expected,
            "generated": generated
        })

    # 사용자 수동 비교용 출력
    out_path = Path("data/tests/gosan_compare_result.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote compare set ({len(results)} items) to {out_path}")
    print("수동 비교: 'expected' vs 'generated' 확인해주세요.")


if __name__ == "__main__":
    main()


