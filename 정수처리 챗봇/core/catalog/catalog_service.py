"""
데이터 카탈로그/승인 워크플로우 스켈레톤
"""

from typing import Dict, Any, List
from pathlib import Path
import json

CATALOG_PATH = Path("data/catalog.json")
REQUESTS_PATH = Path("data/catalog_requests.json")


def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def load_catalog() -> List[Dict[str, Any]]:
    if not CATALOG_PATH.exists():
        return []
    with CATALOG_PATH.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return []


def save_catalog(items: List[Dict[str, Any]]):
    _ensure_parent(CATALOG_PATH)
    with CATALOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def submit_change_request(item: Dict[str, Any]) -> int:
    _ensure_parent(REQUESTS_PATH)
    reqs = []
    if REQUESTS_PATH.exists():
        try:
            reqs = json.loads(REQUESTS_PATH.read_text(encoding="utf-8"))
        except Exception:
            reqs = []
    reqs.append(item)
    REQUESTS_PATH.write_text(json.dumps(reqs, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(reqs)


def list_change_requests() -> List[Dict[str, Any]]:
    if not REQUESTS_PATH.exists():
        return []
    try:
        return json.loads(REQUESTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def approve_request(idx: int) -> bool:
    reqs = list_change_requests()
    if 0 <= idx < len(reqs):
        item = reqs.pop(idx)
        # 카탈로그에 반영
        items = load_catalog()
        items.append(item)
        save_catalog(items)
        REQUESTS_PATH.write_text(json.dumps(reqs, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    return False


def reject_request(idx: int) -> bool:
    reqs = list_change_requests()
    if 0 <= idx < len(reqs):
        reqs.pop(idx)
        REQUESTS_PATH.write_text(json.dumps(reqs, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    return False


