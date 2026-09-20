"""Selective Knowledge Core ingestion CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))
sys.path.insert(0, str(project_root))

from chatbot.knowledge.ingestion import (
    IngestionStateStore,
    SelectiveIngestionManager,
    SourceDocument,
)


def _load_documents(path: Path) -> list[SourceDocument]:
    documents = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                documents.append(
                    SourceDocument(
                        doc_id=str(value["doc_id"]),
                        source_uri=str(value["source_uri"]),
                        text=str(value["text"]),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid JSONL record at line {line_number}: {exc}"
                ) from exc
    return documents


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Selectively ingest changed text sources into the Knowledge Core."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="JSONL with doc_id, source_uri and text fields.",
    )
    parser.add_argument(
        "--state",
        default="knowledge-workspace/ingestion-state.json",
        help="Derived ingestion state path.",
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="Do not remove stored sources absent from this input batch.",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable LLM-assisted extraction/validation.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    state_path = Path(args.state).expanduser()
    if not state_path.is_absolute():
        state_path = project_root / state_path

    documents = _load_documents(input_path)
    manager = SelectiveIngestionManager(
        project_root=project_root,
        state_store=IngestionStateStore(state_path),
        use_llm=args.use_llm,
    )
    report = manager.sync(
        documents,
        prune_missing=not args.no_prune,
    )
    print(
        json.dumps(
            report.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
