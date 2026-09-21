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
    FileInventoryStateStore,
    SelectiveFileIngestionManager,
    IngestionStateStore,
    SelectiveIngestionManager,
    SourceDocument,
)
from chatbot.knowledge.replay import KnowledgeReplayStore


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


def _resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Selectively ingest changed text or local source files "
            "into the Knowledge Core."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--input",
        help="JSONL with doc_id, source_uri and text fields.",
    )
    source_group.add_argument(
        "--source-dir",
        help=(
            "Directory containing PDF/TXT/MD sources. Raw bytes are "
            "hashed before text extraction."
        ),
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help=(
            "Glob pattern used with --source-dir. Repeat for multiple "
            "patterns. Defaults to **/*.pdf, **/*.txt, **/*.md."
        ),
    )
    parser.add_argument(
        "--state",
        default="knowledge-workspace/ingestion-state.json",
        help="Derived Knowledge Core ingestion state path.",
    )
    parser.add_argument(
        "--file-inventory-state",
        default="knowledge-workspace/file-inventory.json",
        help="Derived raw-file inventory state path.",
    )
    parser.add_argument(
        "--replay-dir",
        default="knowledge-workspace/replay",
        help="Immutable Knowledge Core replay history directory.",
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

    state_path = _resolve_project_path(args.state)
    replay_dir = _resolve_project_path(args.replay_dir)
    ingestion_manager = SelectiveIngestionManager(
        project_root=project_root,
        state_store=IngestionStateStore(state_path),
        replay_store=KnowledgeReplayStore(replay_dir),
        use_llm=args.use_llm,
    )

    if args.input:
        input_path = Path(args.input).expanduser().resolve()
        documents = _load_documents(input_path)
        report = ingestion_manager.sync(
            documents,
            prune_missing=not args.no_prune,
        )
        payload = report.to_dict()
        failed = report.failed
    else:
        source_root = Path(args.source_dir).expanduser().resolve()
        inventory_path = _resolve_project_path(args.file_inventory_state)
        file_manager = SelectiveFileIngestionManager(
            project_root=project_root,
            ingestion_manager=ingestion_manager,
            inventory_store=FileInventoryStateStore(inventory_path),
        )
        kwargs = {}
        if args.pattern:
            kwargs["patterns"] = tuple(args.pattern)
        report = file_manager.sync(
            source_root,
            prune_missing=not args.no_prune,
            **kwargs,
        )
        payload = report.to_dict()
        failed = report.failed

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
