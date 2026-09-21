"""Inspect and verify immutable Knowledge Core replay history."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))
sys.path.insert(0, str(project_root))

from chatbot.knowledge.ingestion import IngestionStateStore
from chatbot.knowledge.replay import (
    KnowledgeReplayStore,
    SnapshotChangeSummary,
    datetime_from_text,
)


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def _load_bootstrap_state(value: str):
    path = _resolve(value)
    if not path.is_file():
        raise FileNotFoundError(f"bootstrap state file not found: {path}")
    return IngestionStateStore(path).load()


def _snapshot_payload(snapshot) -> dict:
    return {
        "snapshot_id": snapshot.snapshot_id,
        "sequence": snapshot.sequence,
        "committed_at": snapshot.committed_at.isoformat(),
        "parent_snapshot_id": snapshot.parent_snapshot_id,
        "state_digest": snapshot.state_digest,
        "origin": snapshot.origin,
        "sources": sorted(snapshot.state.records),
        "change_summary": snapshot.change_summary.to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect immutable Knowledge Core as-of replay history."
    )
    parser.add_argument(
        "--replay-dir",
        default="knowledge-workspace/replay",
        help="Replay history directory.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List snapshot index entries.")

    show = subparsers.add_parser("show", help="Show one snapshot.")
    show.add_argument("--snapshot", required=True)

    as_of = subparsers.add_parser("as-of", help="Resolve state at or before time.")
    as_of.add_argument(
        "--at",
        required=True,
        help="Timezone-aware ISO-8601 timestamp.",
    )

    subparsers.add_parser("verify", help="Verify the complete replay chain.")

    bootstrap = subparsers.add_parser(
        "bootstrap",
        help="Create the first snapshot from the current ingestion state.",
    )
    bootstrap.add_argument(
        "--state",
        default="knowledge-workspace/ingestion-state.json",
    )

    args = parser.parse_args()
    store = KnowledgeReplayStore(_resolve(args.replay_dir))

    if args.command == "list":
        payload = [entry.to_dict() for entry in store.list()]
    elif args.command == "show":
        payload = _snapshot_payload(store.get(args.snapshot))
    elif args.command == "as-of":
        snapshot = store.as_of(datetime_from_text(args.at))
        payload = None if snapshot is None else _snapshot_payload(snapshot)
    elif args.command == "verify":
        payload = store.verify().to_dict()
    else:
        if store.latest() is not None:
            raise ValueError("bootstrap requires an empty replay history")
        state = _load_bootstrap_state(args.state)
        snapshot = store.record(
            state,
            committed_at=datetime.now(timezone.utc),
            change_summary=SnapshotChangeSummary(
                action_counts={"bootstrap": 1},
                source_uris=tuple(state.records),
            ),
            origin="bootstrap",
        )
        payload = _snapshot_payload(snapshot)

    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
