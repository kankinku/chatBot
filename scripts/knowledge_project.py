"""Run deterministic scenario/regime projections over Knowledge Core state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))
sys.path.insert(0, str(project_root))

from chatbot.knowledge.ingestion import IngestionStateStore
from chatbot.knowledge.projection import (
    ProjectionBaseResolver,
    ProjectionService,
    ProjectionStore,
    RegimeSpec,
    ScenarioSpec,
)
from chatbot.knowledge.replay import KnowledgeReplayService, KnowledgeReplayStore
from chatbot.knowledge.replay import datetime_from_text


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def _load_json(path: str) -> dict:
    target = _resolve(path)
    if not target.is_file():
        raise FileNotFoundError(f"JSON input not found: {target}")
    return json.loads(target.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Project hypothetical scenario/regime effects without mutating Knowledge Core."
    )
    parser.add_argument(
        "--replay-dir",
        default="knowledge-workspace/replay",
    )
    parser.add_argument(
        "--state",
        default="knowledge-workspace/ingestion-state.json",
    )
    parser.add_argument(
        "--projection-dir",
        default="knowledge-workspace/projections",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    project = sub.add_parser("project")
    project.add_argument("--scenario", required=True)
    project.add_argument("--regime")
    base = project.add_mutually_exclusive_group(required=True)
    base.add_argument("--snapshot")
    base.add_argument("--as-of")
    base.add_argument("--current", action="store_true")
    project.add_argument("--persist", action="store_true")

    show = sub.add_parser("show")
    show.add_argument("--projection", required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("--projection", required=True)

    args = parser.parse_args()
    store = ProjectionStore(_resolve(args.projection_dir))

    if args.command == "show":
        payload = store.load(args.projection)
    elif args.command == "verify":
        payload = store.verify(args.projection)
    else:
        replay_store = KnowledgeReplayStore(_resolve(args.replay_dir))
        resolver = ProjectionBaseResolver(
            replay_service=KnowledgeReplayService(replay_store),
            current_state_store=IngestionStateStore(_resolve(args.state)),
        )
        service = ProjectionService(
            base_resolver=resolver,
            store=store,
        )
        scenario = ScenarioSpec.from_dict(_load_json(args.scenario))
        regime = (
            RegimeSpec.from_dict(_load_json(args.regime))
            if args.regime
            else None
        )
        if args.snapshot:
            projection = service.project_snapshot(
                args.snapshot,
                scenario,
                regime,
                persist=args.persist,
            )
        elif args.as_of:
            projection = service.project_as_of(
                datetime_from_text(args.as_of),
                scenario,
                regime,
                persist=args.persist,
            )
        else:
            projection = service.project_current(
                scenario,
                regime,
                persist=args.persist,
            )
        payload = projection.to_dict()

    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
