"""CLI for the local knowledge relationship/provenance workspace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))
sys.path.insert(0, str(project_root))

from chatbot.knowledge.workspace import (
    blast_radius,
    build_workspace,
    check_workspace,
    load_workspace_graph,
)


def _json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _root(value: str | None) -> Path:
    return Path(value).expanduser().resolve() if value else project_root


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build/check/query the local knowledge relationship workspace."
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Repository root. Defaults to the current project root.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build",
        help="Build or incrementally refresh the generated workspace.",
    )
    build.add_argument(
        "--cold",
        action="store_true",
        help="Ignore the parse cache and rebuild every source.",
    )

    subparsers.add_parser(
        "check",
        help="Check whether source hashes or generator code have drifted.",
    )

    impact = subparsers.add_parser(
        "impact",
        help="Walk dependency edges from one or more node ids.",
    )
    impact.add_argument("seed", nargs="+", help="Seed node id(s).")
    impact.add_argument(
        "--allow-stale",
        action="store_true",
        help="Allow impact queries even when the generated graph is stale.",
    )

    args = parser.parse_args()
    root = _root(args.root)

    if args.command == "build":
        result = build_workspace(root, reuse=not args.cold)
        _json(result.to_dict())
        return 0

    if args.command == "check":
        report = check_workspace(root)
        _json(report.to_dict())
        return 0 if report.clean else 1

    if args.command == "impact":
        drift = check_workspace(root)
        if not drift.clean and not args.allow_stale:
            _json(
                {
                    "error": "workspace is stale; run build first",
                    "drift": drift.to_dict(),
                }
            )
            return 2
        graph = load_workspace_graph(root)
        hits = blast_radius(graph, args.seed)
        _json(
            {
                "seeds": sorted(set(args.seed)),
                "hits": [
                    {
                        "node_id": hit.node_id,
                        "from_id": hit.from_id,
                        "relation": hit.relation,
                        "depth": hit.depth,
                    }
                    for hit in hits
                ],
            }
        )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
