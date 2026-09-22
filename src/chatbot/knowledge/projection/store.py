"""Immutable content-addressed cache for reproducible scenario projections."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

from chatbot.knowledge.workspace.hashing import canonical_json, hash_value

from .models import ScenarioProjection


class ProjectionStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def save(self, projection: ScenarioProjection) -> Path:
        if projection.base.snapshot_id is None:
            raise ValueError("only snapshot-backed projections may be persisted")
        payload = projection.to_dict()
        self._verify_payload(payload, expected_id=projection.projection_id)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(projection.projection_id)
        serialized = (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            if existing != serialized:
                raise ValueError(
                    f"conflicting immutable projection: {projection.projection_id}"
                )
            return path

        fd, temp_name = tempfile.mkstemp(
            prefix=f".{projection.projection_id}.",
            suffix=".tmp",
            dir=self.root,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temp_name, path)
            except FileExistsError:
                existing = path.read_text(encoding="utf-8")
                if existing != serialized:
                    raise ValueError(
                        f"conflicting immutable projection: {projection.projection_id}"
                    )
            return path
        finally:
            try:
                Path(temp_name).unlink()
            except FileNotFoundError:
                pass

    def load(self, projection_id: str) -> dict[str, Any]:
        path = self._path(projection_id)
        if not path.is_file():
            raise KeyError(f"unknown projection: {projection_id}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("invalid projection JSON") from exc
        self._verify_payload(payload, expected_id=projection_id)
        return payload

    def verify(self, projection_id: str) -> dict[str, Any]:
        payload = self.load(projection_id)
        return {
            "projection_id": projection_id,
            "valid": True,
            "output_digest": payload["output_digest"],
        }

    def _path(self, projection_id: str) -> Path:
        if (
            not projection_id.startswith("proj_")
            or "/" in projection_id
            or "\\" in projection_id
            or ".." in projection_id
        ):
            raise ValueError("invalid projection id")
        return self.root / f"{projection_id}.json"

    @staticmethod
    def _verify_payload(
        payload: dict[str, Any],
        *,
        expected_id: str,
    ) -> None:
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported projection schema")
        if payload.get("projection_id") != expected_id:
            raise ValueError("projection id mismatch")
        base = payload.get("base", {})
        expected_projection_id = "proj_" + hash_value(
            {
                "engine_version": payload.get("engine_version"),
                "base": {
                    "state_digest": base.get("state_digest"),
                    "snapshot_id": base.get("snapshot_id"),
                    "origin": base.get("origin"),
                },
                "scenario_spec_id": payload.get("scenario_spec_id"),
                "regime_spec_id": payload.get("regime_spec_id"),
            }
        )
        if expected_projection_id != expected_id:
            raise ValueError("projection semantic identity mismatch")
        trace = payload.get("trace", {})
        if trace.get("projection_id") != expected_id:
            raise ValueError("projection trace id mismatch")
        if (
            trace.get("engine_version") != payload.get("engine_version")
            or trace.get("base") != base
            or trace.get("scenario_spec_id") != payload.get("scenario_spec_id")
            or trace.get("regime_spec_id") != payload.get("regime_spec_id")
        ):
            raise ValueError("projection trace metadata mismatch")
        core = {
            "projection_id": payload.get("projection_id"),
            "engine_version": payload.get("engine_version"),
            "base": payload.get("base"),
            "scenario_spec_id": payload.get("scenario_spec_id"),
            "regime_spec_id": payload.get("regime_spec_id"),
            "relations": payload.get("relations"),
            "impacts": payload.get("impacts"),
            "node_summaries": payload.get("node_summaries"),
            "sensitivity": payload.get("sensitivity"),
            "dependencies": trace.get("dependencies"),
            "warnings": trace.get("warnings"),
        }
        expected_digest = hash_value(core)
        if payload.get("output_digest") != expected_digest:
            raise ValueError("projection output digest mismatch")
        if trace.get("output_digest") != expected_digest:
            raise ValueError("projection trace digest mismatch")
