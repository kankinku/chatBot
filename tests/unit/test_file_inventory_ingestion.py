"""Raw-file inventory and pre-extraction selective ingestion contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from chatbot.knowledge.ingestion import (
    ExtractedTextCache,
    FileInventoryStateStore,
    SelectiveFileIngestionManager,
)
from chatbot.knowledge.ingestion.models import SelectiveIngestionReport


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _FakeExtractor:
    def __init__(self):
        self.calls: list[Path] = []

    def extract(self, path: str | Path) -> str:
        source = Path(path)
        self.calls.append(source)
        text = source.read_text(encoding="utf-8")
        if text == "FAIL":
            raise RuntimeError("synthetic extraction failure")
        return text


class _FakeIngestionManager:
    def __init__(self):
        self.calls = []

    def sync(
        self,
        documents,
        *,
        prune_missing=True,
        prune_scope=None,
    ):
        docs = list(documents)
        self.calls.append(
            {
                "documents": docs,
                "prune_missing": prune_missing,
                "prune_scope": set(prune_scope or ()),
            }
        )
        return SelectiveIngestionReport()


def _manager(
    tmp_path: Path,
    *,
    extractor=None,
    stamp_box=None,
    source_root=None,
):
    extractor = extractor or _FakeExtractor()
    ingestion = _FakeIngestionManager()
    stamp_box = stamp_box or ["a" * 64]
    workspace = tmp_path / "workspace"
    manager = SelectiveFileIngestionManager(
        project_root=PROJECT_ROOT,
        ingestion_manager=ingestion,
        inventory_store=FileInventoryStateStore(
            workspace / "file-inventory.json"
        ),
        text_cache=ExtractedTextCache(
            workspace / "extracted-text"
        ),
        extractor=extractor,
        stamp_provider=lambda root: stamp_box[0],
    )
    root = source_root or (tmp_path / "sources")
    root.mkdir(parents=True, exist_ok=True)
    return manager, extractor, ingestion, stamp_box, root


def test_unchanged_file_reuses_cached_text_without_reextracting(tmp_path: Path):
    manager, extractor, ingestion, _, root = _manager(tmp_path)
    source = root / "a.txt"
    source.write_text("alpha", encoding="utf-8")

    first = manager.sync(root, patterns=("**/*.txt",))
    second = manager.sync(root, patterns=("**/*.txt",))

    assert first.extracted == 1
    assert first.reused == 0
    assert second.extracted == 0
    assert second.reused == 1
    assert len(extractor.calls) == 1
    assert ingestion.calls[-1]["documents"][0].text == "alpha"


def test_mtime_only_change_does_not_trigger_text_extraction(tmp_path: Path):
    manager, extractor, _, _, root = _manager(tmp_path)
    source = root / "a.txt"
    source.write_text("alpha", encoding="utf-8")

    manager.sync(root, patterns=("**/*.txt",))
    stat = source.stat()
    os.utime(source, (stat.st_atime + 30, stat.st_mtime + 30))
    report = manager.sync(root, patterns=("**/*.txt",))

    assert report.reused == 1
    assert len(extractor.calls) == 1


def test_byte_change_reextracts_only_changed_source(tmp_path: Path):
    manager, extractor, ingestion, _, root = _manager(tmp_path)
    first_path = root / "one.txt"
    second_path = root / "two.txt"
    first_path.write_text("one-v1", encoding="utf-8")
    second_path.write_text("two-v1", encoding="utf-8")

    manager.sync(root, patterns=("**/*.txt",))
    first_path.write_text("one-v2", encoding="utf-8")
    report = manager.sync(root, patterns=("**/*.txt",))

    assert report.extracted == 1
    assert report.reused == 1
    assert len(extractor.calls) == 3
    docs = {
        doc.source_uri: doc.text
        for doc in ingestion.calls[-1]["documents"]
    }
    assert docs == {
        "file:one.txt": "one-v2",
        "file:two.txt": "two-v1",
    }


def test_extractor_stamp_change_forces_reextraction(tmp_path: Path):
    stamp_box = ["a" * 64]
    manager, extractor, _, _, root = _manager(
        tmp_path,
        stamp_box=stamp_box,
    )
    source = root / "a.txt"
    source.write_text("alpha", encoding="utf-8")

    manager.sync(root, patterns=("**/*.txt",))
    stamp_box[0] = "b" * 64
    report = manager.sync(root, patterns=("**/*.txt",))

    assert report.extracted == 1
    assert report.file_results[0].reason == "extractor_changed"
    assert len(extractor.calls) == 2


def test_corrupt_text_cache_is_rebuilt_from_source(tmp_path: Path):
    manager, extractor, _, _, root = _manager(tmp_path)
    source = root / "a.txt"
    source.write_text("alpha", encoding="utf-8")

    manager.sync(root, patterns=("**/*.txt",))
    state = manager.inventory_store.load(root)
    record = state.records["file:a.txt"]
    manager.text_cache.path_for(record.cache_key).write_text(
        "corrupt",
        encoding="utf-8",
    )

    report = manager.sync(root, patterns=("**/*.txt",))

    assert report.extracted == 1
    assert report.file_results[0].reason == "cache_missing_or_corrupt"
    assert len(extractor.calls) == 2


def test_changed_file_extraction_failure_preserves_previous_cached_document(
    tmp_path: Path,
):
    manager, extractor, ingestion, _, root = _manager(tmp_path)
    source = root / "a.txt"
    source.write_text("good", encoding="utf-8")
    manager.sync(root, patterns=("**/*.txt",))
    before = manager.inventory_store.load(root)
    before_hash = before.records["file:a.txt"].byte_hash

    source.write_text("FAIL", encoding="utf-8")
    report = manager.sync(root, patterns=("**/*.txt",))

    assert report.file_results[0].action == "failed"
    assert ingestion.calls[-1]["prune_missing"] is True
    assert ingestion.calls[-1]["documents"][0].text == "good"
    after = manager.inventory_store.load(root)
    assert after.records["file:a.txt"].byte_hash == before_hash
    assert len(extractor.calls) == 2


def test_missing_previous_cache_and_failed_reextract_disables_pruning(
    tmp_path: Path,
):
    manager, _, ingestion, stamp_box, root = _manager(tmp_path)
    source = root / "a.txt"
    source.write_text("good", encoding="utf-8")
    manager.sync(root, patterns=("**/*.txt",))
    state = manager.inventory_store.load(root)
    record = state.records["file:a.txt"]
    manager.text_cache.path_for(record.cache_key).unlink()

    source.write_text("FAIL", encoding="utf-8")
    stamp_box[0] = "b" * 64
    report = manager.sync(root, patterns=("**/*.txt",))

    assert report.file_results[0].action == "failed"
    assert ingestion.calls[-1]["prune_missing"] is False


def test_removed_file_is_removed_from_inventory_and_downstream_batch(
    tmp_path: Path,
):
    manager, _, ingestion, _, root = _manager(tmp_path)
    one = root / "one.txt"
    two = root / "two.txt"
    one.write_text("one", encoding="utf-8")
    two.write_text("two", encoding="utf-8")
    manager.sync(root, patterns=("**/*.txt",))

    one.unlink()
    report = manager.sync(root, patterns=("**/*.txt",))

    assert report.removed == 1
    assert report.file_results[-1].source_uri == "file:one.txt"
    assert {
        doc.source_uri
        for doc in ingestion.calls[-1]["documents"]
    } == {"file:two.txt"}
    state = manager.inventory_store.load(root)
    assert set(state.records) == {"file:two.txt"}


def test_nested_relative_path_gives_stable_source_uri_and_doc_id(tmp_path: Path):
    manager, _, ingestion, _, root = _manager(tmp_path)
    nested = root / "reports" / "daily"
    nested.mkdir(parents=True)
    source = nested / "risk.md"
    source.write_text("risk text", encoding="utf-8")

    manager.sync(root, patterns=("**/*.md",))
    document = ingestion.calls[-1]["documents"][0]

    assert document.source_uri == "file:reports/daily/risk.md"
    assert document.doc_id == "reports/daily/risk"


def test_inventory_state_rejects_reuse_with_different_source_root(tmp_path: Path):
    manager, _, _, _, first_root = _manager(tmp_path)
    (first_root / "a.txt").write_text("alpha", encoding="utf-8")
    manager.sync(first_root, patterns=("**/*.txt",))

    second_root = tmp_path / "other"
    second_root.mkdir()
    (second_root / "a.txt").write_text("alpha", encoding="utf-8")

    with pytest.raises(ValueError, match="source root mismatch"):
        manager.sync(second_root, patterns=("**/*.txt",))


def test_source_symlink_escape_is_rejected(tmp_path: Path):
    if not hasattr(os, "symlink"):
        pytest.skip("symlink not supported")

    manager, _, _, _, root = _manager(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = root / "escape.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted")

    with pytest.raises(ValueError, match="escapes source root"):
        manager.sync(root, patterns=("**/*.txt",))


def test_inventory_state_and_text_cache_are_derived_not_source_truth(tmp_path: Path):
    manager, _, _, _, root = _manager(tmp_path)
    source = root / "a.txt"
    source.write_text("alpha", encoding="utf-8")
    manager.sync(root, patterns=("**/*.txt",))

    state = manager.inventory_store.load(root)
    record = state.records["file:a.txt"]
    cache_path = manager.text_cache.path_for(record.cache_key)

    assert manager.inventory_store.path.is_file()
    assert cache_path.is_file()
    assert cache_path.read_text(encoding="utf-8") == "alpha"


def test_inventory_rejects_tampered_cache_key(tmp_path: Path):
    manager, _, _, _, root = _manager(tmp_path)
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    manager.sync(root, patterns=("**/*.txt",))

    path = manager.inventory_store.path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"][0]["cache_key"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cache key mismatch"):
        manager.inventory_store.load(root)


def test_extractor_stamp_changes_when_extractor_source_changes(tmp_path: Path):
    from chatbot.knowledge.ingestion.file_inventory import extractor_stamp

    first_file = (
        tmp_path
        / "src/chatbot/knowledge/ingestion/file_inventory.py"
    )
    second_file = (
        tmp_path
        / "src/chatbot/preprocessing/pdf_extractor.py"
    )
    first_file.parent.mkdir(parents=True)
    second_file.parent.mkdir(parents=True)
    first_file.write_text("VERSION = 1\n", encoding="utf-8")
    second_file.write_text("VERSION = 1\n", encoding="utf-8")

    first = extractor_stamp(tmp_path)
    second_file.write_text("VERSION = 2\n", encoding="utf-8")
    second = extractor_stamp(tmp_path)

    assert len(first) == 64
    assert len(second) == 64
    assert first != second
