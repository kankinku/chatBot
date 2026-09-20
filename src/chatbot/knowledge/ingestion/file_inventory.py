"""Raw-file inventory and extracted-text cache for selective ingestion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Callable, Iterable

from chatbot.preprocessing.pdf_extractor import PDFExtractor
from chatbot.knowledge.workspace.hashing import hash_file, hash_value


FILE_INVENTORY_SCHEMA_VERSION = 1
DEFAULT_FILE_PATTERNS = ("**/*.pdf", "**/*.txt", "**/*.md")
TEXT_SUFFIXES = {".txt", ".md"}


def hash_source_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash source bytes without loading a large document fully into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def extractor_stamp(project_root: str | Path) -> str:
    """Fingerprint code that can change byte->text extraction output."""
    root = Path(project_root).resolve()
    paths = (
        root / "src/chatbot/knowledge/ingestion/file_inventory.py",
        root / "src/chatbot/preprocessing/pdf_extractor.py",
    )
    lines = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"extractor fingerprint input not found: {path}")
        lines.append(f"{path.relative_to(root).as_posix()}:{hash_file(path)}")
    try:
        pymupdf_version = metadata.version("PyMuPDF")
    except metadata.PackageNotFoundError:
        pymupdf_version = "missing"
    lines.append(f"dependency:PyMuPDF:{pymupdf_version}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FileInventoryRecord:
    relative_path: str
    source_uri: str
    byte_hash: str
    extractor_stamp: str
    text_hash: str
    cache_key: str

    def to_dict(self) -> dict[str, str]:
        return {
            "relative_path": self.relative_path,
            "source_uri": self.source_uri,
            "byte_hash": self.byte_hash,
            "extractor_stamp": self.extractor_stamp,
            "text_hash": self.text_hash,
            "cache_key": self.cache_key,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "FileInventoryRecord":
        return cls(
            relative_path=str(value["relative_path"]),
            source_uri=str(value["source_uri"]),
            byte_hash=str(value["byte_hash"]),
            extractor_stamp=str(value["extractor_stamp"]),
            text_hash=str(value["text_hash"]),
            cache_key=str(value["cache_key"]),
        )


@dataclass
class FileInventoryState:
    source_root: str
    records: dict[str, FileInventoryRecord] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema_version": FILE_INVENTORY_SCHEMA_VERSION,
            "source_root": self.source_root,
            "records": [
                self.records[source_uri].to_dict()
                for source_uri in sorted(self.records)
            ],
        }

    @classmethod
    def from_dict(cls, value: dict) -> "FileInventoryState":
        if value.get("schema_version") != FILE_INVENTORY_SCHEMA_VERSION:
            raise ValueError("unsupported file inventory schema")
        state = cls(source_root=str(value["source_root"]))
        for item in value.get("records", []):
            record = FileInventoryRecord.from_dict(item)
            if record.source_uri in state.records:
                raise ValueError(
                    f"duplicate file inventory source: {record.source_uri}"
                )
            state.records[record.source_uri] = record
        state.validate()
        return state

    def validate(self) -> None:
        root = Path(self.source_root)
        if not root.is_absolute():
            raise ValueError("file inventory source_root must be absolute")
        for source_uri, record in self.records.items():
            if source_uri != record.source_uri:
                raise ValueError(f"file inventory key mismatch: {source_uri}")
            relative = Path(record.relative_path)
            if (
                not record.relative_path
                or relative.is_absolute()
                or ".." in relative.parts
            ):
                raise ValueError(f"invalid relative source path: {source_uri}")
            if source_uri != f"file:{record.relative_path}":
                raise ValueError(f"invalid source uri: {source_uri}")
            expected_cache_key = hash_value(
                {
                    "byte_hash": record.byte_hash,
                    "extractor_stamp": record.extractor_stamp,
                }
            )
            if record.cache_key != expected_cache_key:
                raise ValueError(f"file inventory cache key mismatch: {source_uri}")
            for name, value in (
                ("byte_hash", record.byte_hash),
                ("extractor_stamp", record.extractor_stamp),
                ("text_hash", record.text_hash),
                ("cache_key", record.cache_key),
            ):
                if len(value) != 64:
                    raise ValueError(f"invalid {name}: {source_uri}")
                try:
                    int(value, 16)
                except ValueError as exc:
                    raise ValueError(f"invalid {name}: {source_uri}") from exc


class FileInventoryStateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self, source_root: str | Path) -> FileInventoryState:
        root = str(Path(source_root).resolve())
        if not self.path.exists():
            return FileInventoryState(source_root=root)
        state = FileInventoryState.from_dict(
            json.loads(self.path.read_text(encoding="utf-8"))
        )
        if state.source_root != root:
            raise ValueError(
                "file inventory source root mismatch: "
                f"stored={state.source_root} current={root}"
            )
        return state

    def save(self, state: FileInventoryState) -> None:
        state.validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(f".{self.path.name}.tmp")
        temp.write_text(
            json.dumps(
                state.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temp.replace(self.path)


class ExtractedTextCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    @staticmethod
    def cache_key(byte_hash: str, stamp: str) -> str:
        return hash_value(
            {
                "byte_hash": byte_hash,
                "extractor_stamp": stamp,
            }
        )

    def path_for(self, cache_key: str) -> Path:
        if len(cache_key) != 64:
            raise ValueError("invalid extracted-text cache key")
        try:
            int(cache_key, 16)
        except ValueError as exc:
            raise ValueError("invalid extracted-text cache key") from exc
        return self.root / f"{cache_key}.txt"

    def read(self, record: FileInventoryRecord) -> str | None:
        path = self.path_for(record.cache_key)
        if not path.is_file():
            return None
        text = path.read_text(encoding="utf-8")
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_hash != record.text_hash:
            return None
        return text

    def write(self, cache_key: str, text: str) -> str:
        path = self.path_for(cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.tmp")
        temp.write_text(text, encoding="utf-8")
        temp.replace(path)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def prune(self, referenced_keys: Iterable[str]) -> None:
        if not self.root.exists():
            return
        keep = set(referenced_keys)
        for path in self.root.glob("*.txt"):
            if path.stem not in keep:
                path.unlink()


class FileTextExtractor:
    """Extract canonical text from supported local source files."""

    def __init__(self, pdf_extractor: PDFExtractor | None = None):
        self.pdf_extractor = pdf_extractor or PDFExtractor()

    def extract(self, path: str | Path) -> str:
        source = Path(path)
        suffix = source.suffix.lower()
        if suffix == ".pdf":
            return self.pdf_extractor.extract_text_from_file(source)
        if suffix in TEXT_SUFFIXES:
            return source.read_text(encoding="utf-8")
        raise ValueError(f"unsupported source file type: {source.suffix}")


def scan_source_files(
    source_root: str | Path,
    patterns: Iterable[str] = DEFAULT_FILE_PATTERNS,
) -> list[tuple[str, Path]]:
    """Return stable relative paths and resolved files, rejecting escapes."""
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"source directory not found: {root}")

    found: dict[Path, str] = {}
    for pattern in patterns:
        for candidate in root.glob(pattern):
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            try:
                relative = resolved.relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError(
                    f"source file escapes source root: {candidate}"
                ) from exc
            if resolved in found and found[resolved] != relative:
                raise ValueError(
                    f"source file resolved through multiple paths: {candidate}"
                )
            found[resolved] = relative

    by_relative: dict[str, Path] = {}
    for resolved, relative in found.items():
        if relative in by_relative and by_relative[relative] != resolved:
            raise ValueError(f"duplicate source relative path: {relative}")
        by_relative[relative] = resolved

    return [
        (relative, by_relative[relative])
        for relative in sorted(by_relative)
    ]


def source_uri_for(relative_path: str) -> str:
    return f"file:{relative_path}"


def doc_id_for(relative_path: str) -> str:
    path = Path(relative_path)
    return path.with_suffix("").as_posix()
