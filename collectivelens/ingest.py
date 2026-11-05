"""Read documents and their metadata from a folder."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .textutils import detect_language, normalize_key, slugify

log = logging.getLogger(__name__)

SUPPORTED = {".pdf", ".txt", ".md"}
MAX_PDF_PAGES = 300


@dataclass
class RawDocument:
    id: str
    filename: str
    title: str
    org: str
    year: int | None
    language: str
    text: str
    source_url: str = ""
    gold_topic: str = ""
    problems: list[str] = field(default_factory=list)


def read_text(path: Path) -> str:
    """Text of a PDF, .txt or .md file. Pages that cannot be read are skipped."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix != ".pdf":
        raise ValueError(f"unsupported file type: {path.name}")
    import pdfplumber  # imported here so the app starts without it

    parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages[:MAX_PDF_PAGES]:
            try:
                text = page.extract_text()
            except Exception:  # a damaged page should not lose the whole document
                continue
            if text:
                parts.append(text)
    return " ".join(parts)


def _to_year(value: object) -> int | None:
    try:
        year = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    return year if 1900 <= year <= 2100 else None


def load_metadata(csv_path: Path | str | None) -> dict[str, dict[str, str]]:
    """Metadata rows by file key. Columns: file_name, title, member_org, publication_year,
    and optionally source_url and gold_topic. Column names are case-insensitive."""
    if not csv_path:
        return {}
    path = Path(csv_path)
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(raw[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows: dict[str, dict[str, str]] = {}
    for row in csv.DictReader(raw.splitlines(), dialect=dialect):
        clean = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
        key = normalize_key(clean.get("file_name", ""))
        if key and key not in rows:  # the first row wins when a file appears twice
            rows[key] = clean
    return rows


def ingest_folder(input_dir: Path | str, metadata_csv: Path | str | None = None) -> list[RawDocument]:
    folder = Path(input_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    metadata = load_metadata(metadata_csv)
    docs: list[RawDocument] = []
    used_ids: set[str] = set()
    for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED):
        try:
            text = read_text(path)
        except Exception as exc:
            log.warning("Skipped %s: %s", path.name, exc)
            continue
        if len(text.strip()) < 100:
            log.warning("Skipped %s: almost no text (a scanned PDF needs OCR first)", path.name)
            continue
        row = metadata.get(normalize_key(path.name), {})
        doc_id = base = slugify(path.stem)
        n = 2
        while doc_id in used_ids:
            doc_id = f"{base}-{n}"
            n += 1
        used_ids.add(doc_id)
        problems = [] if row else ["no metadata row"]
        docs.append(
            RawDocument(
                id=doc_id,
                filename=path.name,
                title=row.get("title") or path.stem.replace("_", " ").replace("-", " "),
                org=row.get("member_org") or row.get("org") or "Unknown",
                year=_to_year(row.get("publication_year") or row.get("year")),
                language=detect_language(text),
                text=" ".join(text.split()),
                source_url=row.get("source_url", ""),
                gold_topic=row.get("gold_topic", ""),
                problems=problems,
            )
        )
    return docs
