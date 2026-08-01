import hashlib
import os
from pathlib import Path
from typing import Generator, Iterable, List, Tuple

from pypdf import PdfReader


def list_pdf_files(data_dir: str) -> List[Path]:
    base = Path(data_dir)
    if not base.exists() or not base.is_dir():
        return []
    return sorted([path for path in base.glob("*.pdf") if path.is_file()])


def extract_pdf_pages(path: Path) -> List[Tuple[int, str]]:
    reader = PdfReader(str(path))
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").replace("\x00", " ").strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    clean = " ".join(text.split())
    if not clean:
        return []

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks: List[str] = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(clean):
            break
        start = max(0, end - overlap)
    return chunks


def deterministic_record_id(source: str, page: int, chunk_number: int, chunk_text_value: str) -> str:
    payload = f"{source}|{page}|{chunk_number}|{chunk_text_value}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:24]
    safe_source = source.replace(os.sep, "_").replace(" ", "_")[:60]
    return f"{safe_source}-p{page}-c{chunk_number}-{digest}"


def batched(items: Iterable[dict], size: int) -> Generator[List[dict], None, None]:
    batch: List[dict] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
