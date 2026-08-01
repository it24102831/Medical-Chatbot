import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from dotenv import load_dotenv
from pinecone import Pinecone

from src.helper import batched, chunk_text, deterministic_record_id, extract_pdf_pages, list_pdf_files

load_dotenv()


@dataclass
class IngestSummary:
    pdfs_processed: int = 0
    pages_processed: int = 0
    chunks_generated: int = 0
    records_uploaded: int = 0


def env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else value


def is_placeholder(value: Optional[str]) -> bool:
    if not value:
        return True
    lowered = value.lower()
    return "your_" in lowered or lowered in {"changeme", "replace-me", "example"}


def retry_call(action_name: str, fn, attempts: int = 5):
    delay = 0.5
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            text = str(exc).lower()
            transient = any(x in text for x in ["timeout", "tempor", "rate", "connection", "503", "502"])
            if attempt >= attempts or not transient:
                raise RuntimeError(f"{action_name} failed: {exc}") from exc
            print(f"[{action_name}] transient failure ({attempt}/{attempts}): retrying in {delay:.1f}s")
            time.sleep(delay)
            delay *= 2


def resolve_model_from_index_description(desc: object) -> Optional[str]:
    if isinstance(desc, dict):
        embed = desc.get("spec", {}).get("embed") if isinstance(desc.get("spec"), dict) else None
        if isinstance(embed, dict):
            return embed.get("model")
        return desc.get("embed", {}).get("model") if isinstance(desc.get("embed"), dict) else None

    spec = getattr(desc, "spec", None)
    if spec is not None:
        embed = getattr(spec, "embed", None)
        if embed is not None:
            model = getattr(embed, "model", None)
            if model:
                return model

    embed = getattr(desc, "embed", None)
    if embed is not None:
        return getattr(embed, "model", None)
    return None


def index_ready(desc: object) -> bool:
    if isinstance(desc, dict):
        status = desc.get("status", {})
        return bool(status.get("ready", False)) if isinstance(status, dict) else False

    status = getattr(desc, "status", None)
    if status is None:
        return False
    if isinstance(status, dict):
        return bool(status.get("ready", False))
    return bool(getattr(status, "ready", False))


def wait_for_index_ready(client: Pinecone, index_name: str, timeout_seconds: int = 180) -> object:
    start = time.time()
    while True:
        desc = retry_call("describe_index", lambda: client.describe_index(index_name), attempts=3)
        if index_ready(desc):
            return desc
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for index '{index_name}' to become ready")
        time.sleep(2)


def ensure_index(client: Pinecone, index_name: str, embed_model: str, cloud: str, region: str) -> object:
    names = retry_call("list_indexes", lambda: client.list_indexes().names(), attempts=3)
    if index_name not in names:
        print(f"Creating integrated embedding index: {index_name}")
        retry_call(
            "create_index_for_model",
            lambda: client.create_index_for_model(
                name=index_name,
                cloud=cloud,
                region=region,
                embed={"model": embed_model, "field_map": {"text": "text"}},
            ),
            attempts=3,
        )
    desc = wait_for_index_ready(client, index_name)

    configured_model = resolve_model_from_index_description(desc)
    if configured_model and configured_model != embed_model:
        raise RuntimeError(
            f"Index '{index_name}' uses embed model '{configured_model}', expected '{embed_model}'. "
            "Create a separate index for this embedding model."
        )

    if not configured_model:
        print("Warning: unable to confirm embed model from index description; continuing.")

    return desc


def build_records(pdf_files: List[Path], chunk_size: int, chunk_overlap: int) -> Iterable[Dict[str, object]]:
    for pdf_path in pdf_files:
        rel_source = str(pdf_path)
        filename = pdf_path.name
        pages = extract_pdf_pages(pdf_path)
        for page_number, page_text in pages:
            chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=chunk_overlap)
            for chunk_number, chunk in enumerate(chunks, start=1):
                record_id = deterministic_record_id(rel_source, page_number, chunk_number, chunk)
                yield {
                    "_id": record_id,
                    "text": chunk,
                    "source": rel_source,
                    "filename": filename,
                    "page": page_number,
                    "chunk_number": chunk_number,
                }


def maybe_clear_namespace(index, namespace: str, clear_namespace: bool):
    if not clear_namespace:
        return
    print(f"Clearing namespace '{namespace}' before re-ingestion")
    retry_call("delete_namespace_records", lambda: index.delete(delete_all=True, namespace=namespace), attempts=5)


def run() -> int:
    api_key = env("PINECONE_API_KEY")
    if is_placeholder(api_key):
        print("Error: PINECONE_API_KEY is missing or placeholder.", file=sys.stderr)
        return 2

    index_name = env("PINECONE_INDEX_NAME", "medical-chatbot-v2")
    namespace = env("PINECONE_NAMESPACE", "medical-book")
    embed_model = env("PINECONE_EMBED_MODEL", "llama-text-embed-v2")
    data_dir = env("DATA_DIR", "data")
    cloud = env("PINECONE_CLOUD", "aws")
    region = env("PINECONE_REGION", "us-east-1")
    batch_size = max(1, min(int(env("UPSERT_BATCH_SIZE", "64") or "64"), 200))
    chunk_size = max(200, int(env("CHUNK_SIZE", "900") or "900"))
    chunk_overlap = max(0, int(env("CHUNK_OVERLAP", "120") or "120"))
    clear_namespace = (env("CLEAR_NAMESPACE", "false") or "false").lower() == "true"

    pdf_files = list_pdf_files(data_dir)
    if not pdf_files:
        print(f"Error: no PDF files found in {data_dir}", file=sys.stderr)
        return 3

    summary = IngestSummary()

    try:
        client = Pinecone(api_key=api_key)
        ensure_index(client, index_name, embed_model, cloud, region)
        index = client.Index(index_name)

        maybe_clear_namespace(index, namespace, clear_namespace)

        # Build records once to compute summary deterministically.
        record_buffer: List[Dict[str, object]] = []
        for pdf_path in pdf_files:
            pages = extract_pdf_pages(pdf_path)
            summary.pdfs_processed += 1
            summary.pages_processed += len(pages)
            for page_number, page_text in pages:
                chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=chunk_overlap)
                for chunk_number, chunk in enumerate(chunks, start=1):
                    record_buffer.append(
                        {
                            "_id": deterministic_record_id(str(pdf_path), page_number, chunk_number, chunk),
                            "text": chunk,
                            "source": str(pdf_path),
                            "filename": pdf_path.name,
                            "page": page_number,
                            "chunk_number": chunk_number,
                        }
                    )
                    summary.chunks_generated += 1

        if not record_buffer:
            raise RuntimeError("No text chunks generated from provided PDFs.")

        for batch in batched(record_buffer, batch_size):
            retry_call(
                "upsert_records",
                lambda b=batch: index.upsert_records(namespace=namespace, records=b),
                attempts=5,
            )
            summary.records_uploaded += len(batch)

        print("\nIngestion completed successfully.")
        print(f"PDFs processed: {summary.pdfs_processed}")
        print(f"Pages processed: {summary.pages_processed}")
        print(f"Chunks generated: {summary.chunks_generated}")
        print(f"Records uploaded: {summary.records_uploaded}")
        print(f"Index name: {index_name}")
        print(f"Namespace: {namespace}")
        print(f"Embedding model: {embed_model}")
        return 0
    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run())
