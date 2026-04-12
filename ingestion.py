"""
ingestion.py — Load, clean, and chunk documents from /data
"""
import json
import re
from pathlib import Path
from typing import Generator
from dataclasses import dataclass, field, asdict

import tiktoken
from pypdf import PdfReader
from rich.console import Console
from rich.progress import track

import config

console = Console()
enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    text: str
    source: str          # relative path from DATA_DIR
    chunk_index: int
    token_count: int
    file_type: str
    metadata: dict = field(default_factory=dict)


# ── readers ──────────────────────────────────────────────────────────────────

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    return read_text(path)


# ── cleaning ─────────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    # collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # strip null bytes and weird control chars
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


# ── chunking ─────────────────────────────────────────────────────────────────

def token_count(text: str) -> int:
    return len(enc.encode(text))


def split_into_chunks(text: str, source: str, file_type: str) -> list[Chunk]:
    """Sliding window token-based chunking."""
    tokens = enc.encode(text)
    chunks = []
    start = 0
    idx = 0

    while start < len(tokens):
        end = min(start + config.CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(Chunk(
            text=chunk_text,
            source=source,
            chunk_index=idx,
            token_count=len(chunk_tokens),
            file_type=file_type,
        ))
        start += config.CHUNK_SIZE - config.CHUNK_OVERLAP
        idx += 1

    return chunks


# ── main pipeline ─────────────────────────────────────────────────────────────

def iter_files(data_dir: Path) -> Generator[Path, None, None]:
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in config.SUPPORTED_EXTENSIONS:
            yield path


def ingest(data_dir: Path = config.DATA_DIR) -> list[Chunk]:
    files = list(iter_files(data_dir))
    if not files:
        console.print(f"[yellow]No supported files found in {data_dir}[/yellow]")
        console.print(f"[dim]Drop .txt .md .pdf .py .js etc. files into {data_dir} and re-run.[/dim]")
        return []

    all_chunks: list[Chunk] = []
    for path in track(files, description="Ingesting files…"):
        try:
            raw = read_file(path)
            text = clean(raw)
            if not text:
                continue
            relative = str(path.relative_to(data_dir))
            chunks = split_into_chunks(text, source=relative, file_type=path.suffix.lower())
            all_chunks.extend(chunks)
        except Exception as e:
            console.print(f"[red]  ✗ {path.name}: {e}[/red]")

    console.print(f"[green]✓ {len(files)} files → {len(all_chunks)} chunks[/green]")
    return all_chunks


def save_chunks(chunks: list[Chunk], path: Path = config.METADATA_PATH):
    """Persist chunk metadata (text + source) alongside the FAISS index."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)
    console.print(f"[dim]Metadata saved → {path}[/dim]")


def load_chunks(path: Path = config.METADATA_PATH) -> list[Chunk]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [Chunk(**d) for d in data]


if __name__ == "__main__":
    chunks = ingest()
    if chunks:
        save_chunks(chunks)
