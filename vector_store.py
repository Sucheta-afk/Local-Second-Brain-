"""
vector_store.py — Build and query a FAISS index over your embeddings.
"""
import numpy as np
import faiss
from rich.console import Console

import config
from ingestion import Chunk, load_chunks
from embedder import embed_query

console = Console()
_index: faiss.IndexFlatIP | None = None
_chunks: list[Chunk] | None = None


# ── building ──────────────────────────────────────────────────────────────────

def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Inner-product index (cosine, since vecs are normalized)."""
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def save_index(index: faiss.IndexFlatIP, path=config.FAISS_INDEX_PATH):
    faiss.write_index(index, str(path))
    console.print(f"[dim]FAISS index saved → {path}[/dim]")


def load_index(path=config.FAISS_INDEX_PATH) -> faiss.IndexFlatIP:
    return faiss.read_index(str(path))


# ── retrieval ─────────────────────────────────────────────────────────────────

def get_index() -> faiss.IndexFlatIP:
    global _index
    if _index is None:
        if not config.FAISS_INDEX_PATH.exists():
            raise FileNotFoundError("FAISS index not found. Run vector_store.py first.")
        _index = load_index()
    return _index


def get_chunks() -> list[Chunk]:
    global _chunks
    if _chunks is None:
        _chunks = load_chunks()
    return _chunks


def search(query: str, k: int = config.TOP_K) -> list[tuple[Chunk, float]]:
    """Return top-k (chunk, score) pairs for a query string."""
    index = get_index()
    chunks = get_chunks()
    query_vec = embed_query(query)

    scores, indices = index.search(query_vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or score < config.SIMILARITY_THRESHOLD:
            continue
        results.append((chunks[idx], float(score)))
    return results


def format_context(results: list[tuple[Chunk, float]]) -> str:
    """Build a context string for the LLM prompt."""
    parts = []
    for chunk, score in results:
        parts.append(
            f"[Source: {chunk.source} | chunk {chunk.chunk_index} | score {score:.2f}]\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    vectors_path = config.EMBEDDINGS_DIR / "vectors.npy"
    if not vectors_path.exists():
        console.print("[red]vectors.npy not found. Run embedder.py first.[/red]")
    else:
        vectors = np.load(str(vectors_path))
        index = build_index(vectors)
        save_index(index)
        console.print(f"[green]✓ FAISS index built with {index.ntotal} vectors[/green]")
