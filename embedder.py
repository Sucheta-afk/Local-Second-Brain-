"""
embedder.py — Generate sentence embeddings for all chunks.
Uses sentence-transformers (CPU-friendly, ~90MB model).
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from rich.console import Console

import config
from ingestion import Chunk, load_chunks

console = Console()
_model = SentenceTransformer("./models/all-MiniLM-L6-v2")

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        console.print(f"[dim]Loading embedding model: {config.EMBEDDING_MODEL}…[/dim]")
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine sim == dot product
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    model = get_model()
    vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    return vec.astype("float32")


if __name__ == "__main__":
    chunks = load_chunks()
    if not chunks:
        console.print("[red]No chunks found. Run ingestion.py first.[/red]")
    else:
        texts = [c.text for c in chunks]
        console.print(f"Embedding {len(texts)} chunks…")
        vectors = embed_texts(texts)
        # Save raw vectors so vector_store.py can build the index
        out = config.EMBEDDINGS_DIR / "vectors.npy"
        np.save(str(out), vectors)
        console.print(f"[green]✓ Saved {vectors.shape} embeddings → {out}[/green]")
