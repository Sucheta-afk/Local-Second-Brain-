"""
watcher.py — Watch /data for new/changed files and auto-reindex.
Run in background: python src/watcher.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from rich.console import Console

import config

console = Console()


class ReindexHandler(FileSystemEventHandler):
    def __init__(self):
        self._pending = False

    def on_any_event(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() in config.SUPPORTED_EXTENSIONS:
            console.print(f"[yellow]Change detected: {path.name}[/yellow]")
            self._pending = True

    def flush_if_pending(self):
        if self._pending:
            self._pending = False
            _reindex()


def _reindex():
    import numpy as np
    from ingestion import ingest, save_chunks
    from embedder import embed_texts
    from vector_store import build_index, save_index

    console.rule("Auto-reindexing…")
    chunks = ingest(config.DATA_DIR)
    if not chunks:
        return
    save_chunks(chunks)
    vecs = embed_texts([c.text for c in chunks])
    np.save(str(config.EMBEDDINGS_DIR / "vectors.npy"), vecs)
    idx = build_index(vecs)
    save_index(idx)
    console.print(f"[green]✓ Reindexed {len(chunks)} chunks[/green]")


if __name__ == "__main__":
    handler = ReindexHandler()
    observer = Observer()
    observer.schedule(handler, str(config.DATA_DIR), recursive=True)
    observer.start()
    console.print(f"[dim]Watching {config.DATA_DIR} for changes…  Ctrl+C to stop[/dim]")
    try:
        while True:
            time.sleep(5)
            handler.flush_if_pending()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
