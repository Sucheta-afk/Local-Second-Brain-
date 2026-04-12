#!/usr/bin/env python3
"""
app.py — CLI entry point for your Local Second Brain.

Usage:
  python app.py              # interactive chat loop
  python app.py --ideas      # generate project ideas
  python app.py --connect    # find connections between notes
  python app.py --reflect    # daily reflection summary
  python app.py --search "query"  # raw similarity search
  python app.py --reindex    # re-run full ingestion + embedding pipeline
"""
import sys
import click
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))


def _ensure_index():
    import config
    if not config.FAISS_INDEX_PATH.exists():
        console.print("[yellow]No index found — running full pipeline…[/yellow]")
        _run_pipeline()


def _run_pipeline():
    import config
    from ingestion import ingest, save_chunks
    from embedder import embed_texts
    from vector_store import build_index, save_index
    import numpy as np

    chunks = ingest(config.DATA_DIR)
    if not chunks:
        return False
    save_chunks(chunks)

    texts = [c.text for c in chunks]
    console.print(f"Embedding {len(texts)} chunks…")
    vectors = embed_texts(texts)

    np.save(str(config.EMBEDDINGS_DIR / "vectors.npy"), vectors)
    index = build_index(vectors)
    save_index(index)
    console.print(f"[green]✓ Pipeline complete — {len(chunks)} chunks indexed[/green]")
    return True


@click.command()
@click.option("--ideas", is_flag=True, help="Generate project ideas from your notes")
@click.option("--connect", is_flag=True, help="Find connections between notes")
@click.option("--reflect", is_flag=True, help="Generate daily reflection")
@click.option("--questions", is_flag=True, help="Generate curiosity questions")
@click.option("--search", "search_query", default=None, help="Raw similarity search")
@click.option("--reindex", is_flag=True, help="Re-run full ingestion + indexing pipeline")
def main(ideas, connect, reflect, questions, search_query, reindex):
    console.print(Panel(
        Text("🧠 Local Second Brain", style="bold white"),
        subtitle="powered by Gemma + FAISS",
        border_style="dim",
    ))

    if reindex:
        _run_pipeline()
        return

    _ensure_index()

    from agent import SecondBrainAgent
    agent = SecondBrainAgent()

    if ideas:
        console.rule("[bold]Project Ideas[/bold]")
        console.print(agent.generate_ideas())
        return

    if connect:
        console.rule("[bold]Connections[/bold]")
        console.print(agent.find_connections())
        return

    if reflect:
        console.rule("[bold]Daily Reflection[/bold]")
        console.print(agent.reflect())
        return

    if questions:
        console.rule("[bold]Curiosity Questions[/bold]")
        console.print(agent.ask_curiosity_questions())
        return

    if search_query:
        agent.search_and_show(search_query)
        return

    # ── interactive chat loop ─────────────────────────────────────────────────
    console.print("[dim]Type your question. Commands: /ideas  /connect  /reflect  /quit[/dim]\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            console.print("[dim]Bye.[/dim]")
            break
        elif user_input == "/ideas":
            console.print(agent.generate_ideas())
        elif user_input == "/connect":
            console.print(agent.find_connections())
        elif user_input == "/reflect":
            console.print(agent.reflect())
        elif user_input == "/questions":
            console.print(agent.ask_curiosity_questions())
        else:
            console.print("[bold cyan]Brain:[/bold cyan]", end=" ")
            agent.chat(user_input)


if __name__ == "__main__":
    main()
