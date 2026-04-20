"""
agent.py — Orchestrates retrieval + LLM for all query types.
"""
from rich.console import Console

import config
from vector_store import search, format_context
from llm import (
    stream_response, complete,
    IDEA_PROMPT, CONNECTIONS_PROMPT, REFLECTION_PROMPT, CURIOSITY_PROMPT,
)

console = Console()


class SecondBrainAgent:
    def __init__(self):
        self.history: list[dict] = []

    def _retrieve(self, query: str, k: int = config.TOP_K) -> str:
        results = search(query, k=k)
        if not results:
            return "[No relevant context found in your knowledge base.]"
        return format_context(results)

    def _record(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    # ── public methods ────────────────────────────────────────────────────────

    def chat(self, query: str) -> str:
        """RAG Q&A with streaming output to console."""
        context = self._retrieve(query)
        full_response = []
        console.print()
        for token in stream_response(query, context, self.history):
            print(token, end="", flush=True)
            full_response.append(token)
        print()
        response = "".join(full_response)
        self._record("user", query)
        self._record("assistant", response)
        return response

    def generate_ideas(self) -> str:
        """Brainstorm project ideas based on entire knowledge base sample."""
        # Broad query to pull diverse context
        context = self._retrieve("projects ideas code experiments research", k=8)
        prompt = IDEA_PROMPT.format(context=context)
        return complete(prompt, context="")

    def find_connections(self) -> str:
        context = self._retrieve("concepts topics research notes", k=6)
        prompt = CONNECTIONS_PROMPT.format(context=context)
        return complete(prompt, context="")

    def reflect(self, extra_context: str = "") -> str:
        context = self._retrieve("today's work learning notes", k=5)
        combined = f"{extra_context}\n\n{context}".strip() if extra_context else context
        prompt = REFLECTION_PROMPT.format(context=combined)
        return complete(prompt, context="")

    def ask_curiosity_questions(self, extra_context: str = "") -> str:
        # Try multiple queries and merge results
        queries = [
            "notes ideas research",
            "learning concepts topics",
            "work code projects",
        ]
        from vector_store import search, format_context
        seen = set()
        results = []
        for q in queries:
            for chunk, score in search(q, k=3):
                if chunk.chunk_index not in seen:
                    seen.add(chunk.chunk_index)
                    results.append((chunk, score))

        if not results:
            # Nuclear fallback — just grab the first N chunks directly
            from ingestion import load_chunks
            chunks = load_chunks()[:6]
            results = [(c, 1.0) for c in chunks]

        context = format_context(results)
        combined = f"{extra_context}\n\n{context}".strip() if extra_context else context
        prompt = CURIOSITY_PROMPT.format(context=combined)
        return complete(prompt, context="")         # ← match other methods

    def search_and_show(self, query: str, k: int = 5):
        """Raw retrieval — show matching chunks without LLM."""
        results = search(query, k=k)
        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return
        for chunk, score in results:
            console.rule(f"[dim]{chunk.source} · chunk {chunk.chunk_index} · score {score:.3f}[/dim]")
            console.print(chunk.text[:500] + ("…" if len(chunk.text) > 500 else ""))
        console.rule()