"""
llm.py — Thin wrapper around Ollama for local Gemma inference.
Streams tokens back so the CLI feels snappy.
"""
from typing import Generator
import ollama
from rich.console import Console

import config

console = Console()


SYSTEM_PROMPT = """You are a personal second brain — a local AI assistant that 
thinks through the user's own notes, code, and documents. 

Rules:
- Ground every answer in the provided context snippets.
- If the context doesn't contain enough info, say so honestly.
- Be concise but insightful. Surface non-obvious connections.
- When asked to generate ideas, relate them to the user's actual work.
- Never hallucinate sources or facts not present in the context."""


def _build_messages(query: str, context: str, history: list[dict] | None = None) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-6:])  # last 3 turns
    user_content = f"""Context from your knowledge base:
{context}

---
Question: {query}"""
    messages.append({"role": "user", "content": user_content})
    return messages


def stream_response(
    query: str,
    context: str,
    history: list[dict] | None = None,
    model: str = config.LLM_MODEL,
) -> Generator[str, None, None]:
    """Yields text tokens as they arrive from Ollama."""
    messages = _build_messages(query, context, history)
    try:
        stream = ollama.chat(
            model=model,
            messages=messages,
            stream=True,
            options={"temperature": 0.7, "num_ctx": 4096},
        )
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield token
    except ollama.ResponseError as e:
        if "not found" in str(e).lower():
            yield f"\n[Error] Model '{model}' not found. Run: ollama pull {model}\n"
        else:
            yield f"\n[Error] Ollama error: {e}\n"
    except Exception as e:
        yield f"\n[Error] Could not connect to Ollama. Is it running? Start with: ollama serve\nDetail: {e}\n"


def complete(query: str, context: str, history: list[dict] | None = None) -> str:
    """Blocking version — returns full response string."""
    return "".join(stream_response(query, context, history))


# ── special prompts ───────────────────────────────────────────────────────────

IDEA_PROMPT = """Given the following notes and code from my knowledge base, 
generate 3-5 original project ideas that build on my existing work. 
Be specific — name the technologies/topics I've already explored.

Context:
{context}"""

CONNECTIONS_PROMPT = """Analyse the following passages from my notes.
Find 2-3 non-obvious conceptual connections between them.
Format: Connection → [topic A] ↔ [topic B]: one-sentence explanation.

Context:
{context}"""

REFLECTION_PROMPT = """You are summarising today's knowledge activity.
Given these recent notes/chunks, produce:
1. A 3-sentence summary of what I explored.
2. The 3 most important insights.
3. One open question worth pursuing.

Context:
{context}"""

CURIOSITY_PROMPT = """Based on my knowledge base context below, ask me 3 probing 
questions that would help me deepen my understanding or surface unfinished threads. 
Each question should reference a specific topic or project visible in the context.

Context:
{context}"""
