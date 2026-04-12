"""
reflection.py — Daily intelligence loop.
Run on a cron or manually: python src/reflection.py
Appends a timestamped reflection to data/reflections.md
"""
import json
from datetime import datetime
from pathlib import Path

from rich.console import Console

import config
from agent import SecondBrainAgent

console = Console()
REFLECTIONS_FILE = config.DATA_DIR / "reflections.md"


def run_reflection():
    console.rule("[bold]Daily Reflection[/bold]")
    agent = SecondBrainAgent()

    console.print("[dim]Generating reflection…[/dim]")
    reflection = agent.reflect()

    console.print("[dim]Generating curiosity questions…[/dim]")
    questions = agent.ask_curiosity_questions()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"""
## {timestamp}

### Reflection
{reflection}

### Open Questions
{questions}

---
"""
    with open(REFLECTIONS_FILE, "a", encoding="utf-8") as f:
        f.write(entry)

    console.print(f"[green]✓ Reflection appended to {REFLECTIONS_FILE}[/green]")
    console.print()
    console.print(reflection)
    console.print()
    console.print(questions)


if __name__ == "__main__":
    run_reflection()
