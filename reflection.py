"""
reflection.py — Daily coding check-in.
Run on a cron or manually: python reflection.py
Appends a timestamped summary to data/reflections.md
"""
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import requests
from rich.console import Console

import config
from agent import SecondBrainAgent

console = Console()
REFLECTIONS_FILE = config.DATA_DIR / "reflections.md"


# ---------------------------------------------------------------------------
# VSCode context collection
# ---------------------------------------------------------------------------

def _vscode_global_storage_path() -> Path | None:
    system = platform.system()
    home = Path.home()
    candidates = {
        "Darwin": home / "Library/Application Support/Code/User/globalStorage",
        "Linux": home / ".config/Code/User/globalStorage",
        "Windows": Path(os.environ.get("APPDATA", "")) / "Code/User/globalStorage",
    }
    path = candidates.get(system)
    return path if path and path.exists() else None


def collect_recently_opened_files() -> list[str]:
    results: list[str] = []
    storage = _vscode_global_storage_path()
    if not storage:
        return results

    db_path = storage / "state.vscdb"
    if db_path.exists():
        try:
            import sqlite3
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cur = con.execute(
                "SELECT value FROM ItemTable WHERE key = 'history.recentlyOpenedPathsList'"
            )
            row = cur.fetchone()
            con.close()
            if row:
                data = json.loads(row[0])
                for e in data.get("entries", []):
                    uri = (
                        e.get("fileUri")
                        or e.get("folderUri")
                        or e.get("workspace", {}).get("configPath", "")
                    )
                    if uri:
                        results.append(uri.replace("file://", ""))
        except Exception as exc:
            console.print(f"[yellow]⚠ Could not read state.vscdb: {exc}[/yellow]")
        return results

    legacy = storage.parent / "recentlyOpened.json"
    if legacy.exists():
        try:
            data = json.loads(legacy.read_text(encoding="utf-8"))
            for e in data.get("entries", []):
                uri = e.get("fileUri") or e.get("folderUri") or ""
                if uri:
                    results.append(uri.replace("file://", ""))
        except Exception as exc:
            console.print(f"[yellow]⚠ Could not read recentlyOpened.json: {exc}[/yellow]")

    return results


def collect_vscode_timeline(workspace_paths: list[Path] | None = None) -> str:
    if workspace_paths is None:
        workspace_paths = [p for p in Path.home().iterdir() if p.is_dir()]

    entries: list[str] = []
    today = datetime.now().date()

    for ws in workspace_paths:
        timeline_dir = ws / ".vscode" / "timeline"
        if not timeline_dir.exists():
            continue
        for jf in sorted(timeline_dir.glob("*.json")):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                for item in data if isinstance(data, list) else [data]:
                    ts = item.get("timestamp", 0)
                    dt = datetime.fromtimestamp(ts / 1000)
                    if dt.date() == today:
                        label = item.get("label") or item.get("description") or jf.stem
                        entries.append(f"- [{dt.strftime('%H:%M')}] {ws.name}/{label}")
            except Exception:
                pass

    return "\n".join(entries)


# ---------------------------------------------------------------------------
# GitHub context collection
# ---------------------------------------------------------------------------

def collect_github_activity(token: str | None = None) -> str:
    token = token or os.environ.get("GITHUB_TOKEN", "")
    if not token:
        console.print("[yellow]  ⚠ GITHUB_TOKEN not set — skipping GitHub activity[/yellow]")
        return ""

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    try:
        me = requests.get("https://api.github.com/user", headers=headers, timeout=10)
        me.raise_for_status()
        username = me.json()["login"]
    except Exception as exc:
        console.print(f"[yellow]  ⚠ GitHub auth failed: {exc}[/yellow]")
        return ""

    today = datetime.now(tz=timezone.utc).date()
    lines: list[str] = []

    try:
        resp = requests.get(
            f"https://api.github.com/users/{username}/events",
            headers=headers,
            params={"per_page": 100},
            timeout=10,
        )
        resp.raise_for_status()
        events = resp.json()
    except Exception as exc:
        console.print(f"[yellow]  ⚠ Could not fetch GitHub events: {exc}[/yellow]")
        return ""

    for event in events:
        created_at = event.get("created_at", "")
        try:
            event_date = datetime.fromisoformat(created_at.replace("Z", "+00:00")).date()
        except ValueError:
            continue
        if event_date != today:
            continue

        repo = event.get("repo", {}).get("name", "unknown/repo")
        etype = event.get("type", "")
        payload = event.get("payload", {})

        if etype == "PushEvent":
            for c in payload.get("commits", []):
                msg = c.get("message", "").splitlines()[0]
                sha = c.get("sha", "")[:7]
                lines.append(f"- **push** `{repo}` [{sha}] {msg}")
        elif etype == "PullRequestEvent":
            pr = payload.get("pull_request", {})
            lines.append(f"- **PR {payload.get('action','')}** `{repo}` — {pr.get('title','')}")
        elif etype == "PullRequestReviewEvent":
            pr = payload.get("pull_request", {})
            state = payload.get("review", {}).get("state", "")
            lines.append(f"- **PR review** `{repo}` [{state}] {pr.get('title','')}")
        elif etype == "IssuesEvent":
            issue = payload.get("issue", {})
            lines.append(f"- **issue {payload.get('action','')}** `{repo}` — {issue.get('title','')}")
        elif etype == "IssueCommentEvent":
            issue = payload.get("issue", {})
            lines.append(f"- **issue comment** `{repo}` on: {issue.get('title','')}")
        elif etype == "CreateEvent":
            lines.append(f"- **created {payload.get('ref_type','')}** `{payload.get('ref') or repo}` in `{repo}`")

    if not lines:
        return ""

    return "## GitHub Activity\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Aggregate context
# ---------------------------------------------------------------------------

def build_work_context() -> str:
    today_str = datetime.now().strftime("%Y-%m-%d")
    parts: list[str] = []

    console.print("[dim]  → recently opened files…[/dim]")
    files = collect_recently_opened_files()
    if files:
        parts.append("## Recently Opened Files\n" + "\n".join(f"- {f}" for f in files[:30]))

    console.print("[dim]  → VSCode timeline…[/dim]")
    timeline = collect_vscode_timeline()
    if timeline:
        parts.append(f"## VSCode Timeline\n{timeline}")

    console.print("[dim]  → GitHub activity…[/dim]")
    github = collect_github_activity()
    if github:
        parts.append(github)

    if not parts:
        return ""

    return f"# Work Context — {today_str}\n\n" + "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Summarise with LLM
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = """You are reviewing a developer's daily activity log. Produce a structured daily work report using ONLY the information present in the context. Do not invent or assume anything not explicitly mentioned.

Use this exact format:

✨ Projects Touched
List each project One line each:
  • <project name> — <one sentence on what was done or which files were opened>

⚙️ What Was Built / Changed
List concrete things that were created, modified, or committed. Pull from file names, commit messages, and timeline entries. Be specific — name the files or features.
  • ...

🏃‍♂️ Momentum
One of: 🟢 Productive | 🟡 Moderate | 🔴 Light
Follow with one sentence explaining why.

📝 Notable Details
Any specific commit messages, PR titles, or file names worth highlighting. If none, write "Nothing notable beyond the above."

Stick strictly to what the context shows. If a section has no data, write "No data available."

Work Context:
{context}
"""


def summarise(context: str) -> str:
    from llm import complete
    prompt = SUMMARY_PROMPT.format(context=context)
    return complete(prompt, context="")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_reflection():
    console.rule("[bold]Daily Coding Check-in[/bold]")

    console.print("[dim]Collecting work context…[/dim]")
    work_context = build_work_context()

    if not work_context:
        console.print("[yellow]⚠ No activity found for today — nothing to summarise.[/yellow]")
        return

    console.print("[green]  ✓ Context collected[/green]")
    console.print("[dim]Summarising today's work…[/dim]")
    summary = summarise(work_context)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"""
## {timestamp}

{summary}

<details>
<summary>Raw work context</summary>

{work_context}

</details>

---
"""
    with open(REFLECTIONS_FILE, "a", encoding="utf-8") as f:
        f.write(entry)

    console.rule("[green]Summary[/green]")
    console.print(summary)
    console.print()
    console.print(f"[dim]✓ Appended to {REFLECTIONS_FILE}[/dim]")


if __name__ == "__main__":
    run_reflection()