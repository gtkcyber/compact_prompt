"""An MCP server exposing compactprompt's review/compact capabilities.

This is the single, reusable integration point for AI agents (Claude Code,
Codex, Cursor, Gemini, and any other MCP-capable client). It wraps the existing
library API — :func:`~compactprompt.review_file`,
:func:`~compactprompt.compact_file`, and :meth:`CompactPrompt.compact` — so the
logic lives in one place rather than being reimplemented per tool.

Run it (after ``pip install 'compactprompt[mcp]'``)::

    compactprompt-mcp            # stdio MCP server

Then register ``compactprompt-mcp`` as an MCP server in your agent (see
``agent-skills/README.md`` for per-tool configuration).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "compactprompt-mcp needs the MCP SDK. Install it with: "
        "pip install 'compactprompt[mcp]'"
    ) from exc

from . import CompactPrompt, compact_file, review_file
from .tokens import count_tokens as _count_tokens

mcp = FastMCP("compactprompt")


@mcp.tool()
def count_tokens(text: str) -> int:
    """Count the tokens in a piece of text (the unit LLMs bill by)."""
    return _count_tokens(text)


@mcp.tool()
def compact_prompt(prompt: str, ratio: float = 0.5, engine: str = "builtin") -> dict:
    """Shorten a prompt string while preserving meaning.

    Args:
        prompt: The text to compact.
        ratio: Fraction of tokens to remove (0-1).
        engine: ``"builtin"`` (offline), ``"llmlingua"``, or ``"caveman"``.

    Returns:
        ``text`` (the shortened prompt) plus token counts and the ratio.
    """
    result = CompactPrompt.compact(prompt, ratio=ratio, engine=engine)
    return {
        "text": result.text,
        "tokens_before": result.tokens_before,
        "tokens_after": result.tokens_after,
        "ratio": round(result.ratio, 3),
        "savings": round(result.savings, 3),
    }


@mcp.tool()
def review(path: str) -> dict:
    """Review a markdown/text file for compaction opportunities (read-only).

    Args:
        path: A markdown or text file.

    Returns:
        Token count, structure counts, estimated savings, and flagged issues.
    """
    return asdict(_strip_path(review_file(path)))


@mcp.tool()
def compact(
    path: str,
    engine: str,
    apply: bool = False,
    ratio: float = 0.5,
    budget: Optional[int] = None,
) -> dict:
    """Compact a markdown file or skill. Dry run unless ``apply`` is true.

    Frontmatter, code blocks, and links are preserved; if compaction would break
    structure the file is left unchanged and reported as skipped. When ``apply``
    is true a ``.bak`` backup is written before the file is overwritten.

    Args:
        path: The file to compact.
        engine: Required — ``"builtin"``, ``"llmlingua"``, or ``"caveman"``
            (caveman, an LLM rewrite, is best for human-readable docs).
        apply: Write the result. When false (default) this only previews.
        ratio: Fraction of tokens to remove (0-1).
        budget: Target token count instead of a ratio.

    Returns:
        The before/after token counts, status, and (if written) the backup path.
    """
    result = compact_file(path, engine=engine, apply=apply, ratio=ratio, budget=budget)
    return {
        "path": str(result.path),
        "tokens_before": result.tokens_before,
        "tokens_after": result.tokens_after,
        "savings": round(result.savings, 3),
        "applied": result.applied,
        "skipped": result.skipped,
        "skip_reason": result.skip_reason,
        "backup_path": str(result.backup_path) if result.backup_path else None,
        "compressed_preview": result.compressed[:2000],
    }


def _strip_path(report) -> object:
    """Replace the Path field with a string so the report is JSON-serializable."""
    report.path = str(report.path)
    return report


def main() -> None:
    """Entry point for the ``compactprompt-mcp`` console script (stdio server)."""
    mcp.run()


if __name__ == "__main__":
    main()
