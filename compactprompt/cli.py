"""Command-line interface for compacting and reviewing markdown files / skills.

Two subcommands::

    compactprompt review PATH
    compactprompt compact PATH --engine builtin [--apply]

``review`` is read-only. ``compact`` is a dry run unless ``--apply`` is given.
``--engine`` is required (there is intentionally no default engine).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .files import (
    FileResult,
    ReviewReport,
    compact_directory,
    compact_file,
    review_directory,
    review_file,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compactprompt",
        description="Compact and review markdown files and Claude Code skills.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    rev = sub.add_parser("review", help="Analyse files for compaction opportunities.")
    rev.add_argument("path", help="A markdown/text file or a directory.")
    rev.add_argument("--glob", default="**/*.md", help="Glob for directories.")
    rev.add_argument("--llm", action="store_true", help="Add LLM-written suggestions.")
    rev.add_argument("--json", action="store_true", help="Emit JSON.")

    comp = sub.add_parser("compact", help="Compact files (dry run unless --apply).")
    comp.add_argument("path", help="A markdown/text file or a directory.")
    comp.add_argument(
        "--engine", required=True, choices=["builtin", "llmlingua", "caveman"],
        help="Compression engine (required; no default).",
    )
    comp.add_argument("--glob", default="**/*.md", help="Glob for directories.")
    target = comp.add_mutually_exclusive_group()
    target.add_argument("--ratio", type=float, default=0.5, help="Fraction of tokens to remove.")
    target.add_argument("--budget", type=int, help="Target token count instead of a ratio.")
    comp.add_argument("--abbreviate", action="store_true", help="Also abbreviate n-grams.")
    comp.add_argument("--include-frontmatter", action="store_true", help="Compact frontmatter too.")
    comp.add_argument("--apply", action="store_true", help="Write changes (else dry run).")
    comp.add_argument("--no-backup", dest="backup", action="store_false", help="Skip .bak backup.")
    comp.add_argument("--output", help="Write results to this directory instead of in place.")
    comp.add_argument("--json", action="store_true", help="Emit JSON.")
    return parser


def _review_dict(r: ReviewReport) -> dict:
    return {
        "path": str(r.path),
        "tokens": r.tokens,
        "n_headings": r.n_headings,
        "n_code_blocks": r.n_code_blocks,
        "frontmatter_tokens": r.frontmatter_tokens,
        "est_filler_pct": round(r.est_filler_pct, 3),
        "est_repetition_pct": round(r.est_repetition_pct, 3),
        "est_savings": round(r.est_savings, 3),
        "issues": r.issues,
        "suggestions": r.suggestions,
    }


def _cmd_review(args: argparse.Namespace) -> int:
    llm = None
    if args.llm:
        from .caveman import default_anthropic_llm

        llm = default_anthropic_llm()

    path = Path(args.path)
    if path.is_file():
        reports = [review_file(path, llm=llm)]
    elif path.is_dir():
        reports = review_directory(path, args.glob, llm=llm)
    else:
        print(f"error: not found: {path}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps([_review_dict(r) for r in reports], indent=2))
        return 0

    for r in reports:
        print(str(r.path))
        print(
            f"  {r.tokens} tokens | {r.n_headings} headings | "
            f"{r.n_code_blocks} code blocks | frontmatter {r.frontmatter_tokens} tokens"
        )
        print(
            f"  repetition ~{r.est_repetition_pct:.0%} | filler ~{r.est_filler_pct:.0%} | "
            f"est. savings ~{r.est_savings:.0%}"
        )
        for issue in r.issues:
            print(f"  - {issue}")
        for suggestion in r.suggestions:
            print(f"  > {suggestion}")
    return 0


def _cmd_compact(args: argparse.Namespace) -> int:
    kwargs = {
        "engine": args.engine,
        "ratio": args.ratio,
        "budget": args.budget,
        "abbreviate": args.abbreviate,
        "include_frontmatter": args.include_frontmatter,
        "apply": args.apply,
        "backup": args.backup,
        "output": args.output,
    }
    path = Path(args.path)
    try:
        if path.is_file():
            results: List[FileResult] = [compact_file(path, **kwargs)]
        elif path.is_dir():
            results = compact_directory(path, glob=args.glob, **kwargs)
        else:
            print(f"error: not found: {path}", file=sys.stderr)
            return 2
    except ImportError as exc:  # a required engine extra is not installed
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps([_result_dict(r) for r in results], indent=2))
        return 0

    before = after = 0
    for res in results:
        before += res.tokens_before
        after += res.tokens_after
        if res.skipped:
            print(f"SKIP  {res.path}: {res.skip_reason}")
        else:
            label = "WRITE" if res.applied else "DRY  "
            print(
                f"{label} {res.path}: {res.tokens_before} -> "
                f"{res.tokens_after} ({res.savings:.0%} saved)"
            )
            if res.applied and res.backup_path:
                print(f"      backup: {res.backup_path}")
    saved = (1 - after / before) if before else 0.0
    print(f"\nTotal: {before} -> {after} tokens ({saved:.0%} saved)")
    if not args.apply:
        print("Dry run — re-run with --apply to write changes.")
    return 0


def _result_dict(r: FileResult) -> dict:
    return {
        "path": str(r.path),
        "tokens_before": r.tokens_before,
        "tokens_after": r.tokens_after,
        "savings": round(r.savings, 3),
        "applied": r.applied,
        "skipped": r.skipped,
        "skip_reason": r.skip_reason,
        "backup_path": str(r.backup_path) if r.backup_path else None,
    }


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _build_parser().parse_args(argv)
    if args.command == "review":
        return _cmd_review(args)
    if args.command == "compact":
        return _cmd_compact(args)
    return 1  # pragma: no cover - argparse enforces a subcommand
