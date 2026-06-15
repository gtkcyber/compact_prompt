"""Compact and review markdown files and Claude Code skills.

This is the file layer on top of the string-based engines. It compacts the
*body* of markdown/text files while:

* preserving YAML frontmatter verbatim (skills' ``name``/``description`` drive
  triggering),
* never feeding fenced code blocks to a lossy engine,
* validating structure after compaction and **reverting** the file if any
  heading, code block, URL, or inline-code span was lost, and
* writing nothing unless ``apply=True`` (and then only after a verified backup).

It also provides an analytical :func:`review_file` that reports compaction
opportunities (filler, repetition, oversized frontmatter) without rewriting.

The natural-language detection and sensitive-path refusal are adapted from the
Caveman project by Julius Brussee (MIT); see ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Union

from .markdown import (
    extract_code_blocks,
    extract_headings,
    prose_segments,
    split_frontmatter,
    validate_structure,
)
from .ngram import NgramAbbreviator
from .tokens import count_tokens, simple_word_tokens

# --- file classification (adapted from caveman's detect.py) ------------------
COMPRESSIBLE_EXTENSIONS = {".md", ".markdown", ".txt", ".rst", ".typ", ".tex"}
SKIP_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml", ".toml",
    ".lock", ".css", ".scss", ".html", ".xml", ".sql", ".sh", ".bash", ".zsh",
    ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".swift",
    ".kt", ".lua", ".csv", ".ini", ".cfg", ".env",
}
_CODE_LINE_RE = re.compile(
    r"^\s*(import |from .+ import |const |let |var |def |class |function |"
    r"export |if\s*\(|for\s*\(|while\s*\(|@\w+|[\}\]\);]+\s*$)"
)
_HEADING_LINE_RE = re.compile(r"^\s{0,3}#{1,6}\s")

# --- sensitive-path refusal (adapted from caveman's compress.py) -------------
_SENSITIVE_BASENAME_RE = re.compile(
    r"(?ix)^("
    r"\.env(\..+)?|\.netrc|credentials(\..+)?|secrets?(\..+)?|passwords?(\..+)?"
    r"|id_(rsa|dsa|ecdsa|ed25519)(\.pub)?|authorized_keys|known_hosts"
    r"|.*\.(pem|key|p12|pfx|crt|cer|jks|keystore|asc|gpg))$"
)
_SENSITIVE_PATH_COMPONENTS = frozenset({".ssh", ".aws", ".gnupg", ".kube", ".docker"})
_SENSITIVE_NAME_TOKENS = (
    "secret", "credential", "password", "passwd", "apikey", "accesskey",
    "token", "privatekey",
)

MAX_FILE_SIZE = 1_000_000  # 1 MB; larger files are skipped for safety
_FRONTMATTER_WARN_TOKENS = 200

# Common function/filler words. Their density is an indicator of verbosity; it is
# not a claim that all of them are removable (much function-word use is needed).
_FILLER_WORDS = frozenset("""
a an the this that these those of to in on at by for with from into over under
and or but so as if then than is are was were be been being am do does did
have has had will would shall should can could may might must i you he she it we
they me him her us them my your his its our their
please kindly really very just actually basically simply quite rather somewhat
honestly literally essentially truly definitely certainly perhaps maybe
go ahead want need make sure able order provide given using use used like also
that's there here when which who whom whose what why how
""".split())


@dataclass
class FileResult:
    """Outcome of compacting a single file."""

    path: Path
    original: str
    compressed: str
    tokens_before: int
    tokens_after: int
    applied: bool = False
    backup_path: Optional[Path] = None
    skipped: bool = False
    skip_reason: str = ""
    validation_errors: List[str] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        return self.tokens_before / self.tokens_after if self.tokens_after else 1.0

    @property
    def savings(self) -> float:
        if not self.tokens_before:
            return 0.0
        return 1.0 - self.tokens_after / self.tokens_before


@dataclass
class ReviewReport:
    """Analytical review of a single file (no rewriting)."""

    path: Path
    tokens: int
    n_headings: int
    n_code_blocks: int
    frontmatter_tokens: int
    est_filler_pct: float
    est_repetition_pct: float
    est_savings: float
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


# --- classification ----------------------------------------------------------
def detect_file_type(path: Path) -> str:
    """Classify a file as ``natural_language``, ``code``, ``config`` or ``unknown``."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in COMPRESSIBLE_EXTENSIONS:
        return "natural_language"
    if ext in SKIP_EXTENSIONS:
        config_exts = {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env"}
        return "config" if ext in config_exts else "code"
    if ext:
        return "unknown"
    # Extensionless: decide from content (e.g. a "TODO" or "Dockerfile").
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return "unknown"
    lines = [ln for ln in text.splitlines()[:50] if ln.strip()]
    if not lines:
        return "natural_language"
    code_lines = sum(1 for ln in lines if _CODE_LINE_RE.match(ln))
    return "code" if code_lines / len(lines) > 0.4 else "natural_language"


def is_compactable(path: Path) -> bool:
    """True if ``path`` is a natural-language file worth compacting."""
    path = Path(path)
    if not path.is_file():
        return False
    if path.name.endswith((".bak", ".original.md")):
        return False
    return detect_file_type(path) == "natural_language"


def is_sensitive_path(path: Path) -> bool:
    """True if a file looks like it holds secrets/PII and must never be sent out."""
    path = Path(path)
    if _SENSITIVE_BASENAME_RE.match(path.name):
        return True
    if {p.lower() for p in path.parts} & _SENSITIVE_PATH_COMPONENTS:
        return True
    lowered = re.sub(r"[_\-\s.]", "", path.name.lower())
    return any(tok in lowered for tok in _SENSITIVE_NAME_TOKENS)


# --- compaction --------------------------------------------------------------
def _precheck(path: Path, original: str) -> Optional[str]:
    """Return a skip reason for ``path``, or ``None`` if it is safe to compact."""
    if not path.is_file():
        return "file not found"
    if path.stat().st_size > MAX_FILE_SIZE:
        return "file too large"
    if is_sensitive_path(path):
        return "looks sensitive (credentials/keys)"
    if not is_compactable(path):
        return "not a natural-language file"
    if not original.strip():
        return "empty file"
    return None


def _compact_prose(text: str, engine: str, ratio: float, abbreviate: bool) -> str:
    """Compact prose while leaving heading lines untouched (keeps headings exact)."""
    from .pipeline import CompactPrompt  # local import avoids any import-order issues

    out: List[str] = []
    buf: List[str] = []

    def flush() -> None:
        if not buf:
            return
        chunk = "\n".join(buf)
        buf.clear()
        if chunk.strip():
            res = CompactPrompt.compact(
                chunk, engine=engine, ratio=ratio, abbreviate=abbreviate
            )
            out.append(res.text)
        else:
            out.append(chunk)

    for line in text.split("\n"):
        if _HEADING_LINE_RE.match(line):
            flush()
            out.append(line)
        else:
            buf.append(line)
    flush()
    return "\n".join(out)


def compact_text_for_file(
    body: str,
    *,
    engine: str,
    ratio: float = 0.5,
    budget: Optional[int] = None,
    abbreviate: bool = False,
    llm=None,
):
    """Compact a markdown ``body`` structure-safely.

    Caveman receives the whole body (it preserves structure itself); lossy
    engines compact only prose segments, with code blocks and headings left
    verbatim. Returns ``(new_body, validation_errors)``; a non-empty error list
    means the caller should keep the original.
    """
    from .pipeline import CompactPrompt

    if engine == "caveman":
        pruner = None
        if llm is not None:
            from .caveman import CavemanCompressor

            pruner = CavemanCompressor(llm=llm)
        try:
            res = CompactPrompt.compact(
                body, engine="caveman", pruner=pruner, abbreviate=abbreviate
            )
            new_body = res.text
        except ValueError as exc:  # caveman could not preserve structure
            return body, [str(exc)]
    else:
        eff_ratio = ratio
        if budget is not None:
            total = count_tokens(body)
            eff_ratio = max(0.0, min(0.95, 1.0 - budget / total)) if total else 0.0
        parts: List[str] = []
        for kind, seg in prose_segments(body):
            if kind == "code" or not seg.strip():
                parts.append(seg)
            else:
                parts.append(_compact_prose(seg, engine, eff_ratio, abbreviate))
        new_body = "\n".join(parts)

    return new_body, validate_structure(body, new_body)


def compact_file(
    path: Union[str, Path],
    *,
    engine: str,
    ratio: float = 0.5,
    budget: Optional[int] = None,
    abbreviate: bool = False,
    include_frontmatter: bool = False,
    apply: bool = False,
    backup: bool = True,
    output: Optional[str] = None,
    llm: Optional[Callable[[str], str]] = None,
) -> FileResult:
    """Compact a single markdown/text file.

    Args:
        path: The file to compact.
        engine: Required — ``"builtin"``, ``"llmlingua"``, or ``"caveman"``.
        ratio: How aggressively to compact (see :class:`CompactPrompt`).
        budget: Target token count instead of a ratio.
        abbreviate: Also apply reversible n-gram abbreviation (advanced).
        include_frontmatter: Also compact YAML frontmatter (off by default).
        apply: Write the result. When ``False`` (default) this is a dry run.
        backup: When applying in place, first write ``<file>.bak``.
        output: Write the result to this directory instead of in place.
        llm: Pluggable LLM callable for the ``caveman`` engine.

    Returns:
        A :class:`FileResult`. The file is left untouched if compaction would
        break its structure, if it is sensitive, or if it is not prose.
    """
    path = Path(path)
    original = path.read_text(encoding="utf-8", errors="ignore") if path.is_file() else ""

    reason = _precheck(path, original)
    if reason is not None:
        toks = count_tokens(original)
        return FileResult(path, original, original, toks, toks, skipped=True, skip_reason=reason)

    if include_frontmatter:
        new_text, errors = compact_text_for_file(
            original, engine=engine, ratio=ratio, budget=budget,
            abbreviate=abbreviate, llm=llm,
        )
    else:
        frontmatter, body = split_frontmatter(original)
        new_body, errors = compact_text_for_file(
            body, engine=engine, ratio=ratio, budget=budget,
            abbreviate=abbreviate, llm=llm,
        )
        new_text = frontmatter + new_body

    result = FileResult(
        path=path,
        original=original,
        compressed=new_text,
        tokens_before=count_tokens(original),
        tokens_after=count_tokens(new_text),
        validation_errors=errors,
    )

    if errors:
        result.compressed = original
        result.tokens_after = result.tokens_before
        result.skipped = True
        result.skip_reason = "would break structure: " + "; ".join(errors)
        return result
    if new_text == original:
        result.skipped = True
        result.skip_reason = "no change"
        return result

    if apply:
        _write_result(path, original, new_text, backup, output, result)
    return result


def _write_result(path, original, new_text, backup, output, result) -> None:
    """Persist a compacted file, either to an output dir or in place with backup."""
    if output is not None:
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / path.name
        target.write_text(new_text, encoding="utf-8")
        result.applied = True
        return
    if backup:
        backup_path = path.with_name(path.name + ".bak")
        backup_path.write_text(original, encoding="utf-8")
        if backup_path.read_text(encoding="utf-8", errors="ignore") != original:  # pragma: no cover
            backup_path.unlink(missing_ok=True)
            result.skipped = True
            result.skip_reason = "backup verification failed; original untouched"
            return
        result.backup_path = backup_path
    path.write_text(new_text, encoding="utf-8")
    result.applied = True


def compact_directory(path, *, engine: str, glob: str = "**/*.md", **kwargs) -> List[FileResult]:
    """Compact every matching file under ``path`` (see :func:`compact_file`)."""
    base = Path(path)
    results: List[FileResult] = []
    for candidate in sorted(base.glob(glob)):
        if candidate.is_file():
            results.append(compact_file(candidate, engine=engine, **kwargs))
    return results


# --- review ------------------------------------------------------------------
def _estimate_filler(prose: str) -> float:
    """Density of common function/filler words (a verbosity indicator, 0-1)."""
    words = [w.lower() for w in simple_word_tokens(prose) if w.isalpha()]
    if not words:
        return 0.0
    return sum(1 for w in words if w in _FILLER_WORDS) / len(words)


def _estimate_repetition(prose: str) -> float:
    total = count_tokens(prose)
    if not total:
        return 0.0
    abbr = NgramAbbreviator(n=3, top_k=200).compress(prose)
    saved = total - count_tokens(abbr.text)
    return max(0.0, saved / total)


def _llm_suggestions(body: str, llm) -> List[str]:
    prompt = (
        "Review this markdown for concision. List 2-4 short, concrete suggestions "
        "for what to tighten or remove, one per line, without rewriting it:\n\n" + body
    )
    try:
        text = llm(prompt)
    except Exception:  # pragma: no cover - defensive around external LLM
        return []
    return [ln.strip("-* ").strip() for ln in text.splitlines() if ln.strip()][:4]


def review_file(path, *, llm=None) -> ReviewReport:
    """Analyse a markdown/text file and report compaction opportunities."""
    path = Path(path)
    original = path.read_text(encoding="utf-8", errors="ignore") if path.is_file() else ""
    frontmatter, body = split_frontmatter(original)
    prose = " ".join(seg for kind, seg in prose_segments(body) if kind == "prose")

    filler = _estimate_filler(prose)
    repetition = _estimate_repetition(prose)
    fm_tokens = count_tokens(frontmatter)

    issues: List[str] = []
    if not is_compactable(path):
        issues.append("Not a natural-language file; compaction is not recommended.")
    if fm_tokens > _FRONTMATTER_WARN_TOKENS:
        issues.append(
            f"Large frontmatter ({fm_tokens} tokens) — consider trimming the description."
        )
    if repetition > 0.10:
        issues.append(f"Repetitive phrasing (~{repetition:.0%} recoverable losslessly).")
    if filler > 0.55:
        issues.append(f"Verbose wording (~{filler:.0%} function/filler words).")

    # Conservative: lossless repetition plus a portion of above-baseline verbosity.
    est_savings = min(0.9, repetition + 0.4 * max(0.0, filler - 0.35))

    report = ReviewReport(
        path=path,
        tokens=count_tokens(original),
        n_headings=len(extract_headings(body)),
        n_code_blocks=len(extract_code_blocks(body)),
        frontmatter_tokens=fm_tokens,
        est_filler_pct=filler,
        est_repetition_pct=repetition,
        est_savings=est_savings,
        issues=issues,
    )
    if llm is not None:
        report.suggestions = _llm_suggestions(body, llm)
    return report


def review_directory(path, glob: str = "**/*.md", *, llm=None) -> List[ReviewReport]:
    """Review every matching file under ``path`` (see :func:`review_file`)."""
    base = Path(path)
    return [
        review_file(p, llm=llm)
        for p in sorted(base.glob(glob))
        if p.is_file()
    ]
