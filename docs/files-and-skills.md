# Files & skills

CompactPrompt can compact whole markdown files — documentation, `CLAUDE.md`,
notes, and Claude Code **skills** (`SKILL.md`) — not just strings. It can also
**review** a file or folder first to report where the savings are.

## The safety model

Files are valuable, so compaction follows a strict *validate-or-revert* rule. A
file is never corrupted:

1. **Non-prose files are skipped.** Only `.md`, `.markdown`, `.txt`, `.rst`,
   `.tex`, and natural-language extensionless files are processed; code and
   config are skipped.
2. **Sensitive files are refused.** Anything that looks like credentials, keys,
   or `.env`/`.ssh`/`.aws` content is never sent to an engine.
3. **Frontmatter is preserved verbatim.** A leading YAML block (a skill's
   `name`/`description`) is kept exactly; only the body is compacted.
4. **Code blocks are never fed to lossy engines.**
5. **The result is validated.** If compaction would change a heading, a fenced
   code block, a URL, or an inline-code span, the original is kept and the file
   is reported as skipped.
6. **Nothing is written without `--apply`** (CLI) / `apply=True` (Python), which
   first writes a verified `.bak` backup.

Because lossy engines (built-in, LLMLingua) often trip the validation check on
markdown with inline links and code, **Caveman is usually the best engine for
human-readable files**. There is deliberately no default engine — you choose one
each time.

## Command line

```bash
# Review a folder (read-only)
compactprompt review ./skills

# Preview compacting one skill (writes nothing)
compactprompt compact ./skills/my-skill/SKILL.md --engine builtin

# Apply it (writes SKILL.md.bak, then rewrites the file)
compactprompt compact ./skills/my-skill/SKILL.md --engine caveman --apply
```

### `compactprompt review`

```
compactprompt review PATH [--glob PATTERN] [--llm] [--json]
```

| Option | Meaning |
|--------|---------|
| `PATH` | A file or a directory. |
| `--glob` | Pattern for directories (default `**/*.md`). |
| `--llm` | Add LLM-written suggestions (needs an LLM, as for the Caveman engine). |
| `--json` | Emit machine-readable JSON. |

### `compactprompt compact`

```
compactprompt compact PATH --engine {builtin,llmlingua,caveman}
    [--ratio R | --budget N] [--abbreviate] [--include-frontmatter]
    [--apply] [--no-backup] [--output DIR] [--glob PATTERN] [--json]
```

| Option | Meaning |
|--------|---------|
| `--engine` | **Required** — `builtin`, `llmlingua`, or `caveman`. |
| `--ratio` | Fraction of tokens to remove (default 0.5). |
| `--budget` | Target token count instead of a ratio. |
| `--abbreviate` | Also apply reversible n-gram abbreviation. |
| `--include-frontmatter` | Also compact the YAML frontmatter (off by default). |
| `--apply` | Write changes. Without it, this is a dry run. |
| `--no-backup` | Do not write a `.bak` when applying in place. |
| `--output DIR` | Write results to `DIR` instead of in place. |
| `--glob` | Pattern for directories (default `**/*.md`). |

A dry run prints the savings and a `DRY` / `SKIP` status per file; nothing is
written.

## Python

```python
from compactprompt import review_file, review_directory, compact_file, compact_directory

# Review
report = review_file("SKILL.md")
print(report.tokens, report.est_savings, report.issues)

# Compact (dry run)
result = compact_file("SKILL.md", engine="builtin")
print(result.skipped, result.skip_reason, result.savings)

# Apply (writes + .bak)
result = compact_file("SKILL.md", engine="caveman", apply=True)

# A whole directory
results = compact_directory("./skills", engine="caveman", glob="**/*.md")
```

`engine` is a required keyword argument — calling without it raises `TypeError`.

### Review report

[`review_file`](api-reference.md#compactprompt.review_file) returns a
[`ReviewReport`](api-reference.md#compactprompt.ReviewReport):

| Field | Meaning |
|-------|---------|
| `tokens` | Total tokens in the file. |
| `n_headings` / `n_code_blocks` | Structure counts. |
| `frontmatter_tokens` | Size of the YAML frontmatter. |
| `est_repetition_pct` | Losslessly recoverable repetition. |
| `est_filler_pct` | Function/filler-word density (a verbosity indicator). |
| `est_savings` | A conservative estimate of achievable reduction. |
| `issues` | Flags such as oversized frontmatter or high repetition. |
| `suggestions` | LLM-written notes (only when an `llm` is given). |

### File result

[`compact_file`](api-reference.md#compactprompt.compact_file) returns a
[`FileResult`](api-reference.md#compactprompt.FileResult):

| Field | Meaning |
|-------|---------|
| `compressed` | The compacted text (or the original, if skipped). |
| `tokens_before` / `tokens_after`, `ratio`, `savings` | Size metrics. |
| `applied` | Whether the file was written. |
| `backup_path` | Path to the `.bak`, if one was written. |
| `skipped` / `skip_reason` | Why a file was left untouched. |
| `validation_errors` | Structure problems that caused a revert. |

## Interactive

The Streamlit application has a **Files & Skills** tab that runs the same review
and compaction on pasted markdown, with a download button:

```bash
pip install 'compactprompt[app]'
streamlit run compactprompt_app.py
```
