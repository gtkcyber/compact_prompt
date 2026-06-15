# Compacting prompts, docs, and skills with compactprompt

Use **compactprompt** to reduce token usage by shrinking prompts, markdown
documentation, and AI-agent skills while preserving their meaning and structure.

## When to use

- The user wants to **compact / shrink / trim / reduce** the size or token count
  of: a prompt, a markdown file (`README.md`, `CLAUDE.md`, docs, notes), or an
  agent skill (`SKILL.md` / `AGENTS.md`).
- The user wants to **review** a file or folder for compaction opportunities.

## How to call it

**Preferred — the MCP tools** (when the `compactprompt` MCP server is configured):

- `review(path)` — read-only report of tokens, repetition, and estimated savings.
- `compact(path, engine, apply=false)` — compact a file/skill; previews unless `apply` is true.
- `compact_prompt(prompt, ratio, engine)` — shorten a prompt string.
- `count_tokens(text)` — count tokens.

**Or the CLI** (the package ships a `compactprompt` command):

```bash
pip install compactprompt                 # once; add [caveman] or [llmlingua] for those engines
compactprompt review PATH                 # read-only report
compactprompt compact PATH --engine ENGINE          # dry run (writes nothing)
compactprompt compact PATH --engine ENGINE --apply  # writes; makes a .bak backup first
```

For a one-off prompt string in Python:

```python
from compactprompt import CompactPrompt
print(CompactPrompt.compact(prompt, ratio=0.4).text)
```

## Engines (required — choose one per task)

- `builtin` — offline, no setup; good for plain prose.
- `llmlingua` — perplexity-based; needs the `[llmlingua]` extra.
- `caveman` — an LLM rewrites the prose tersely; **best for human-readable docs
  and skills**; needs an LLM (an `ANTHROPIC_API_KEY` or the `claude` CLI).

## Safety and behavior

- Compaction is **dry-run by default** — it writes nothing until `apply` /
  `--apply`, which makes a `.bak` backup first.
- YAML frontmatter, fenced code blocks, and links are always preserved. If
  compaction would change a heading, code block, or URL, the file is left
  unchanged and reported as skipped.
- Code and config files are skipped automatically — never compact them.
- Always **review or preview first**, show the user the savings, and only apply
  once they confirm. Prefer the `caveman` engine for prose written for humans.
