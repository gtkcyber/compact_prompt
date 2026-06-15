# Getting started

This page takes you from installation to your first compaction, and explains
which optional pieces to install for each feature.

## Install

The core library has no required dependencies and works on a clean Python
install:

```bash
pip install compactprompt
```

That is enough for prompt trimming (the built-in engine) and reversible n-gram
abbreviation. Other features depend on extra components, described below.

## Your first compaction

```python
from compactprompt import CompactPrompt

result = CompactPrompt.compact(
    "Please could you very kindly go ahead and provide a really concise "
    "summary of the quarterly report."
)

print(result.text)
print(f"{result.ratio:.1f}x smaller "
      f"({result.tokens_before} -> {result.tokens_after} tokens)")
```

```
a really concise summary of the quarterly report.
1.7x smaller (22 -> 13 tokens)
```

`compact()` returns a [`CompactResult`](api-reference.md#compactprompt.CompactResult);
the most useful fields are `.text` (the shortened prompt), `.ratio`, and
`.savings`. See the [Strategies guide](strategies.md) for the full set of
options.

## Optional features

Each feature installs only what it needs. If you call a feature without its
dependency, CompactPrompt raises an error that names the exact extra to install.

| Install | Adds |
|---------|------|
| `pip install compactprompt` | Core: prompt trimming + reversible abbreviation |
| `pip install 'compactprompt[freq]'` | Better word-rarity scores from a global frequency table (`wordfreq`) |
| `pip install 'compactprompt[dynamic]'` | Context-aware scoring with a local model (`torch`, `transformers`) |
| `pip install 'compactprompt[phrases]'` | Grammar-preserving phrase pruning (`spaCy`) |
| `pip install 'compactprompt[ml]'` | Numeric quantization and example selection (`scikit-learn`) |
| `pip install 'compactprompt[embeddings]'` | Semantic embeddings for selection and fidelity (`sentence-transformers`) |
| `pip install 'compactprompt[tokenizer]'` | Exact token counts matching the OpenAI/Anthropic tokenizers (`tiktoken`) |
| `pip install 'compactprompt[llmlingua]'` | The LLMLingua engine |
| `pip install 'compactprompt[caveman]'` | The Caveman engine (Anthropic SDK) |
| `pip install 'compactprompt[app]'` | The Streamlit demo application |
| `pip install 'compactprompt[all]'` | Everything |

### spaCy model

Phrase-level pruning (used by the built-in engine when available) needs a spaCy
model, downloaded once:

```bash
python -m spacy download en_core_web_sm
```

Without it, the built-in engine still runs — it falls back to scoring individual
words instead of grammatical phrases.

## Token counting

Compaction is measured in tokens. With the `tokenizer` extra (`tiktoken`)
installed, counts match the GPT-4-family tokenizer; otherwise a built-in
word/punctuation counter is used. The two differ slightly in absolute numbers
but track the same compression behavior.

## Where to next

- [Strategies](strategies.md) — the four ways to make a prompt smaller.
- [Engines](engines.md) — choosing how the trimming is performed.
- [Files & skills](files-and-skills.md) — compacting markdown files and
  Claude Code skills from the command line or Python.
- [API reference](api-reference.md) — every public function and class.
