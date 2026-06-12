# CompactPrompt

[![Tests](https://github.com/gtkcyber/compact_prompt/actions/workflows/tests.yml/badge.svg)](https://github.com/gtkcyber/compact_prompt/actions/workflows/tests.yml)
[![Pylint](https://github.com/gtkcyber/compact_prompt/actions/workflows/pylint.yml/badge.svg)](https://github.com/gtkcyber/compact_prompt/actions/workflows/pylint.yml)
[![Documentation Status](https://readthedocs.org/projects/compact-prompt/badge/?version=latest)](https://compact-prompt.readthedocs.io/en/latest/)

CompactPrompt shortens the text you send to an AI model while preserving its
meaning. The result costs less to run, returns faster, and is less likely to
exceed the model's input limit. The common case is a single function call, and
no background in machine learning is required to use it.

## Background

An AI model reads an input — the *prompt* — and returns a response. Providers
charge according to the amount of text processed, measured in *tokens* (each
token is roughly three-quarters of a word), and every model has a maximum input
size. A long prompt that combines instructions, documents, tables, and examples
therefore costs more, responds more slowly, and may not fit at all.

CompactPrompt reduces the size of a prompt while retaining the information that
matters, so you keep the substance and discard the overhead.

## Getting started

Install the library:

```bash
pip install compactprompt
```

Shorten a prompt:

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

Output:

```
a really concise summary of the quarterly report.
1.7x smaller (22 -> 13 tokens)
```

The filler — *"Please could you very kindly go ahead and"* — is removed, and the
meaning is unchanged.

## What it does

CompactPrompt provides several methods for reducing the size of a prompt. They
can be used individually or together. Each is described below in plain terms,
followed by an optional technical note.

### Trimming low-value wording

Removes words that carry little meaning, such as conversational filler, and
keeps the words that do. This is lossy: the removed words are not recoverable,
but the result is ready to use as it is.

```python
from compactprompt import CompactPrompt

# Remove approximately 40% of the tokens
result = CompactPrompt.compact(prompt, ratio=0.4)

# Or target a specific size
result = CompactPrompt.compact(prompt, budget=64)
```

<details>
<summary>Technical detail</summary>

Each word receives an information score that combines how rare it is in general
(static self-information) with how predictable it is in context (dynamic
self-information from a small language model). Low-scoring words are removed.
Whole grammatical phrases are removed together, using spaCy, so the result
remains readable, and names and numbers are protected. This implements the
fusion rule from the *CompactPrompt* paper.
</details>

### Reversible shortening of repeated phrases

When a phrase recurs, it is replaced with a short placeholder, and a key records
what each placeholder stands for. This is lossless: the exact original can be
restored at any time.

```python
import compactprompt as cp

doc = "operating cash flow rose. operating cash flow fell. operating cash flow held."
abbr = cp.abbreviate(doc, n=3)

print(abbr.text)        # '@0 rose. @0 fell. @0 held.'
print(abbr.dictionary)  # {'@0': 'operating cash flow'}
print(abbr.restore())   # the exact original
```

Retain `abbr.dictionary` so the placeholders can be expanded again later.

### Reducing the size of numeric data

Large tables of numbers consume many tokens. This lowers their precision to save
space while guaranteeing that the rounding never exceeds a known bound.

```python
import compactprompt as cp

q = cp.quantize([1.0, 2.5, 3.3, 4.8, 9.2, 10.0], bits=8)
q.reconstruct()   # the rounded values
q.max_error       # the guaranteed maximum error
```

### Selecting representative examples

Models perform better when shown a few examples. If you have many candidate
examples, this selects a small, varied subset that still reflects the full
range, so you send a representative few rather than all of them.

```python
from compactprompt import select_examples

chosen = select_examples(my_examples)
chosen.examples
```

## Choosing how the wording is trimmed

The wording-trimming step can be carried out by any of three interchangeable
engines. All of them shorten text; they differ in how they decide what to remove
and in what they require to run. Select one with the `engine` argument — nothing
else in your code changes.

| Engine | Approach | Requirements |
|--------|----------|--------------|
| **Built-in** (default) | Scores each word and removes the least useful. Runs locally. | None |
| **LLMLingua** | Microsoft's established tool, which uses a small model to decide what to remove. | Downloads a model |
| **Caveman** | Rewrites the text in a concise style, preserving code, links, and headings. | Access to a language model |

```python
CompactPrompt.compact(prompt)                       # built-in, no extra install
CompactPrompt.compact(prompt, engine="llmlingua")   # pip install 'compactprompt[llmlingua]'
CompactPrompt.compact(prompt, engine="caveman")     # pip install 'compactprompt[caveman]'
```

The built-in engine and the other core features implement the
[*CompactPrompt* research paper](https://arxiv.org/abs/2510.18043). LLMLingua and
Caveman are independent open-source tools that this library integrates; see
[Attribution](#attribution).

## Optional features

The basic installation requires no setup. Additional features depend on extra
components, which you install only as needed:

```bash
pip install compactprompt                  # core: trimming and reversible shortening
pip install 'compactprompt[ml]'            # numeric reduction and example selection
pip install 'compactprompt[llmlingua]'     # the LLMLingua engine
pip install 'compactprompt[caveman]'       # the Caveman engine
pip install 'compactprompt[app]'           # the interactive application
pip install 'compactprompt[all]'           # everything
```

When a feature needs a component that is not installed, CompactPrompt reports
exactly what to install.

## Interactive application

A small web application lets you paste a prompt and see it shortened, with the
savings reported as you go:

```bash
pip install 'compactprompt[app]'
streamlit run streamlit_app.py
```

It opens in the browser. Use the sidebar to choose an engine and set how much to
remove.

## Confirming the meaning is preserved

To check that a shortened prompt still means the same thing, you can measure the
similarity between the original and the result, where 1.0 indicates identical
meaning:

```python
from compactprompt import cosine_fidelity   # pip install 'compactprompt[embeddings]'

score = cosine_fidelity(original_text, result.text)
print(score.mean)
```

## Reference

`CompactPrompt.compact(...)` returns a result object with the following fields:

| Field | Meaning |
|-------|---------|
| `.text` | The shortened prompt. |
| `.original` | The input. |
| `.tokens_before` / `.tokens_after` | Size before and after. |
| `.ratio` | How many times smaller (for example, `2.3`). |
| `.savings` | Fraction of tokens saved (for example, `0.4`). |
| `.dictionary` | The key for restoring shortened phrases, when used. |
| `.restore()` | Reverses the reversible shortening step. |

The principal options:

```python
CompactPrompt.compact(
    prompt,
    ratio=0.5,          # how much to remove: 0.5 targets about half the tokens
    budget=None,        # alternatively, a specific target token count
    prune=True,         # trim the wording (default)
    abbreviate=False,   # also shorten repeated phrases (reversible)
    engine="builtin",   # "builtin", "llmlingua", or "caveman"
)
```

The complete reference, including the advanced options, is in the
[documentation](https://compact-prompt.readthedocs.io/).

## Development

Run the tests:

```bash
pip install pytest
pytest
```

The suite runs against the dependency-free core; tests for optional features are
skipped automatically when those components are absent.

Build the documentation locally:

```bash
pip install 'compactprompt[docs]'
mkdocs serve
```

## Citation

This library implements the methodology from:

```bibtex
@article{choi2025compactprompt,
  title={CompactPrompt: A Unified Pipeline for Prompt and Data Compression in LLM Workflows},
  author={Choi, Joong Ho and Zhao, Jiayang and Shah, Jeel and Sonawane, Ritvika and
          Singh, Vedant and Appalla, Avani and Flanagan, Will and Condessa, Filipe},
  journal={arXiv preprint arXiv:2510.18043},
  year={2025}
}
```

It is an independent implementation and is not affiliated with the authors of
the paper.

## Attribution

The Caveman engine (`compactprompt/caveman.py`) is adapted from
[Caveman](https://github.com/JuliusBrussee/caveman) by Julius Brussee (MIT). The
LLMLingua engine uses [LLMLingua](https://github.com/microsoft/LLMLingua) by
Microsoft (MIT). Full third-party attributions and license notices are in
[`THIRD_PARTY_NOTICES.md`](https://github.com/gtkcyber/compact_prompt/blob/main/THIRD_PARTY_NOTICES.md).
