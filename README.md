# CompactPrompt

[![Tests](https://github.com/gtkcyber/compact_prompt/actions/workflows/tests.yml/badge.svg)](https://github.com/gtkcyber/compact_prompt/actions/workflows/tests.yml)
[![Pylint](https://github.com/gtkcyber/compact_prompt/actions/workflows/pylint.yml/badge.svg)](https://github.com/gtkcyber/compact_prompt/actions/workflows/pylint.yml)
[![Documentation Status](https://readthedocs.org/projects/compactprompt/badge/?version=latest)](https://compactprompt.readthedocs.io/en/latest/)

**CompactPrompt makes the text you send to an AI shorter — without losing the
meaning — so it costs less, runs faster, and fits inside the AI's size limit.**

You don't need to be an AI expert to use it. The simplest version is one line of
Python.

---

## What problem does this solve?

When you use an AI model (like ChatGPT or Claude), you send it text — a
**prompt** — and it sends text back. Two things matter:

- **AI models charge by the amount of text.** The unit they count is called a
  **token** (roughly ¾ of a word). More text = more money and slower replies.
- **Every model has a size limit** (a "context window"). Very long prompts
  simply don't fit.

So if your prompt is long — full of instructions, documents, tables, and
examples — you pay more, wait longer, and can hit the limit.

**CompactPrompt shrinks the prompt while keeping what matters.** Think of it like
packing a suitcase more efficiently: the same clothes, less space.

### A few words you'll see

| Word | What it means |
|------|---------------|
| **Prompt** | The text you send to an AI model. |
| **Token** | The chunk of text AI models count and charge for (~¾ of a word). |
| **LLM** | "Large Language Model" — the AI that reads your prompt (e.g. Claude, GPT). |
| **Lossless** | Can be reversed perfectly back to the original. |
| **Lossy** | Saves more space, but some wording is gone for good. |

---

## Try it in 30 seconds

Install it:

```bash
pip install compactprompt
```

Shrink a prompt:

```python
from compactprompt import CompactPrompt

result = CompactPrompt.compact(
    "Please could you very kindly go ahead and provide a really concise "
    "summary of the quarterly report."
)

print(result.text)     # the shorter version
print(f"{result.ratio:.1f}x smaller "
      f"({result.tokens_before} → {result.tokens_after} tokens)")
```

Output:

```
a really concise summary of the quarterly report.
1.7x smaller (22 → 13 tokens)
```

That's it. The wording you didn't need ("Please could you very kindly go ahead
and") is gone; the meaning stays.

> **Want to see it without writing code?** There's a [point-and-click demo
> app](#point-and-click-demo) below.

---

## What it can do

CompactPrompt offers several ways to make a prompt smaller. You can use them on
their own or together. Each one is explained in plain language first, with a
short "how it works" note for the curious.

### 1. Trim the wording (hard prompt pruning)

Removes low-value words — filler like *"please could you kindly go ahead and"* —
and keeps the words that carry the meaning. This is **lossy** (the removed words
are gone), but the result is ready to use as-is.

```python
from compactprompt import CompactPrompt

# Aim to cut about 40% of the tokens
result = CompactPrompt.compact(prompt, ratio=0.4)

# ...or aim for a specific size
result = CompactPrompt.compact(prompt, budget=64)   # ~64 tokens
```

<details>
<summary>How it works (technical)</summary>

Each word gets an "information score" combining how rare it is in general
(static self-information) and how surprising it is in context (dynamic
self-information from a small language model). Low-scoring words are dropped.
Whole grammatical phrases are removed together (using spaCy) so the result stays
readable, and names/numbers are protected. This follows the *CompactPrompt*
paper's Δ=0.1 fusion rule.
</details>

### 2. Shorten repeated phrases — reversibly (n-gram abbreviation)

If a phrase repeats a lot, it's replaced with a short placeholder, and a
"legend" remembers what each placeholder means. This is **lossless** — you can
restore the exact original at any time.

```python
import compactprompt as cp

doc = "operating cash flow rose. operating cash flow fell. operating cash flow held."
abbr = cp.abbreviate(doc, n=3)

print(abbr.text)        # '@0 rose. @0 fell. @0 held.'
print(abbr.dictionary)  # {'@0': 'operating cash flow'}
print(abbr.restore())   # back to the exact original
```

Keep `abbr.dictionary` so you (or the AI) can expand the placeholders later.

### 3. Round numbers safely (numeric quantization)

Big tables of numbers take up a lot of tokens. This rounds them to save space,
while guaranteeing the rounding never exceeds a known limit.

```python
import compactprompt as cp

q = cp.quantize([1.0, 2.5, 3.3, 4.8, 9.2, 10.0], bits=8)
q.reconstruct()   # the rounded-back numbers
q.max_error       # the guaranteed maximum rounding error
```

### 4. Pick the best examples (few-shot selection)

AI works better when you show it a few examples. If you have *hundreds* of
candidate examples, this chooses a small, varied handful that still covers the
range — so you send a few instead of all of them.

```python
from compactprompt import select_examples

chosen = select_examples(my_examples)
chosen.examples   # the handful to actually send
```

---

## Choosing how the trimming is done (engines)

Step 1 (trimming the wording) can be done three different ways. They all make
text shorter; they differ in *how* they decide what to cut and what you need
installed. You pick one with `engine="..."` — the rest of your code is identical.

| Engine | In plain terms | Needs |
|--------|----------------|-------|
| **Built-in** (default) | Scores each word and drops the least useful. Fast, runs on your machine. | Nothing extra |
| **LLMLingua** | Microsoft's well-tested tool that uses a small AI to decide what to cut. | Downloads a model |
| **Caveman** | Asks an AI to *rewrite* your text in a terse style, keeping code, links, and headings intact. | Access to an AI (see below) |

```python
# Use the default — nothing to install:
CompactPrompt.compact(prompt)

# Use LLMLingua instead:
CompactPrompt.compact(prompt, engine="llmlingua")   # pip install 'compactprompt[llmlingua]'

# Use Caveman:
CompactPrompt.compact(prompt, engine="caveman")     # pip install 'compactprompt[caveman]'
```

The built-in engine and CompactPrompt's other features come from the
[*CompactPrompt* research paper](https://arxiv.org/abs/2510.18043). LLMLingua and
Caveman are excellent open-source tools that this library plugs in for you (see
[Attribution](#attribution)).

---

## Installing the extra features

The basic install works with no setup. Some features need extra pieces — install
only the ones you want:

```bash
pip install compactprompt                  # core (trimming + reversible abbreviation)
pip install 'compactprompt[ml]'            # number rounding + example selection
pip install 'compactprompt[llmlingua]'     # the LLMLingua engine
pip install 'compactprompt[caveman]'       # the Caveman engine
pip install 'compactprompt[app]'           # the point-and-click demo app
pip install 'compactprompt[all]'           # everything
```

If a feature needs something you haven't installed, CompactPrompt tells you
exactly what to run.

---

## Point-and-click demo

Prefer not to write code? A small web app lets you paste a prompt and watch it
shrink, with the savings shown live:

```bash
pip install 'compactprompt[app]'
streamlit run streamlit_app.py
```

It opens in your browser. Use the sidebar to choose an engine and how much to
trim.

---

## Checking the meaning was kept

Worried a shorter prompt changed the meaning? You can measure how similar the
"before" and "after" are (1.0 = identical meaning):

```python
from compactprompt import cosine_fidelity   # needs: pip install 'compactprompt[embeddings]'

score = cosine_fidelity(original_text, result.text)
print(score.mean)   # e.g. 0.85
```

---

## Reference

### What `compact()` gives you back

`CompactPrompt.compact(...)` returns a result object:

| Attribute | Meaning |
|-----------|---------|
| `.text` | The shortened prompt. |
| `.original` | What you put in. |
| `.tokens_before` / `.tokens_after` | Size before and after. |
| `.ratio` | How many times smaller (e.g. `2.3` = 2.3× smaller). |
| `.savings` | Fraction of tokens saved (e.g. `0.4` = 40%). |
| `.dictionary` | The legend for restoring abbreviations (if used). |
| `.restore()` | Undo the reversible abbreviation step. |

### Main options

```python
CompactPrompt.compact(
    prompt,
    ratio=0.5,          # how much to cut: 0.5 = aim to remove ~half the tokens
    budget=None,        # OR aim for a specific token count
    prune=True,         # trim the wording (on by default)
    abbreviate=False,   # also shorten repeated phrases (reversible)
    engine="builtin",   # "builtin", "llmlingua", or "caveman"
)
```

Full details — including the advanced options — are in the
[documentation](https://compactprompt.readthedocs.io/).

---

## For developers

### Run the tests

```bash
pip install pytest
pytest
```

The test suite runs on the no-dependency core; tests for optional features skip
themselves automatically when those pieces aren't installed.

### Build the docs locally

```bash
pip install 'compactprompt[docs]'
mkdocs serve     # then open http://127.0.0.1:8000
```

---

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

It is an independent implementation and is not affiliated with the paper's
authors.

## Attribution

The Caveman engine (`compactprompt/caveman.py`) is adapted from
[Caveman](https://github.com/JuliusBrussee/caveman) by Julius Brussee (MIT). The
LLMLingua engine uses [LLMLingua](https://github.com/microsoft/LLMLingua) by
Microsoft (MIT). Full third-party attributions and license notices are in
[`THIRD_PARTY_NOTICES.md`](https://github.com/gtkcyber/compact_prompt/blob/main/THIRD_PARTY_NOTICES.md).
