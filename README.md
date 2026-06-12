# CompactPrompt

[![Tests](https://github.com/gtkcyber/compact_prompt/actions/workflows/tests.yml/badge.svg)](https://github.com/gtkcyber/compact_prompt/actions/workflows/tests.yml)
[![Pylint](https://github.com/gtkcyber/compact_prompt/actions/workflows/pylint.yml/badge.svg)](https://github.com/gtkcyber/compact_prompt/actions/workflows/pylint.yml)

Easy-to-use **prompt & data compression** for LLM workflows — a clean, faithful
implementation of the four strategies from
[*CompactPrompt: A Unified Pipeline for Prompt and Data Compression in LLM
Workflows*](https://arxiv.org/abs/2510.18043) (Choi et al., 2025).

The headline call is one line:

```python
from compactprompt import CompactPrompt

result = CompactPrompt.compact("Please could you very kindly go ahead and provide "
                              "a really concise summary of the quarterly report.")

print(result.text)            # the compressed prompt
print(f"{result.ratio:.2f}x smaller "
      f"({result.tokens_before} -> {result.tokens_after} tokens)")
```

```
a really concise summary of the quarterly report.
1.7x smaller (22 -> 13 tokens)
```

## Why

Long, data-rich prompts are expensive and bump into context limits.
`compactprompt` shrinks them while preserving meaning, using four complementary
techniques from the paper.

## Install

The core (hard-prompt pruning + n-gram abbreviation) has **no required
dependencies** — `CompactPrompt.compact()` works on a clean Python install.
Heavy libraries are imported lazily, only when a strategy needs them. Install
just the extras you want:

```bash
pip install compactprompt                  # core
pip install 'compactprompt[freq]'          # better static scores (wordfreq)
pip install 'compactprompt[dynamic]'       # context-aware scoring (torch + transformers)
pip install 'compactprompt[phrases]'       # grammar-preserving pruning (spaCy)
pip install 'compactprompt[ml]'            # k-means quantization + exemplar selection
pip install 'compactprompt[embeddings]'    # semantic embeddings (all-mpnet-base-v2)
pip install 'compactprompt[llmlingua]'     # LLMLingua as an alternative pruning engine
pip install 'compactprompt[caveman]'       # caveman LLM-based compression engine
pip install 'compactprompt[all]'           # everything, faithful to the paper
```

For phrase-level pruning also download a spaCy model once:

```bash
python -m spacy download en_core_web_sm
```

## Interactive demo

A tiny Streamlit app lets you paste a prompt and watch it compact in real time,
with live token-savings metrics:

```bash
pip install 'compactprompt[app]'
streamlit run streamlit_app.py
```

The sidebar toggles each strategy (pruning aggressiveness, phrase preservation,
context-aware scoring, reversible abbreviation, fidelity measurement); controls
for unavailable optional dependencies are disabled with a hint to install them.

## The four strategies

### 1. Hard Prompt Compression (lossy)

Drops low-information words/phrases, scored with hybrid **static** (corpus
rarity, `-log2 p(t)`) and **dynamic** (context surprisal, `-log2 P_model(t|c)`)
self-information, fused with the paper's Δ=0.1 rule. Phrases are pruned as units
(via spaCy dependency parsing) to keep grammar intact, and named entities /
numbers are protected.

```python
from compactprompt import CompactPrompt

# Target removing ~40% of tokens
r = CompactPrompt.compact(prompt, ratio=0.4)

# Or pin an absolute token budget
r = CompactPrompt.compact(prompt, budget=64)
```

Context-aware scoring is **pluggable**. Use the bundled offline model, or supply
your own scorer (any `text -> [(token, start, end, bits), ...]` callable):

```python
from compactprompt import CompactPrompt, LocalLMScorer

r = CompactPrompt.compact(prompt, scorer=LocalLMScorer("gpt2"))   # offline, no API key
```

#### Swappable pruning engine: built-in or LLMLingua

The pruning engine is pluggable. As well as the built-in self-information
pruner, you can prune with Microsoft's
[LLMLingua](https://github.com/microsoft/LLMLingua) (a mature, perplexity-based
compressor) — it slots in as a drop-in engine:

```python
from compactprompt import CompactPrompt

# Shortcut: use LLMLingua with sensible defaults (LLMLingua-2, CPU)
r = CompactPrompt.compact(prompt, engine="llmlingua")   # needs the [llmlingua] extra

# Or configure it explicitly and reuse the loaded model across prompts
from compactprompt.llmlingua import LLMLinguaCompressor

pruner = LLMLinguaCompressor(device_map="cuda")          # pick model / device
r = CompactPrompt.compact(prompt, pruner=pruner)
```

`ratio` and `budget` map onto LLMLingua's `rate`/`target_token`, and the result
is the same `CompactResult`, so the rest of your code is unchanged. Any object
exposing `compress(text, ratio=, budget=) -> HardPromptResult` works as an
engine — see the [comparison of the engines](#pruning-engines-built-in-llmlingua-caveman)
below.

A third engine, **caveman**, takes a different approach: it asks an LLM to
rewrite prose into terse "caveman speak" while preserving code, URLs, and
headings verbatim (validated, with a fix-retry loop). It's a port of the
[Caveman](https://github.com/JuliusBrussee/caveman) `caveman-compress` skill,
with a pluggable LLM:

```python
from compactprompt import CompactPrompt
from compactprompt.caveman import CavemanCompressor

# Bring any LLM: a callable that takes a prompt string and returns a string.
r = CompactPrompt.compact(prompt, pruner=CavemanCompressor(llm=my_llm))

# Shortcut (default LLM = Anthropic SDK if ANTHROPIC_API_KEY is set, else `claude` CLI):
r = CompactPrompt.compact(prompt, engine="caveman")    # needs the [caveman] extra
```

Because it rewrites rather than drops tokens, `ratio`/`budget` are ignored by
the caveman engine.

### 2. Textual N-gram Abbreviation (lossless / reversible)

Replaces frequent multi-word patterns with short, token-cheap placeholders, and
guarantees an exact round trip. A token-savings guard ensures the output is
never longer than the input.

```python
import compactprompt as cp

doc = "operating cash flow rose. operating cash flow fell. operating cash flow held."
abbr = cp.abbreviate(doc, n=3)
print(abbr.text)          # '@0 rose. @0 fell. @0 held.'
print(abbr.dictionary)    # {'@0': 'operating cash flow'}
assert abbr.restore() == doc
```

Keep `abbr.dictionary` as a legend so the downstream model (or you) can expand it.
Enable it inside the pipeline with `CompactPrompt.compact(text, abbreviate=True)`.

### 3. Numerical Quantization (bounded-error)

Lowers the precision of numeric columns to save tokens, within a guaranteed
error bound.

```python
import compactprompt as cp

q = cp.quantize([1.0, 2.5, 3.3, 4.8, 9.2, 10.0], method="uniform", bits=8)
q.reconstruct()    # approx originals
q.max_error        # epsilon_max bound

q = cp.quantize(values, method="kmeans", k=16)   # needs the `ml` extra

# Or a whole DataFrame:
new_df, results = cp.quantize_dataframe(df, bits=8)
```

### 4. Representative Example Selection (few-shot)

Picks a small, diverse set of exemplars by embedding candidates
(`all-mpnet-base-v2`), running k-means for `k ∈ [5, 50]`, choosing `k*` by
maximum silhouette score, and keeping the point nearest each centroid.

```python
from compactprompt import select_examples   # needs `embeddings` + `ml`

sel = select_examples(candidate_texts, k_range=(5, 50))
few_shot = sel.examples       # the chosen prototypes
sel.k_star                    # selected number of clusters
```

## Pruning engines: built-in, LLMLingua, caveman

All three reduce prompt text, but differently — and any of them slots into
`CompactPrompt.compact(engine=...)` or `pruner=`:

| | Built-in | LLMLingua | caveman |
|---|---|---|---|
| Method | Static+dynamic self-information (Δ=0.1) + phrase pruning | Perplexity / token-classification (LLMLingua-2) | LLM rewrites prose into terse "caveman" style |
| Needs | Zero-dep core; LM scorer optional | Downloads a compressor model | An LLM (pluggable; Anthropic/`claude` by default) |
| Strengths | Grammar-preserving, lightweight, pluggable scorer | Mature, benchmarked, query-aware | Highest prose reduction; preserves code/URLs/headings (validated) |
| Token target | `ratio`/`budget` honored | `ratio`/`budget` honored | rewrites to its own degree (`ratio`/`budget` ignored) |
| Reversible | No (use n-gram abbreviation) | No | No |

They compose: pick whichever engine fits the prose, and still use this
library's reversible n-gram abbreviation, numeric quantization, and few-shot
selection on the data-heavy parts.

## Measuring fidelity

```python
from compactprompt import cosine_fidelity    # needs `embeddings`

f = cosine_fidelity(original_text, result.text)
print(f.mean, f.p5)   # mean and worst-case (5th pct) cosine similarity
```

## The result object

`CompactPrompt.compact(...)` returns a `CompactResult`:

| attribute | meaning |
|---|---|
| `.text` | the compressed prompt (also `str(result)`) |
| `.original` | the input |
| `.tokens_before` / `.tokens_after` | token counts (tiktoken if available) |
| `.ratio` | `tokens_before / tokens_after` (e.g. 2.3x) |
| `.savings` | fraction of tokens removed, `[0, 1]` |
| `.dictionary` | reversible n-gram map (when `abbreviate=True`) |
| `.restore()` | reverse the lossless abbreviation step |
| `.steps` / `.stats` | which strategies ran, with diagnostics |

## All `compact()` options

```python
CompactPrompt.compact(
    prompt,
    ratio=0.5,             # fraction of tokens to remove via pruning (ignored if budget set)
    budget=None,           # absolute target token count
    prune=True,            # hard-prompt pruning (lossy, usable as-is)
    abbreviate=False,      # also apply reversible n-gram abbreviation
    ngram=2,               # n-gram length (paper's best: 2)
    top_k=100,             # number of frequent n-grams to abbreviate
    scorer=None,           # pluggable dynamic self-information scorer
    static=None,           # static self-information scorer (default: best available)
    delta_threshold=0.1,   # static/dynamic fusion threshold (paper default)
    use_phrases=True,      # grammar-preserving phrase pruning (needs spaCy)
    spacy_model="en_core_web_sm",
)
```

Reuse an instance to amortize an expensive scorer/model across many prompts:

```python
cp = CompactPrompt(scorer=LocalLMScorer("gpt2"))
cp.run(prompt_a)
cp.run(prompt_b)
```

## Tests

```bash
pip install pytest
pytest
```

The suite runs on the zero-dependency core; embedding/clustering tests skip
automatically when their optional dependencies are absent.

## Citation

```bibtex
@article{choi2025compactprompt,
  title={CompactPrompt: A Unified Pipeline for Prompt and Data Compression in LLM Workflows},
  author={Choi, Joong Ho and Zhao, Jiayang and Shah, Jeel and Sonawane, Ritvika and
          Singh, Vedant and Appalla, Avani and Flanagan, Will and Condessa, Filipe},
  journal={arXiv preprint arXiv:2510.18043},
  year={2025}
}
```

This is an independent implementation of the methodology; it is not affiliated
with the paper's authors.

## Attribution

The caveman engine (`compactprompt/caveman.py`) is ported from
[Caveman](https://github.com/JuliusBrussee/caveman) by Julius Brussee (MIT).
Full third-party attributions and license notices are in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
