# Strategies

CompactPrompt offers four complementary ways to reduce the size of a prompt,
from the [*CompactPrompt* paper](https://arxiv.org/abs/2510.18043). They can be
used on their own or together. This page covers each in depth.

| Strategy | Lossy? | Best for |
|----------|--------|----------|
| [Trimming wording](#1-trimming-wording) | Lossy | Verbose instructions and prose |
| [N-gram abbreviation](#2-reversible-n-gram-abbreviation) | Lossless | Repetitive documents you control both ends of |
| [Numeric quantization](#3-numeric-quantization) | Bounded error | Tables of floating-point numbers |
| [Example selection](#4-representative-example-selection) | Selection | Large pools of few-shot examples |

---

## 1. Trimming wording

Removes words that carry little meaning while keeping the words that do. This is
the headline `compact()` call.

```python
from compactprompt import CompactPrompt

# Remove roughly 40% of the tokens
result = CompactPrompt.compact(prompt, ratio=0.4)

# Or target an absolute size
result = CompactPrompt.compact(prompt, budget=64)
```

### Options

```python
CompactPrompt.compact(
    prompt,
    ratio=0.5,            # fraction of tokens to remove (ignored if budget set)
    budget=None,          # target token count
    prune=True,           # do the wording trim
    abbreviate=False,     # also run n-gram abbreviation (see below)
    engine="builtin",     # which engine performs the trim (see Engines guide)
    scorer=None,          # pluggable context-aware scorer
    static=None,          # static word-rarity scorer
    delta_threshold=0.1,  # static/dynamic fusion threshold (paper default)
    use_phrases=True,     # prune whole grammatical phrases (needs spaCy)
    spacy_model="en_core_web_sm",
)
```

### How it works

Each word receives an information score combining how rare it is in general
(*static self-information*) with how predictable it is in context (*dynamic
self-information*). The two are fused with the paper's rule: when they disagree
by more than `delta_threshold`, the context-aware score wins. The
lowest-scoring words are removed. With spaCy installed, whole phrases are pruned
together so the result stays readable, and named entities and numbers are
protected from removal.

Context-aware scoring is optional and pluggable. The bundled offline option uses
a small local model:

```python
from compactprompt import CompactPrompt, LocalLMScorer

result = CompactPrompt.compact(prompt, scorer=LocalLMScorer("gpt2"))
```

You can also pass any callable `text -> [(token, start, end, bits), ...]`.

### The result

```python
result.text            # the shortened prompt (also str(result))
result.tokens_before   # original token count
result.tokens_after    # compressed token count
result.ratio           # tokens_before / tokens_after, e.g. 2.3
result.savings         # fraction removed, 0-1
result.steps           # which strategies ran
```

---

## 2. Reversible n-gram abbreviation

Replaces frequently repeated phrases with short placeholders and remembers what
each one means. This is **lossless** — the exact original can always be restored.

```python
import compactprompt as cp

doc = "operating cash flow rose. operating cash flow fell. operating cash flow held."
abbr = cp.abbreviate(doc, n=3)

print(abbr.text)        # '@0 rose. @0 fell. @0 held.'
print(abbr.dictionary)  # {'@0': 'operating cash flow'}
print(abbr.restore())   # the exact original
```

### Options

```python
from compactprompt import NgramAbbreviator

NgramAbbreviator(
    n=2,                  # phrase length in words (paper's best: 2)
    top_k=100,            # how many frequent phrases to abbreviate
    min_count=2,          # only abbreviate phrases seen at least this often
    marker="@",           # placeholder prefix (@0, @1, ...)
    require_savings=True, # only abbreviate when it actually reduces tokens
)
```

Keep `abbr.dictionary` as a legend so a downstream model — or you — can expand
the placeholders again.

!!! note "The token-savings guard"
    With `require_savings=True` (the default), a phrase is only abbreviated if
    its placeholder costs strictly fewer tokens than the phrase. Whether a given
    phrase qualifies depends on the tokenizer: under the dependency-free counter,
    a two-word phrase and its `@0` placeholder both cost two tokens, so 2-grams
    are skipped. Use 3-grams (or `require_savings=False`) when you need
    abbreviation to fire regardless.

Inside the pipeline, enable it with `CompactPrompt.compact(text, abbreviate=True)`.
Because abbreviated text is not human-readable without its legend, abbreviation
is off by default and intended for cases where you control both ends.

---

## 3. Numeric quantization

Reduces the precision of numeric data to save tokens, within a guaranteed error
bound.

```python
import compactprompt as cp

q = cp.quantize([1.0, 2.5, 3.3, 4.8, 9.2, 10.0], method="uniform", bits=8)
q.reconstruct()   # the rounded-back values
q.max_error       # the guaranteed maximum absolute error
```

Two methods:

- **Uniform** (`method="uniform"`, `bits=8`) — maps values onto `2**bits` evenly
  spaced levels. Pure Python; the maximum error is `(max - min) / (2**bits - 1)`.
- **K-means** (`method="kmeans"`, `k=16`) — maps values onto `k` learned
  centroids, minimizing average error. Needs the `ml` extra.

For tabular data, quantize whole columns:

```python
new_df, results = cp.quantize_dataframe(df, bits=8)
results["price"].max_error
```

`quantize_dataframe` returns a copy of the frame with quantized numeric columns
plus a per-column [`QuantizedColumn`](api-reference.md#compactprompt.QuantizedColumn).

---

## 4. Representative example selection

Chooses a small, varied subset of few-shot examples that still covers the range,
instead of sending all of them. Needs the `embeddings` and `ml` extras.

```python
from compactprompt import select_examples

result = select_examples(candidate_texts, k_range=(5, 50))
result.examples   # the chosen prototypes
result.k_star     # how many clusters were selected
result.indices    # their positions in the original list
```

### How it works

Candidates are embedded with `all-mpnet-base-v2`, clustered with k-means for each
`k` in the range, and the `k` with the best silhouette score is chosen. From each
cluster, the example nearest the center is kept. Optional `numeric_features` are
standardized and combined with the text embeddings.

This reduces the *number* of examples; to also shorten each one, run
`CompactPrompt.compact` on the selected texts.

---

## Confirming meaning is preserved

After compacting, measure how close the result is to the original (1.0 = identical
meaning). Needs the `embeddings` extra.

```python
from compactprompt import cosine_fidelity

score = cosine_fidelity(original_text, result.text)
print(score.mean, score.p5)   # mean and worst-case (5th percentile)
```
