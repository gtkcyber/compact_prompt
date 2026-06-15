# Engines

The wording-trimming step (strategy 1) can be performed by any of three
interchangeable **engines**. They all shorten text; they differ in how they
decide what to remove and in what they require to run. You select one with the
`engine` argument — nothing else in your code changes.

```python
from compactprompt import CompactPrompt

CompactPrompt.compact(prompt)                       # built-in (default)
CompactPrompt.compact(prompt, engine="llmlingua")   # LLMLingua
CompactPrompt.compact(prompt, engine="caveman")     # Caveman
```

## Comparison

| | Built-in | LLMLingua | Caveman |
|---|---|---|---|
| Method | Static + dynamic self-information, phrase-level | Perplexity / token classification (LLMLingua-2) | An LLM rewrites the prose tersely |
| Requirements | None (core) | Downloads a model | Access to an LLM |
| Honors `ratio` / `budget` | Yes | Yes | No (rewrites to its own degree) |
| Preserves code/links/headings | Best-effort, validated by the file layer | Best-effort | Yes, with validation and a fix-retry loop |
| Runs offline | Yes | Yes (after download) | Only with a local LLM |

## Built-in

The default. Scores each word and removes the least useful, preferring to prune
whole grammatical phrases (with spaCy) so the result stays readable. Runs locally
with no extra install. See [Strategies](strategies.md#1-trimming-wording) for its
options (`scorer`, `static`, `delta_threshold`, `use_phrases`).

## LLMLingua

Microsoft's [LLMLingua](https://github.com/microsoft/LLMLingua) — a mature,
benchmarked compressor that uses a small model to decide what to drop.

```bash
pip install 'compactprompt[llmlingua]'
```

```python
from compactprompt import CompactPrompt
from compactprompt.llmlingua import LLMLinguaCompressor

# Shortcut with defaults (a small LLMLingua-2 model on CPU)
CompactPrompt.compact(prompt, engine="llmlingua")

# Or configure the model / device and reuse it across calls
pruner = LLMLinguaCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    device_map="cpu",   # or "cuda"
)
CompactPrompt.compact(prompt, pruner=pruner)
```

Your `ratio` maps to LLMLingua's keep-rate and `budget` to its target token
count. It is query-aware: pass `instruction=` / `question=` (via a configured
`LLMLinguaCompressor`) to keep tokens relevant to a question.

## Caveman

Asks an LLM to rewrite the text in a terse style while keeping code, links, and
headings intact — then validates that nothing structural was lost, repairing it
if needed. Adapted from [Caveman](https://github.com/JuliusBrussee/caveman)
(see [Attribution](index.md#attribution)).

```bash
pip install 'compactprompt[caveman]'
```

### Choosing the LLM

Caveman needs an LLM. You can supply one of:

- **`llm=`** — any callable `str -> str` (works with any provider, fully
  testable). This is the recommended path:

  ```python
  from compactprompt import CompactPrompt
  from compactprompt.caveman import CavemanCompressor

  CompactPrompt.compact(prompt, pruner=CavemanCompressor(llm=my_llm))
  ```

- **The default caller** (used by `engine="caveman"`): it uses the Anthropic SDK
  when `ANTHROPIC_API_KEY` is set, otherwise the `claude` CLI if it is on your
  `PATH`. The model defaults to `claude-sonnet-4-6` and is overridable:

  ```bash
  export ANTHROPIC_API_KEY=sk-...
  export CAVEMAN_MODEL=claude-opus-4-8     # optional
  ```

  ```python
  CompactPrompt.compact(prompt, engine="caveman")
  ```

If no LLM is available, the engine raises a clear `ImportError`.

Because Caveman rewrites rather than drops tokens, `ratio`/`budget` are accepted
for interface compatibility but ignored.

## Writing your own engine

An engine is any object exposing:

```python
def compress(self, text, ratio=0.5, budget=None) -> HardPromptResult: ...
```

Return a [`HardPromptResult`](api-reference.md#compactprompt.HardPromptResult);
the simplest way is `HardPromptResult.from_texts(original, compressed)`, which
counts tokens with this library's counter so ratios stay comparable across
engines. Pass an instance with `CompactPrompt.compact(prompt, pruner=...)`.
