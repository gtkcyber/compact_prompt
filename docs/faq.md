# FAQ & troubleshooting

## "It needs the X package / extra"

Features depend on optional components. When one is missing, CompactPrompt raises
an error naming the exact extra to install — for example
`pip install 'compactprompt[ml]'` for quantization and example selection. See the
[install matrix](getting-started.md#optional-features).

## Can I use it without any LLM or API key?

Yes. The **built-in engine**, **n-gram abbreviation**, and **numeric
quantization** run entirely offline with no API key. Only two things involve a
model:

- The **Caveman engine** needs an LLM (an Anthropic key, the `claude` CLI, or a
  callable you provide).
- The **LLMLingua engine** downloads a local model but needs no API key.

Context-aware scoring for the built-in engine is optional; without it, the engine
uses offline static scoring.

## Why was my file skipped?

`compact_file` / `compactprompt compact` leave a file untouched and report a
reason when:

- it is **not a natural-language file** (code or config — only markdown/text is
  processed);
- it **looks sensitive** (credentials, keys, `.env`, `.ssh/.aws/...`);
- compaction **would break structure** (a heading, code block, URL, or
  inline-code span would be lost) — the validate-or-revert guarantee;
- the result was **identical** to the input (nothing to do); or
- the file is **larger than 1 MB**.

The reason is in `result.skip_reason` (or the `SKIP` line in the CLI).

## The built-in engine mangled my markdown

The built-in and LLMLingua engines drop tokens, which can read awkwardly in prose
written for humans and frequently trips the structure check on markdown with
inline links and code (so the file is safely skipped rather than corrupted). For
human-readable files, use the **Caveman engine**, which rewrites prose while
preserving structure:

```bash
compactprompt compact FILE.md --engine caveman --apply
```

## Token counts look different from what my model reports

With the `tokenizer` extra (`tiktoken`) installed, counts match the GPT-4-family
tokenizer. Without it, a built-in word/punctuation counter is used; absolute
numbers differ slightly but the compression behavior is the same. Install
`pip install 'compactprompt[tokenizer]'` for exact counts.

## How do I get the original text back after abbreviation?

N-gram abbreviation is lossless. Keep the dictionary and call `restore()`:

```python
abbr = cp.abbreviate(doc, n=3)
original = abbr.restore()                       # from the Abbreviation object
# or, later, from the stored mapping:
original = cp.restore(abbr.text, abbr.dictionary)
```

Wording-trimming (the built-in/LLMLingua/Caveman engines) is **lossy** and cannot
be reversed.

## Does it handle few-shot examples?

Yes, but as a separate strategy. `select_examples` reduces the *number* of
examples by picking a representative subset; it does not shorten each example.
To also shorten them, run `CompactPrompt.compact` on the selected texts. See
[Strategies](strategies.md#4-representative-example-selection).

## Which model does the Caveman engine use?

By default `claude-sonnet-4-6`, overridable with the `CAVEMAN_MODEL` environment
variable or `CavemanCompressor(model="...")`. If you pass your own `llm=`
callable, the module does not choose a model at all. See the
[Engines guide](engines.md#caveman).

## How is the package version determined?

From git tags, via `setuptools-scm`. A tagged commit such as `v0.1.0` produces
version `0.1.0`; between tags you get a development version like
`0.1.1.dev3+g<hash>`.

## How do I verify a shortened prompt still means the same thing?

Use `cosine_fidelity(original, compressed)` (needs the `embeddings` extra), which
reports a similarity from 0 to 1. See
[Strategies](strategies.md#confirming-meaning-is-preserved).
