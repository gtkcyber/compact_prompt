# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`compactprompt` is a Python library implementing the four prompt/data compression
strategies from *CompactPrompt: A Unified Pipeline for Prompt and Data Compression
in LLM Workflows* (Choi et al., 2025, [arXiv:2510.18043](https://arxiv.org/abs/2510.18043)).
It is an independent, distilled reimplementation — **not** a port of the authors'
reference notebook (which lives one directory up and is a 35-task, config-driven
research harness; this package deliberately collapses that into a one-line API).

## Commands

```bash
# Tests (full suite; optional-dep tests skip when their extras are absent)
pytest
pytest tests/test_ngram.py                  # one file
pytest tests/test_ngram.py::test_round_trip_is_lossless   # one test
pytest -rs                                  # show skip reasons

# Lint (CI gate; must stay at 10.00/10). CI runs over all tracked Python files:
pylint $(git ls-files '*.py')

# Install for development
pip install -e .                            # zero-dep core only
pip install -e '.[all,dev]'                 # everything + pytest, faithful to paper
python -m spacy download en_core_web_sm     # needed for phrase-level pruning

# Runnable tour of every strategy
python examples/quickstart.py

# CLI: review / compact markdown files and Claude Code skills (entry point in cli.py)
compactprompt review ./skills
compactprompt compact FILE.md --engine builtin            # dry run
compactprompt compact FILE.md --engine caveman --apply    # writes + .bak

# Interactive demo (Prompt + Files & Skills tabs)
streamlit run compactprompt_app.py

# Build the docs site (MkDocs); CI/Read the Docs build must stay clean
pip install -e '.[docs]' && mkdocs build --strict
```

CI runs two workflows (`.github/workflows/`): `pylint.yml` and `tests.yml`. The
`tests.yml` **core** job installs the package with no extras across Python
3.9–3.13; the **full** job installs `.[all,dev]` + the spaCy model.

## Releasing

Publishing to PyPI is driven entirely by **version tags** — push a tag and
`publish.yml` builds and uploads via PyPI Trusted Publishing (OIDC, no tokens):

```bash
git tag v0.5.0
git push origin v0.5.0        # triggers publish.yml -> PyPI
```

Key facts:

- The version is derived from the tag by **setuptools-scm** (`v0.5.0` -> `0.5.0`);
  there is nothing to bump in `pyproject.toml`.
- The tag must sit on a commit whose `publish.yml` has the `on: push: tags`
  trigger (i.e. current `main`/`master`), and the build uses the workflow file
  *as it exists at that tagged commit*.
- **Versions must strictly increase** — PyPI permanently rejects re-uploading an
  existing version. Don't reuse or move a tag that already published.
- Don't point two tags at the same commit; setuptools-scm then can't tell which
  version to build.
- `workflow_dispatch` on `publish.yml` publishes to **TestPyPI** instead, for a
  rehearsal. Trusted Publishing must be configured on PyPI for the project (the
  `pypi` / `testpypi` environments and the `publish.yml` publisher).

## Architecture

The central abstraction is `CompactPrompt` (`pipeline.py`), a facade whose
`@classmethod compact(prompt, ...)` is the headline entry point. It composes the
two *text-level* strategies (pruning + abbreviation). The two *data-level*
strategies (quantization, exemplar selection) operate on tables/example-sets, so
they are standalone functions, not part of `compact()`.

The four strategies map to modules:

- **Hard Prompt Compression** (`hard_prompt.py`) — lossy. Scores words via hybrid
  self-information, groups them into grammatical phrases, and prunes the
  lowest-scoring phrases to a token budget.
- **N-gram Abbreviation** (`ngram.py`) — lossless/reversible. Replaces frequent
  multi-word phrases with token-cheap placeholders; `Abbreviation.restore()`
  reverses it exactly.
- **Numerical Quantization** (`quantize.py`) — uniform (paper Eq. 4–5, bounded
  error) and k-means.
- **Representative Example Selection** (`examples.py`) — embed → k-means over
  k∈[5,50] → pick k\* by max silhouette → nearest-centroid point per cluster.

Supporting modules: `scoring.py` (static + dynamic self-information and the
fusion rule), `embedding.py` (shared lazy `all-mpnet-base-v2` loader),
`fidelity.py` (cosine-similarity metric), `tokens.py` (token counting),
`markdown.py` (shared structure helpers — frontmatter split, `validate_structure`,
`prose_segments`; `caveman.py` imports these rather than redefining them).

### File & skill layer

`files.py` compacts markdown files and Claude Code skills (`SKILL.md`) on top of
the string engines, and `cli.py` (entry point `compactprompt`, also `python -m
compactprompt`) wraps it. Two operations:

- **Review** (`review_file` / `review_directory`) — a read-only report (tokens,
  headings, frontmatter size, repetition %, filler %, est. savings, issues).
- **Compact** (`compact_file` / `compact_directory`) — delegates body compaction
  to `CompactPrompt.compact(engine=...)`; `engine` is a **required keyword (no
  default)** by design. Dry run unless `apply=True`.

The Streamlit demo (`compactprompt_app.py`) exposes both the prompt flow and a
**Files & Skills** tab.

### Design invariants — read before changing behavior

These are deliberate decisions; several tests encode them and several are
non-obvious:

1. **Zero-dependency core via lazy imports.** `import compactprompt` must not
   pull in torch/spaCy/sklearn/sentence-transformers/numpy. Every heavy import
   happens *inside* the function/method that needs it and raises an `ImportError`
   naming the extra to install (e.g. `pip install 'compactprompt[dynamic]'`). The
   pyproject `[project.optional-dependencies]` extras (`freq`, `dynamic`,
   `phrases`, `ml`, `embeddings`, `tokenizer`, `all`) are the contract. Never add
   a heavy top-level import.

2. **Pluggable dynamic scorer with offline default.** The dynamic
   self-information scorer is any callable `text -> [(token, start, end, bits)]`.
   `LocalLMScorer` (gpt2 via transformers) is the bundled offline option. Code
   must degrade to static-only scoring when no scorer/spaCy is available rather
   than failing.

   **Swappable pruning engine.** The whole pruner is also replaceable: anything
   exposing `compress(text, ratio=, budget=) -> HardPromptResult` can be passed
   as `CompactPrompt(pruner=...)`. Two backends ship: `llmlingua.py` (Microsoft
   LLMLingua, `engine="llmlingua"`) and `caveman.py` (LLM-based caveman-style
   rewrite, `engine="caveman"`, ported MIT code from JuliusBrussee/caveman with
   a *pluggable* `llm` callable and structure-preservation validation). When
   adding another backend, match that signature and return a `HardPromptResult`
   with token counts from this library's `count_tokens` (not the backend's
   tokenizer) for consistency. Note `caveman` rewrites prose, so it ignores
   `ratio`/`budget` (accepted only for interface compatibility).

3. **`compact()` defaults to prune-only** (`abbreviate=False`). Abbreviated text
   is gibberish to a downstream model without its `dictionary` legend, so
   abbreviation is opt-in and intended for the document-compression case where
   you control both ends.

4. **N-gram token-savings guard** (`require_savings=True`). An abbreviation is
   only applied if its placeholder costs strictly fewer tokens than the phrase,
   so output is never longer than input. **Gotcha:** this depends on the
   tokenizer. `count_tokens` (`tokens.py`) uses `tiktoken` when installed and a
   regex fallback otherwise — under the regex fallback a 2-word phrase and its
   `@N` placeholder both cost 2 tokens, so 2-grams won't be abbreviated. Tests
   that assert shortening must use 3-grams (or `require_savings=False`) to be
   deterministic across environments.

5. **Budget enforcement is two-pass** (`HardPromptCompressor.compress`): a fast
   running-estimate pass, then a correction pass that recomputes the real token
   count (subword merging makes the estimate approximate). Protected units
   (entities/numbers, `score == inf`) are appended last and only pruned as a
   last resort when a tight budget can't otherwise be met.

6. **File compaction is validate-or-revert** (`files.py`). A file is never
   corrupted: YAML frontmatter is preserved verbatim, fenced code blocks are not
   fed to lossy engines, and after compaction `validate_structure` checks that
   headings/code/URLs/inline-code survived — if not, the original is kept and the
   file is reported as skipped. Nothing is written unless `apply=True`, which
   first writes a verified `.bak`. Sensitive/code/config files are skipped up
   front. Because lossy engines (builtin/llmlingua) often trip this on
   inline-structure-heavy markdown, **caveman is the recommended engine for
   human-readable files** — which is why there is no default engine.

### Faithfulness to the paper

Default hyperparameters match the paper and its reference config: fusion
threshold Δ=0.1 (`scoring.combine`), best n-gram config n=2 / top-3, k-means
range [5,50] selected by max silhouette, embedding model `all-mpnet-base-v2`.
Preserve these defaults when editing.
