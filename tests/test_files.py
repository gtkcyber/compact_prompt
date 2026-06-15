"""Tests for the file/skill compaction layer.

These use ``tmp_path`` and the built-in engine (offline). The caveman path is
exercised with an injected fake LLM — no network or model downloads.
"""

import pytest

from compactprompt.files import (
    compact_directory,
    compact_file,
    detect_file_type,
    is_compactable,
    is_sensitive_path,
    review_file,
)
from compactprompt.markdown import extract_urls, validate_structure

SKILL = """\
---
name: example
description: An example skill for testing
---
# Overview

Please could you very kindly go ahead and read the configuration. The configuration
file controls everything. The configuration file is important. The configuration file
must exist. See https://example.com/docs for details. Run `make build` to compile.

```python
x = compute(value)
```
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


# --- classification --------------------------------------------------------
def test_detect_markdown_is_natural_language(tmp_path):
    p = _write(tmp_path, "doc.md", "# Hi\n")
    assert detect_file_type(p) == "natural_language"
    assert is_compactable(p)


def test_detect_code_is_skipped(tmp_path):
    p = _write(tmp_path, "mod.py", "import os\n")
    assert detect_file_type(p) == "code"
    assert not is_compactable(p)


def test_backup_files_not_compactable(tmp_path):
    p = _write(tmp_path, "doc.md.bak", "# Hi\n")
    assert not is_compactable(p)


def test_sensitive_paths_refused(tmp_path):
    assert is_sensitive_path(tmp_path / ".env")
    assert is_sensitive_path(tmp_path / "credentials.md")
    assert is_sensitive_path(tmp_path / "my_secret_notes.md")
    assert not is_sensitive_path(tmp_path / "README.md")


# --- compaction ------------------------------------------------------------
def test_engine_is_required(tmp_path):
    p = _write(tmp_path, "doc.md", SKILL)
    with pytest.raises(TypeError):
        compact_file(p)  # pylint: disable=missing-kwoa  # engine is required


def test_dry_run_writes_nothing(tmp_path):
    p = _write(tmp_path, "SKILL.md", SKILL)
    res = compact_file(p, engine="builtin", ratio=0.4)
    assert not res.applied
    assert p.read_text() == SKILL  # unchanged on disk


def test_frontmatter_preserved(tmp_path):
    p = _write(tmp_path, "SKILL.md", SKILL)
    res = compact_file(p, engine="builtin", ratio=0.4)
    assert res.compressed.startswith("---\nname: example\n")


def test_structure_preserved_or_skipped(tmp_path):
    """Whatever is produced must pass structure validation (else it's skipped)."""
    p = _write(tmp_path, "SKILL.md", SKILL)
    res = compact_file(p, engine="builtin", ratio=0.5)
    if not res.skipped:
        assert not validate_structure(SKILL, res.compressed)
        assert extract_urls(res.compressed) == extract_urls(SKILL)
        assert "x = compute(value)" in res.compressed


def test_apply_writes_and_backs_up(tmp_path):
    p = _write(tmp_path, "SKILL.md", SKILL)
    res = compact_file(p, engine="builtin", ratio=0.4, apply=True)
    if res.skipped:
        pytest.skip("nothing to compact in this environment")
    assert res.applied
    assert res.backup_path.exists()
    assert res.backup_path.read_text() == SKILL  # backup is the original
    assert p.read_text() != SKILL  # file changed


def test_apply_to_output_dir_leaves_original(tmp_path):
    p = _write(tmp_path, "SKILL.md", SKILL)
    out = tmp_path / "out"
    res = compact_file(p, engine="builtin", ratio=0.4, apply=True, output=str(out))
    if res.skipped:
        pytest.skip("nothing to compact")
    assert p.read_text() == SKILL  # original untouched
    assert (out / "SKILL.md").exists()


def test_sensitive_file_skipped(tmp_path):
    p = _write(tmp_path, "credentials.md", SKILL)
    res = compact_file(p, engine="builtin", apply=True)
    assert res.skipped and "sensitive" in res.skip_reason
    assert p.read_text() == SKILL


def test_non_prose_skipped(tmp_path):
    p = _write(tmp_path, "config.json", '{"a": 1}\n')
    res = compact_file(p, engine="builtin")
    assert res.skipped


def test_compact_directory(tmp_path):
    _write(tmp_path, "a.md", SKILL)
    _write(tmp_path, "b.md", SKILL)
    _write(tmp_path, "code.py", "import os\n")
    results = compact_directory(tmp_path, engine="builtin", glob="**/*.md", ratio=0.4)
    assert len(results) == 2  # only the .md files


# --- caveman path (fake LLM, no network) -----------------------------------
def test_caveman_engine_with_fake_llm(tmp_path):
    # The fake LLM returns a terse rewrite that preserves all structure.
    terse = """\
# Overview

Read config first. Controls everything. Must exist.
See https://example.com/docs for details. Run `make build` to compile.

```python
x = compute(value)
```
"""

    def fake_llm(_prompt):
        return terse

    p = _write(tmp_path, "SKILL.md", SKILL)
    res = compact_file(p, engine="caveman", llm=fake_llm, apply=True)
    assert res.applied and not res.skipped
    assert res.tokens_after < res.tokens_before
    assert p.read_text().startswith("---\nname: example\n")  # frontmatter kept
    assert extract_urls(p.read_text()) == extract_urls(SKILL)  # url preserved


def test_caveman_revert_when_structure_breaks(tmp_path):
    def bad_llm(_prompt):
        return "# Overview\n\nlost the url and code"

    p = _write(tmp_path, "SKILL.md", SKILL)
    # max_retries default 2 -> still fails -> file layer skips, original kept.
    res = compact_file(p, engine="caveman", llm=bad_llm, apply=True)
    assert res.skipped
    assert p.read_text() == SKILL


# --- review ----------------------------------------------------------------
def test_review_reports_structure(tmp_path):
    p = _write(tmp_path, "SKILL.md", SKILL)
    r = review_file(p)
    assert r.n_headings == 1
    assert r.n_code_blocks == 1
    assert r.frontmatter_tokens > 0
    assert 0.0 <= r.est_filler_pct <= 1.0
    assert 0.0 <= r.est_savings <= 1.0


def test_review_flags_repetition(tmp_path):
    doc = "# T\n\n" + ("the quarterly revenue report shows growth. " * 6)
    p = _write(tmp_path, "rep.md", doc)
    r = review_file(p)
    assert r.est_repetition_pct > 0.1
    assert any("Repetitive" in i for i in r.issues)
