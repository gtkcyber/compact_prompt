"""Tests for the shared markdown utilities."""

from compactprompt.markdown import (
    extract_code_blocks,
    extract_urls,
    prose_segments,
    split_frontmatter,
    validate_structure,
)

DOC = """\
---
name: x
description: y
---
# Title

Some prose here with a link https://example.com and `inline` code.

```python
a = 1
```

More prose.
"""


def test_split_frontmatter_roundtrip():
    fm, body = split_frontmatter(DOC)
    assert fm.startswith("---") and "name: x" in fm
    assert fm + body == DOC
    assert body.startswith("# Title")


def test_split_frontmatter_absent():
    fm, body = split_frontmatter("# No frontmatter\n")
    assert fm == ""
    assert body == "# No frontmatter\n"


def test_prose_segments_reassemble_losslessly():
    _, body = split_frontmatter(DOC)
    segs = prose_segments(body)
    assert "\n".join(s for _, s in segs) == body
    kinds = {k for k, _ in segs}
    assert "code" in kinds and "prose" in kinds


def test_prose_segments_isolates_code():
    _, body = split_frontmatter(DOC)
    code = [s for k, s in prose_segments(body) if k == "code"]
    assert len(code) == 1
    assert "a = 1" in code[0]


def test_validate_structure_passes_when_preserved():
    assert not validate_structure(DOC, DOC)


def test_validate_structure_flags_lost_url():
    broken = DOC.replace("https://example.com", "")
    assert any("URL" in e for e in validate_structure(DOC, broken))


def test_validate_structure_flags_changed_code():
    broken = DOC.replace("a = 1", "a = 2")
    assert any("Code blocks" in e for e in validate_structure(DOC, broken))


def test_extract_helpers():
    assert extract_urls(DOC) == {"https://example.com"}
    assert len(extract_code_blocks(DOC)) == 1
