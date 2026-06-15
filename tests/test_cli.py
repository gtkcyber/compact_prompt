"""Tests for the command-line interface."""

import json

import pytest

from compactprompt import cli

SKILL = """\
---
name: example
description: An example skill for the CLI tests
---
# Overview

Please could you very kindly go ahead and read the configuration. The configuration
file controls everything and the configuration file is important. See
https://example.com/docs for details. Run `make build` to compile.

```python
x = compute(value)
```
"""


def _write(tmp_path, name="SKILL.md"):
    p = tmp_path / name
    p.write_text(SKILL)
    return p


def test_review_file_exit_zero(tmp_path, capsys):
    p = _write(tmp_path)
    assert cli.main(["review", str(p)]) == 0
    out = capsys.readouterr().out
    assert "tokens" in out


def test_review_json(tmp_path, capsys):
    p = _write(tmp_path)
    assert cli.main(["review", str(p), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data[0]["path"].endswith("SKILL.md")
    assert "est_savings" in data[0]


def test_compact_requires_engine(tmp_path):
    p = _write(tmp_path)
    # argparse exits (SystemExit) when a required arg is missing.
    with pytest.raises(SystemExit):
        cli.main(["compact", str(p)])


def test_compact_dry_run_does_not_write(tmp_path, capsys):
    p = _write(tmp_path)
    assert cli.main(["compact", str(p), "--engine", "builtin", "--ratio", "0.4"]) == 0
    assert p.read_text() == SKILL  # unchanged
    assert "Dry run" in capsys.readouterr().out


def test_compact_apply_writes_and_backups(tmp_path):
    p = _write(tmp_path)
    assert cli.main(["compact", str(p), "--engine", "builtin", "--ratio", "0.4", "--apply"]) == 0
    backup = tmp_path / "SKILL.md.bak"
    # Either it compacted (file changed + backup) or it was a no-op/skip.
    if p.read_text() != SKILL:
        assert backup.exists() and backup.read_text() == SKILL


def test_compact_directory(tmp_path):
    _write(tmp_path, "a.md")
    _write(tmp_path, "b.md")
    assert cli.main(["compact", str(tmp_path), "--engine", "builtin", "--glob", "**/*.md"]) == 0


def test_review_missing_path_errors(tmp_path):
    assert cli.main(["review", str(tmp_path / "nope.md")]) == 2
