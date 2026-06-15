"""Tests for the MCP server (skipped when the `mcp` extra is absent).

The FastMCP ``@tool`` decorator returns the wrapped function, so the tools can be
called directly and deterministically — no server process or network needed.
"""

import asyncio
import importlib.util
import json

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mcp") is None, reason="needs the 'mcp' extra"
)


def test_tools_registered():
    from compactprompt import mcp_server as m

    tools = asyncio.run(m.mcp.list_tools())
    names = {t.name for t in tools}
    assert {"review", "compact", "compact_prompt", "count_tokens"} <= names


def test_compact_prompt_tool():
    from compactprompt import mcp_server as m

    r = m.compact_prompt("Please could you very kindly summarize the report now.", ratio=0.4)
    assert isinstance(r["text"], str)
    assert r["tokens_after"] <= r["tokens_before"]
    json.dumps(r)  # JSON-serializable


def test_review_and_compact_file_tools(tmp_path):
    from compactprompt import mcp_server as m

    p = tmp_path / "SKILL.md"
    p.write_text("---\nname: x\ndescription: y\n---\n# H\n\nPlease kindly read the docs here.\n")

    review = m.review(str(p))
    json.dumps(review)
    assert review["n_headings"] == 1

    result = m.compact(str(p), engine="builtin", apply=False)
    json.dumps(result)
    assert result["applied"] is False
    assert p.read_text().startswith("---")  # dry run left the file untouched


def test_count_tokens_tool():
    from compactprompt import mcp_server as m

    assert m.count_tokens("one two three") >= 1
