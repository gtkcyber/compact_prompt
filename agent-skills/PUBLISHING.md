# Publishing & discovery

How to get the compactprompt MCP server and agent skills in front of users.

## 1. PyPI (the package + MCP server)

The MCP server ships inside the `compactprompt` package (the `[mcp]` extra), so
publishing a release makes it available everywhere:

```bash
git tag v0.5.0 && git push origin v0.5.0    # builds and uploads to PyPI
```

After that, anyone can `pip install 'compactprompt[mcp]'` to get the
`compactprompt-mcp` command. Keep the `version` fields in `server.json` (below)
in sync with the released version.

## 2. The official MCP Registry

The canonical place for MCP servers is the official registry
(<https://registry.modelcontextprotocol.io>). The repo root ships a
[`server.json`](../server.json) manifest. Publish it with the official tool:

```bash
# install the publisher CLI (see the registry docs for the latest method)
mcp-publisher login github          # authenticates the io.github.gtkcyber/* namespace
mcp-publisher publish               # validates and publishes ./server.json
```

Notes before submitting:

- **Validate first.** The registry schema evolves — confirm the `$schema` URL in
  `server.json` is current, and let `mcp-publisher` validate it; it will flag any
  missing/renamed fields.
- The manifest declares the PyPI package and a `stdio` transport. Clients run the
  `compactprompt-mcp` console script (installed with the `[mcp]` extra); confirm
  your client invokes that command. The per-tool config snippets are in
  [`README.md`](README.md).

Once listed, several third-party directories crawl the official registry
automatically.

## 3. Community directories

Listing on a couple of these drives most discovery (most accept a PR or crawl
GitHub/the official registry):

- `punkpeye/awesome-mcp-servers` (GitHub) — submit a PR (entry below).
- Smithery — <https://smithery.ai>
- PulseMCP — <https://www.pulsemcp.com>
- mcp.so, Glama — index servers automatically.

### awesome-mcp-servers entry

Add under a relevant category (e.g. *Text Processing* / *Developer Tools*).
Legend: 🐍 Python, 🏠 runs locally.

```markdown
- [gtkcyber/compact_prompt](https://github.com/gtkcyber/compact_prompt) 🐍 🏠 - Review and compact prompts, markdown docs, and Claude Code skills to save tokens, preserving code, links, and structure.
```

## 4. Per-tool ecosystems (the skill/rules files)

- **Cursor** → <https://cursor.directory> (community rules and MCP listings).
- **Claude Code** → community lists (e.g. `awesome-claude-code`), or package the
  skill as a Claude Code plugin in a marketplace for one-command install.
- **Codex / Gemini** → no central registry; the `install.sh` here is the
  distribution path. Point users at [`README.md`](README.md).

## 5. The repository

Add GitHub **topics** so crawlers and search surface it:
`mcp`, `mcp-server`, `llm`, `prompt-engineering`, `prompt-compression`, `claude`,
`tokens`.
