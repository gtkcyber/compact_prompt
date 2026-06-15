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

## 2. The official MCP Registry — automated

The canonical registry (<https://registry.modelcontextprotocol.io>) is published
to **automatically** by the `publish-mcp-registry` job in
`.github/workflows/publish.yml`, on the same version-tag push that releases to
PyPI. It uses GitHub OIDC (no secret/account setup needed), syncs
[`server.json`](../server.json)'s version to the tag, and runs `mcp-publisher`.

How ownership is proven (already wired up):

- **GitHub OIDC** grants the `io.github.gtkcyber/*` namespace automatically — the
  server name in `server.json` is `io.github.gtkcyber/compactprompt`.
- **PyPI package ownership** is verified by the marker
  `<!-- mcp-name: io.github.gtkcyber/compactprompt -->` in the project README
  (which becomes the PyPI description). The registry job runs *after* the PyPI
  upload so the package exists when it checks.

To release to both PyPI and the registry, just push a tag:

```bash
git tag v0.5.0 && git push origin v0.5.0
```

The registry is in preview, so if the job fails, check the run log; you can also
run `mcp-publisher` locally against `server.json` to debug. Once listed, several
third-party directories crawl the official registry automatically.

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
