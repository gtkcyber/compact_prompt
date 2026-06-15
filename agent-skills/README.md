# compactprompt for AI agents

Make [`compactprompt`](https://pypi.org/project/compactprompt/) usable from AI
coding tools — Claude Code, Codex, Cursor, Gemini CLI, and any other MCP-capable
agent. There are two layers, and you can use either or both:

1. **MCP server** (recommended) — structured `review` / `compact` tools that any
   MCP client can call. One integration for every tool.
2. **Skill / rules files** — lightweight instructions that tell an agent *when*
   to reach for the tool. Generated from a single source so there's nothing to
   keep in sync.

> Everything reuses the same code: the MCP server wraps the library API
> directly, and every per-tool instruction file is generated from
> [`INSTRUCTIONS.md`](INSTRUCTIONS.md).

## 1. The MCP server

```bash
pip install 'compactprompt[mcp]'      # provides the `compactprompt-mcp` command
```

It exposes four tools: `review`, `compact`, `compact_prompt`, and `count_tokens`.
Register the `compactprompt-mcp` command in your tool:

**Claude Code**

```bash
claude mcp add compactprompt -- compactprompt-mcp
```

**Cursor** — `.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global):

```json
{ "mcpServers": { "compactprompt": { "command": "compactprompt-mcp" } } }
```

**Gemini CLI** — `~/.gemini/settings.json`:

```json
{ "mcpServers": { "compactprompt": { "command": "compactprompt-mcp" } } }
```

**Codex** — `~/.codex/config.toml`:

```toml
[mcp_servers.compactprompt]
command = "compactprompt-mcp"
```

Any other MCP client: run `compactprompt-mcp` as a stdio server.

## 2. The skill / rules files

These tell the agent it has compactprompt available and how to use it (via the
MCP tools or the CLI). Install them with:

```bash
./install.sh                 # current project, all tools
./install.sh --user          # your home config instead
./install.sh --claude        # just one tool (any of --claude/--codex/--cursor/--gemini)
```

What it generates from [`INSTRUCTIONS.md`](INSTRUCTIONS.md):

| Tool | File |
|------|------|
| Claude Code | `.claude/skills/compactprompt/SKILL.md` |
| Codex | `AGENTS.md` (a marked block, refreshed in place) |
| Cursor | `.cursor/rules/compactprompt.mdc` |
| Gemini CLI | `GEMINI.md` (a marked block, refreshed in place) |

`--user` writes the Claude / Codex / Gemini files into `~/.claude`, `~/.codex`,
and `~/.gemini`; Cursor rules are always project-scoped.

To change the guidance, edit `INSTRUCTIONS.md` and re-run `install.sh` — every
tool's file is regenerated from it.

## What the agent can do once set up

- **Review** a file or folder for compaction opportunities (read-only).
- **Compact** a prompt, markdown file, or skill — dry-run by default, writing a
  `.bak` backup only when applied, and never breaking frontmatter, code, or links.

See the top-level [project docs](https://compact-prompt.readthedocs.io/) for the
engines (`builtin` / `llmlingua` / `caveman`) and the full safety model.
