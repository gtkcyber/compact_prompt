#!/usr/bin/env bash
#
# Install the compactprompt agent skill / rules for AI coding tools.
#
# Every tool's file is GENERATED from the single source INSTRUCTIONS.md, so the
# guidance lives in exactly one place. (The MCP server is separate — see the
# README; it reuses the library directly.)
#
# Usage:
#   ./install.sh [--project | --user] [--claude] [--codex] [--cursor] [--gemini]
#
#   --project   install into the current directory (default)
#   --user      install into your home config (~/.claude, ~/.codex, ~/.gemini)
#   --claude/--codex/--cursor/--gemini   install only the named tool(s)
#   (with no tool flags, all four are installed)
#
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
BODY="$SRC/INSTRUCTIONS.md"

SCOPE="project"
want_claude=0; want_codex=0; want_cursor=0; want_gemini=0; explicit=0

SKILL_DESC="Reduce token usage by compacting prompts, markdown docs, and agent skills with compactprompt. Use when the user wants to shrink, compact, trim, or reduce the token count of a prompt, a markdown file (README, CLAUDE.md, docs), or a SKILL.md, or to review files for compaction opportunities."
CURSOR_DESC="Use compactprompt to shrink prompts, docs, and skills to save tokens."

usage() { sed -n '3,17p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --project) SCOPE="project" ;;
    --user)    SCOPE="user" ;;
    --claude)  want_claude=1; explicit=1 ;;
    --codex)   want_codex=1;  explicit=1 ;;
    --cursor)  want_cursor=1; explicit=1 ;;
    --gemini)  want_gemini=1; explicit=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

if [ "$explicit" -eq 0 ]; then
  want_claude=1; want_codex=1; want_cursor=1; want_gemini=1
fi

echo "Installing compactprompt agent files (scope: $SCOPE)"

write_skill() {            # Claude Code: a SKILL.md inside a skill directory
  if [ "$SCOPE" = "user" ]; then dir="$HOME/.claude/skills/compactprompt"
  else dir="$PWD/.claude/skills/compactprompt"; fi
  mkdir -p "$dir"
  { printf -- '---\nname: compactprompt\ndescription: %s\n---\n\n' "$SKILL_DESC"
    cat "$BODY"; } > "$dir/SKILL.md"
  echo "  Claude Code skill  -> $dir/SKILL.md"
}

write_cursor() {          # Cursor: a project rule (Cursor has no user-level rules file)
  dir="$PWD/.cursor/rules"
  mkdir -p "$dir"
  { printf -- '---\ndescription: %s\nalwaysApply: false\n---\n\n' "$CURSOR_DESC"
    cat "$BODY"; } > "$dir/compactprompt.mdc"
  echo "  Cursor rule        -> $dir/compactprompt.mdc"
}

upsert_block() {          # add/refresh a marked block in an instruction file
  target="$1"
  mkdir -p "$(dirname "$target")"
  [ -f "$target" ] || : > "$target"
  start='<!-- compactprompt:start -->'
  end='<!-- compactprompt:end -->'
  if grep -qF "$start" "$target"; then
    awk -v s="$start" -v e="$end" '
      $0==s{skip=1} skip==0{print} $0==e{skip=0}
    ' "$target" > "$target.tmp" && mv "$target.tmp" "$target"
  fi
  { echo "$start"; echo; cat "$BODY"; echo; echo "$end"; } >> "$target"
}

write_codex() {
  if [ "$SCOPE" = "user" ]; then t="$HOME/.codex/AGENTS.md"; else t="$PWD/AGENTS.md"; fi
  upsert_block "$t"
  echo "  Codex AGENTS.md    -> $t"
}

write_gemini() {
  if [ "$SCOPE" = "user" ]; then t="$HOME/.gemini/GEMINI.md"; else t="$PWD/GEMINI.md"; fi
  upsert_block "$t"
  echo "  Gemini GEMINI.md   -> $t"
}

if [ "$want_claude" -eq 1 ]; then write_skill; fi
if [ "$want_codex"  -eq 1 ]; then write_codex; fi
if [ "$want_cursor" -eq 1 ]; then write_cursor; fi
if [ "$want_gemini" -eq 1 ]; then write_gemini; fi

cat <<'EOF'

Done. For structured tool-calling, also set up the MCP server:
  pip install 'compactprompt[mcp]'
  # then register the `compactprompt-mcp` command in your tool
  # (see agent-skills/README.md for per-tool MCP config)
EOF
