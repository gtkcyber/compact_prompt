# Minimal image that runs the compactprompt MCP server over stdio.
#
# Used by Glama (and any MCP host) to start the server and verify it responds
# to introspection (initialize / tools-list). Only the core library plus the
# MCP SDK are needed to start and list tools; the heavier engines (LLMLingua,
# embeddings, etc.) load lazily only when a tool that needs them is called.
FROM python:3.12-slim

WORKDIR /app
COPY . /app

# The package version is normally derived from git tags by setuptools-scm; pin
# it here so the image builds without git history in the build context.
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.5.2

RUN pip install --no-cache-dir ".[mcp]"

# The MCP server speaks JSON-RPC over stdio.
ENTRYPOINT ["compactprompt-mcp"]
