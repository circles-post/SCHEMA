"""Launcher shim for the ToolUniverse SMCP stdio server.

Replaces the ``uv run tooluniverse-smcp-stdio ...`` command used in
``test_agent_debug.py`` — we don't have ``uv`` on this host, but the
``tooluniverse`` package is installed in the agentdebug conda env, so we
can call the entry point directly.

Usage (matches the original CLI)::

    python /path/to/run_tooluniverse_stdio.py --exclude-tool-types PackageTool --compact-mode
"""

from __future__ import annotations

import sys

# The CLI parser inside run_stdio_server inspects sys.argv[0] for its help/error
# messages; make it match the original entry-point name.
sys.argv[0] = "tooluniverse-smcp-stdio"

from tooluniverse.smcp_server import run_stdio_server

if __name__ == "__main__":
    run_stdio_server()
