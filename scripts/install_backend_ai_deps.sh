#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/gsdata/work/nlp_tool"
VENV_PY="$ROOT/backend/.venv/bin/python"
VENV_PIP="$ROOT/backend/.venv/bin/pip"

export NO_PROXY="*"
export no_proxy="*"
export PIP_DISABLE_PIP_VERSION_CHECK=1

"$VENV_PIP" install -r "$ROOT/backend/requirements.txt"
