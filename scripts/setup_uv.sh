#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ##########
# Install UV
# ##########

# Expected installation path for uv
# See: https://docs.astral.sh/uv/configuration/installer/#changing-the-install-path
LOCAL_BIN="${LOCAL_BIN:-$HOME/.local/bin}"

# Ensure LOCAL_BIN exists and is on PATH
mkdir -p "$LOCAL_BIN"
if [[ ":$PATH:" != *":$LOCAL_BIN:"* ]]; then
  export PATH="$LOCAL_BIN:$PATH"
fi

VENV_DIR=".venv"

# Install or update uv
if ! command -v uv &>/dev/null; then
  echo "uv not found – installing to $LOCAL_BIN"
  curl -fsSL https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$LOCAL_BIN" sh
else
  current=$(uv --version | awk '{print $2}')
  echo "Found uv v$current"
  if command -v jq &>/dev/null; then
    latest=$(curl -fsS https://api.github.com/repos/astral-sh/uv/releases/latest \
             | jq -r .tag_name)
    if [[ "$current" != "$latest" ]]; then
      echo "Updating uv: $current → $latest"
      uv self update
    fi
  fi
fi

# Bootstrap root .venv
echo "Bootstrapping root .venv in folder $VENV_DIR"
uv venv "$VENV_DIR"
uv sync --group all --active

echo "Done! Root environment is ready in: $VENV_DIR"

# Install pre-commit hooks
echo "Installing pre-commit hooks"
uv run pre-commit install

# After detecting PATH lacked LOCAL_BIN…
if [[ ":$PATH:" != *":$LOCAL_BIN:"* ]]; then
  echo "Note: added $LOCAL_BIN to PATH for this session."
  echo "To make it permanent, add to your shell profile:"
  echo "  export PATH=\"$LOCAL_BIN:\$PATH\""
fi
