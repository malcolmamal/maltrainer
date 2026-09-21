#!/usr/bin/env bash
# maltrainer web UI launcher (Linux / macOS / Git Bash).
#
# Starts the FastAPI backend (port 8008) in the background and the Angular UI (port 3003) in
# the foreground. Ctrl-C stops both. A server that is already listening on its port is left alone.
# On Windows, run_webui.bat is the better choice: it opens each server in its own window.
#
# Settings come from the environment, or from webui.local.sh next to this file (git-ignored).
# Copy webui.local.example.sh to webui.local.sh and edit it:
#   DATASETS_PATH        folder with one sub-folder per dataset   (default: ./datasets)
#   COMFYUI_MODELS_PATH  your ComfyUI "models" folder             (needed for Qwen-Image 2.1)
# Set WEBUI_NO_BROWSER=1 to skip opening the browser.

set -u
cd "$(dirname "$0")"

# shellcheck disable=SC1091
[ -f ./webui.local.sh ] && . ./webui.local.sh

[ -n "${DATASETS_PATH:-}" ] || echo "[maltrainer] DATASETS_PATH is not set, using ./datasets"
[ -n "${COMFYUI_MODELS_PATH:-}" ] || echo "[maltrainer] COMFYUI_MODELS_PATH is not set: Qwen-Image 2.1 training will report its model files as missing"
export DATASETS_PATH="${DATASETS_PATH:-}" COMFYUI_MODELS_PATH="${COMFYUI_MODELS_PATH:-}"
[ -n "$DATASETS_PATH" ] || unset DATASETS_PATH
[ -n "$COMFYUI_MODELS_PATH" ] || unset COMFYUI_MODELS_PATH

if [ -x venv/bin/python ]; then PYTHON=venv/bin/python
elif [ -x venv/Scripts/python.exe ]; then PYTHON=venv/Scripts/python.exe
elif command -v python3 >/dev/null 2>&1; then PYTHON=python3
else PYTHON=python; fi

if ! command -v npm >/dev/null 2>&1; then
    echo "[maltrainer] npm was not found. Install Node.js from https://nodejs.org and run this again."
    exit 1
fi

# listening PORT: succeeds when something is listening on PORT (any address, IPv4 or IPv6)
listening() {
    if command -v ss >/dev/null 2>&1; then
        ss -ltn 2>/dev/null | grep -qE "[:.]$1[[:space:]]"
    elif command -v netstat >/dev/null 2>&1; then
        netstat -an 2>/dev/null | grep -E "[:.]$1[[:space:]]" | grep -qi listen
    else
        (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
    fi
}

BACKEND_PID=""
cleanup() {
    [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null
    return 0
}
trap cleanup EXIT INT TERM

if listening 8008; then
    echo "[maltrainer] Backend is already running on port 8008, leaving it alone."
else
    if ! "$PYTHON" -c "import uvicorn, fastapi" >/dev/null 2>&1; then
        echo "[maltrainer] uvicorn / fastapi are not installed for $PYTHON. Run: pip install -r requirements.txt"
        exit 1
    fi
    echo "[maltrainer] Starting backend on http://127.0.0.1:8008"
    "$PYTHON" -m uvicorn backend.app:app --host 127.0.0.1 --port 8008 &
    BACKEND_PID=$!
fi

open_browser() {
    [ -n "${WEBUI_NO_BROWSER:-}" ] && return 0
    case "$(uname -s)" in
        Darwin) open "$1" ;;
        MINGW*|MSYS*|CYGWIN*) cmd.exe //c start "" "$1" ;;
        *) command -v xdg-open >/dev/null 2>&1 && xdg-open "$1" >/dev/null 2>&1 ;;
    esac
    return 0
}

if listening 3003; then
    echo "[maltrainer] UI is already running on port 3003, leaving it alone."
    open_browser http://localhost:3003
    # keep the backend we started alive until Ctrl-C
    [ -n "$BACKEND_PID" ] && wait "$BACKEND_PID"
    exit 0
fi

if [ ! -d frontend/node_modules ]; then
    echo "[maltrainer] First run: installing frontend dependencies, this takes a minute..."
    (cd frontend && npm install) || exit 1
fi

echo "[maltrainer] Starting UI on http://localhost:3003 (Ctrl-C stops everything)"
# a freshly started dev server needs a few seconds for its first build
( sleep 12; open_browser http://localhost:3003 ) &
(cd frontend && npm start)
