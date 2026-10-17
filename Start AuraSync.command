#!/bin/bash
# Double-click in Finder to start AuraSync. Close this window (or press Ctrl+C) to stop it.
# First time only: right-click the file > Open > Open (macOS asks once for downloaded scripts).

cd "$(dirname "$0")" || exit 1
API="http://127.0.0.1:8000"
PY=python3

echo "=============================================="
echo "  AuraSync — starting"
echo "=============================================="

if ! command -v $PY >/dev/null 2>&1; then
  echo "python3 was not found. Install Python 3 from python.org, then try again."
  read -r -p "Press Enter to close..." _; exit 1
fi

open_frontend() { open "$API/app"; }

# Already running? Just open the page (unless it is an older version without /app).
if curl -s -o /dev/null "$API/health"; then
  if [ "$(curl -s -o /dev/null -w '%{http_code}' "$API/app")" = "200" ]; then
    echo "AuraSync is already running. Opening the wardrobe page."
    open_frontend
    exit 0
  fi
  echo "An older AuraSync is still running in another Terminal window."
  echo "Close that window, then double-click Start AuraSync.command again."
  read -r -p "Press Enter to close..." _
  exit 1
fi

# 1) Settings file: free local model by default (no API key needed)
if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  echo "[1/4] Created .env (free local model: Ollama qwen3:4b-instruct)"
else
  echo "[1/4] Using existing .env"
fi

# 2) Local model server
ollama_up() { curl -s -o /dev/null http://localhost:11434/api/tags; }
if grep -qi '^AURASYNC_LLM_PROVIDER=ollama' .env 2>/dev/null; then
  # Context window for the model (tool definitions need more than Ollama's 4k default).
  # Set the way Ollama documents for the Mac app: launchctl setenv + restart the app.
  CTX=$(grep -i '^AURASYNC_OLLAMA_CONTEXT=' .env | tail -n 1 | cut -d= -f2 | tr -dc '0-9')
  CTX=${CTX:-8192}
  if [ "$(launchctl getenv OLLAMA_CONTEXT_LENGTH 2>/dev/null)" != "$CTX" ]; then
    launchctl setenv OLLAMA_CONTEXT_LENGTH "$CTX"
    if ollama_up; then
      echo "[2/4] Restarting Ollama with a ${CTX}-token context..."
      osascript -e 'tell application "Ollama" to quit' >/dev/null 2>&1
      for i in $(seq 1 15); do ollama_up || break; sleep 1; done
    fi
  fi
  if ! ollama_up; then
    echo "[2/4] Starting the Ollama app..."
    open -a Ollama 2>/dev/null
    for i in $(seq 1 30); do ollama_up && break; sleep 1; done
  fi
  if ollama_up; then
    echo "[2/4] Ollama is running (context ${CTX} tokens)"
  else
    echo "[2/4] WARNING: Ollama is not responding. Open the Ollama app from Applications; the chat needs it."
  fi
else
  echo "[2/4] Using the OpenAI API (see .env)"
fi

# 3) Python packages: install only what is missing, into the Python you already use
missing=$($PY - <<'EOF'
import importlib.util
pkgs = {"fastapi": "fastapi", "uvicorn": "uvicorn[standard]", "dotenv": "python-dotenv", "multipart": "python-multipart",
        "pandas": "pandas", "numpy": "numpy", "scipy": "scipy", "sklearn": "scikit-learn", "lightgbm": "lightgbm",
        "PIL": "Pillow", "openai": "openai"}
print(" ".join(p for m, p in pkgs.items() if importlib.util.find_spec(m) is None))
EOF
)
if [ -n "$missing" ]; then
  echo "[3/4] Installing missing packages: $missing"
  user_flag=$($PY -c 'import sys; print("" if sys.prefix != sys.base_prefix else "--user")')
  $PY -m pip install $user_flag $missing || { echo "Package install failed (see above)."; read -r -p "Press Enter to close..." _; exit 1; }
else
  echo "[3/4] Python packages OK"
fi

# 4) Start the API, then open the page once it answers
echo "[4/4] Starting the API (first start can take ~30 s while models load)..."
(
  for i in $(seq 1 120); do curl -s -o /dev/null "$API/health" && break; sleep 1; done
  if curl -s -o /dev/null "$API/health"; then
    echo ""
    curl -s "$API/chat/status" | $PY -c '
import json, sys
try:
    s = json.load(sys.stdin)
except Exception:
    sys.exit(0)
print("LLM: %s / %s" % (s.get("provider"), s.get("model")))
miss = [a + "." + n for a, flags in s.get("artifacts", {}).items() for n, ok in flags.items() if not ok and n != "customer_personas_legacy"]
print("Not loaded (those answers will say unavailable): " + (", ".join(miss) if miss else "none"))
print("Data folder: %s" % s.get("data_dir"))
print("Wardrobe page: %s" % (s.get("frontend") or "NOT FOUND (see README)"))
'
    if grep -qi '^AURASYNC_LLM_PROVIDER=ollama' .env 2>/dev/null; then
      echo ""
      echo "Loading the language model into memory (first time can take ~30 s)..."
      warm=$(curl -s --max-time 240 "$API/chat/warmup")
      case "$warm" in
        *'"loaded":true'*) echo "Language model ready." ;;
        *) echo "WARNING: could not load the language model: $warm" ;;
      esac
    fi
    echo ""
    echo ">>> AuraSync is ready. Opening the wardrobe page. Keep this window open while you use it."
    open_frontend
  fi
) &

exec $PY -m uvicorn main:app --port 8000
