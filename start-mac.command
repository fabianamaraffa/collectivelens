#!/bin/bash
# Double-click this file to start CollectiveLens. It sets everything up the first time.
# (If macOS says it cannot be opened, right-click the file, choose Open, then Open again.)

cd "$(dirname "$0")" || exit 1
echo "CollectiveLens"
echo "--------------"

pause_and_exit() {
  echo
  read -r -p "Press Return to close this window. " _
  exit "${1:-1}"
}

# 1. Find a Python that works with PyTorch and Gradio (3.12 is the best choice).
PY=""
for candidate in python3.12 python3.13 python3.11 python3.10 \
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
    /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)' 2>/dev/null; then
      PY="$candidate"
      break
    fi
  fi
done

if [ -z "$PY" ]; then
  echo "Python 3.12 was not found on this Mac."
  echo "Install it from https://www.python.org/downloads/ (choose a 3.12 release),"
  echo "then double-click this file again. Python 3.14 and newer are not supported yet."
  open "https://www.python.org/downloads/" 2>/dev/null
  pause_and_exit 1
fi
echo "Using $("$PY" --version) ($PY)"

# 2. Create the private Python environment (first time only).
if [ ! -x .venv/bin/python ]; then
  echo "Creating the environment..."
  "$PY" -m venv .venv || { echo "Could not create the environment."; pause_and_exit 1; }
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 3. Install the requirements (first time, or when requirements.txt changes).
if [ ! -f .venv/.installed ] || [ requirements.txt -nt .venv/.installed ]; then
  echo "Installing the requirements. This takes several minutes the first time (it downloads PyTorch)."
  python -m pip install --quiet --upgrade pip
  if python -m pip install -r requirements.txt; then
    touch .venv/.installed
  else
    echo "The installation failed. Copy the messages above and send them to whoever is helping you."
    pause_and_exit 1
  fi
fi

# 4. Ask for the Anthropic API key once (optional). It is saved only in the file .env on this computer.
if [ ! -f .env ] && [ -z "$ANTHROPIC_API_KEY" ]; then
  echo
  echo "Written answers and summaries need an Anthropic API key (it starts with sk-ant-)."
  echo "Paste it here and press Return. You will not see it as you paste. To skip, just press Return."
  read -r -s -p "API key: " KEY
  echo
  if [ -n "$KEY" ]; then
    ( umask 077; printf 'ANTHROPIC_API_KEY=%s\n' "$KEY" > .env )
    echo "Saved to .env on this computer."
  else
    : > .env
    echo "Skipped. Search and charts still work; answers will show the matching passages."
  fi
fi

# 5. Start the app. The browser opens by itself when it is ready.
echo
echo "Starting the app. The first start downloads the search model (about 2 GB), please wait."
echo "To stop the app, close this window or press Control+C."
export COLLECTIVELENS_OPEN_BROWSER=1
python app.py
pause_and_exit $?
