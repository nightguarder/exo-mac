#!/bin/bash
source .venv/bin/activate

# 1. Stop previous instances
echo "Stopping any running Exo Brain instances..."
if [ -f "./stop_brain.sh" ]; then
    ./stop_brain.sh
else
    echo "Warning: stop_brain.sh not found. Attempting to kill by pattern."
    pkill -f "python -m exo.main"
fi

# Wait for ports to be released
sleep 2

DEFAULT_MODEL="llama-3.1-8b"
MODEL=${1:-$DEFAULT_MODEL}

# 2. Start new instance
echo "Starting Exo Brain with model: $MODEL"
.venv/bin/python -u -m exo.main --default-model "$MODEL" --inference-engine mlx --chatgpt-api-port 52415 > exo.log 2>&1 &
PID=$!
echo $PID > exo_pid.txt
echo "Exo Brain started with PID: $PID"

# 3. Wait and Open Frontend
echo "Waiting for server to initialize..."
sleep 5

echo "Opening frontend at http://localhost:52415..."
# MacOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    open "http://localhost:52415"
# Linux
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v xdg-open > /dev/null; then
        xdg-open "http://localhost:52415"
    fi
else
    echo "Please open http://localhost:52415 in your browser."
fi
