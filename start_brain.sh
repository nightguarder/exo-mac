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

# 2. Open Frontend (in background, after delay)
(
    sleep 5
    echo "Opening frontend at http://localhost:52415..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open "http://localhost:52415"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v xdg-open > /dev/null; then
            xdg-open "http://localhost:52415"
        fi
    fi
) &

# 3. Start new instance (Foreground)
echo "Starting Exo Brain with model: $MODEL"
# We use exec to replace the shell process with exo, so signals are passed correctly
exec .venv/bin/exo --default-model "$MODEL" --inference-engine mlx --chatgpt-api-port 52415
