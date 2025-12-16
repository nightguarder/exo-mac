#!/bin/bash

echo "Stopping all exo brain instances..."

# Kill by PID file if it exists
if [ -f exo_pid.txt ]; then
    PID=$(cat exo_pid.txt)
    if ps -p $PID > /dev/null 2>&1; then
        echo "Killing process from PID file: $PID"
        kill -9 $PID
    else
        echo "Process $PID from PID file is not running"
    fi
    rm -f exo_pid.txt
    echo "Removed PID file"
fi

# Kill all processes matching the pattern
echo "Killing all processes matching 'exo'..."
pkill -9 -f "python -m exo.main"
pkill -9 -f "bin/exo"

# Kill any process listening on port 52415
if lsof -t -i:52415 >/dev/null 2>&1; then
    echo "Killing remaining process on port 52415..."
    lsof -t -i:52415 | xargs kill -9
fi

# Clear logs
echo "Clearing exo logs..."
rm -f exo.log exo_server.log exo_debug.log

# Check if any processes are still running
if pgrep -f "python -m exo.main" > /dev/null; then
    echo "Warning: Some processes may still be running"
    pgrep -f "python -m exo.main"
else
    echo "All exo brain instances stopped successfully"
fi
