#!/usr/bin/env bash

set -e  # Exit on any error

echo "🚀 Installing exo..."

# Check for Python version
if command -v python3.12 &>/dev/null; then
    echo "✅ Python 3.12 is installed, proceeding with python3.12..."
    PYTHON_CMD="python3.12"
elif command -v python3.11 &>/dev/null; then
    echo "✅ Python 3.11 is installed, proceeding with python3.11..."
    PYTHON_CMD="python3.11"
elif command -v python3 &>/dev/null; then
    PYTHON_VERSION=$($python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
    echo "⚠️  The recommended version of Python to run exo with is Python 3.12, but Python $PYTHON_VERSION is installed. Proceeding with Python $PYTHON_VERSION"
    PYTHON_CMD="python3"
else
    echo "❌ Error: Python 3 is not installed. Please install Python 3.11 or higher."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing old one..."
    rm -rf .venv
fi

$PYTHON_CMD -m venv .venv
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Check if we're on Apple Silicon Mac for MLX support
if [[ "$OSTYPE" == "darwin"* ]] && [[ "$(uname -m)" == "arm64" ]]; then
    echo "🍎 Detected Apple Silicon Mac - MLX support will be installed"
    echo "📥 Installing MLX packages..."
    pip install mlx==0.26.1 mlx-lm==0.21.1
fi

# Install exo in development mode
echo "📥 Installing exo and dependencies..."
pip install -e .

echo "✅ Installation complete!"
echo ""
echo "🎉 To use exo, activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "📚 Then run exo with:"
echo "   python exo/main.py"
echo ""
echo "🔧 For local mode with MLX (Apple Silicon):"
echo "   python exo/main.py --local-mode --inference-engine mlx"
