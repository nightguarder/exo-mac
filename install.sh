#!/usr/bin/env bash

# Exo-MLX Installation Script for Apple Silicon Macs
set -e

echo "🍎 Exo-MLX Setup for Apple Silicon..."

# Check if running on Apple Silicon Mac
if [[ "$(uname)" != "Darwin" ]] || [[ "$(uname -m)" != "arm64" ]]; then
    echo "❌ Error: This exo-mlx fork is exclusively for Apple Silicon Macs (M1, M2, M3, M4)"
    echo "   Detected: $(uname) $(uname -m)"
    echo "   For other platforms, use the original exo repository."
    exit 1
fi

# Check and install Python 3.12 (recommended) or fallback to available version
if command -v python3.12 &>/dev/null; then
    echo "✅ Python 3.12 found, using python3.12..."
    PYTHON_CMD="python3.12"
elif command -v python3.11 &>/dev/null; then
    echo "⚠️  Python 3.11 found, using python3.11 (3.12 recommended)..."
    PYTHON_CMD="python3.11"
elif command -v python3 &>/dev/null; then
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    echo "⚠️  Using system Python: $PYTHON_VERSION (3.12 recommended)..."
    PYTHON_CMD="python3"
else
    echo "❌ Error: No Python 3 installation found"
    echo "   Please install Python 3.11+ using Homebrew:"
    echo "   brew install python@3.12"
    exit 1
fi

echo "🔧 Creating virtual environment..."
$PYTHON_CMD -m venv .venv

echo "🔧 Activating virtual environment..."
source .venv/bin/activate

echo "📦 Upgrading pip..."
pip install --upgrade pip

echo "🤖 Installing exo-mlx with Apple Silicon optimizations..."
pip install -e .

echo "⚡ Configuring MLX and Apple Silicon optimizations..."
if [[ -f "./configure_mlx.sh" ]]; then
    chmod +x ./configure_mlx.sh
    ./configure_mlx.sh
else
    echo "⚠️  Warning: configure_mlx.sh not found. Creating basic MLX configuration..."
    
    # Basic MLX configuration
    echo "import mlx.core as mx" > test_mlx.py
    echo "print(f'MLX Metal Support: {mx.metal.is_available()}')" >> test_mlx.py
    echo "print(f'MLX Device: {mx.default_device()}')" >> test_mlx.py
    
    if python test_mlx.py; then
        echo "✅ MLX configuration successful"
    else
        echo "⚠️  MLX configuration may need attention"
    fi
    rm -f test_mlx.py
fi

echo ""
echo "🎉 Exo-MLX installation complete!"
echo ""
echo "� To start the Exo Brain:"
echo "   ./start_brain.sh"
echo ""
echo "🌐 Access the frontend at:"
echo "   http://localhost:52415/"
