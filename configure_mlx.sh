#!/usr/bin/env bash

# MLX Configuration Script for Apple Silicon
set -e

echo "⚡ Configuring MLX for Apple Silicon..."

# Check if we're in virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "❌ Error: Virtual environment not activated"
    echo "   Run: source .venv/bin/activate"
    exit 1
fi

# Verify Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "❌ Error: Not running on Apple Silicon"
    exit 1
fi

echo "🔍 Detecting Apple Silicon chip..."
CHIP_INFO=$(system_profiler SPHardwareDataType | grep "Chip:" | awk -F': ' '{print $2}')
echo "   Detected: $CHIP_INFO"

# Get the total memory in MB for GPU optimization
TOTAL_MEM_MB=$(($(sysctl -n hw.memsize) / 1024 / 1024))
echo "   Total Memory: $TOTAL_MEM_MB MB"

# Install PyTorch with Metal support if not already installed
echo "🔧 Ensuring PyTorch with Metal Performance Shaders support..."
python3 -c "import torch" 2>/dev/null || pip install torch torchvision torchaudio

# Calculate optimal GPU memory limits (Apple Silicon optimization)
EIGHTY_PERCENT=$(($TOTAL_MEM_MB * 80 / 100))
MINUS_5GB=$((($TOTAL_MEM_MB - 5120)))
SEVENTY_PERCENT=$(($TOTAL_MEM_MB * 70 / 100))
MINUS_8GB=$((($TOTAL_MEM_MB - 8192)))

# Set WIRED_LIMIT_MB to higher value
if [ $EIGHTY_PERCENT -gt $MINUS_5GB ]; then
  WIRED_LIMIT_MB=$EIGHTY_PERCENT
else
  WIRED_LIMIT_MB=$MINUS_5GB
fi

# Set WIRED_LWM_MB to higher value  
if [ $SEVENTY_PERCENT -gt $MINUS_8GB ]; then
  WIRED_LWM_MB=$SEVENTY_PERCENT
else
  WIRED_LWM_MB=$MINUS_8GB
fi

echo "🚀 Optimizing GPU memory limits..."
echo "   Maximum limit (iogpu.wired_limit_mb): $WIRED_LIMIT_MB MB"
echo "   Lower bound (iogpu.wired_lwm_mb): $WIRED_LWM_MB MB"

# Apply the values with sysctl, but check if we're already root
if [ "$EUID" -eq 0 ]; then
  sysctl -w iogpu.wired_limit_mb=$WIRED_LIMIT_MB
  sysctl -w iogpu.wired_lwm_mb=$WIRED_LWM_MB
else
  # Try without sudo first, fall back to sudo if needed
  sysctl -w iogpu.wired_limit_mb=$WIRED_LIMIT_MB 2>/dev/null || \
    sudo sysctl -w iogpu.wired_limit_mb=$WIRED_LIMIT_MB
  sysctl -w iogpu.wired_lwm_mb=$WIRED_LWM_MB 2>/dev/null || \
    sudo sysctl -w iogpu.wired_lwm_mb=$WIRED_LWM_MB
fi

# Test MLX and PyTorch installations
echo "🧪 Testing MLX and PyTorch installations..."
echo "✅ MLX configuration complete (test skipped)"

# Set MLX environment variables for optimal performance
echo ""
echo "🔧 Setting MLX environment variables..."
if ! grep -q "MLX_METAL_BUFFER_CACHE_SIZE" ~/.zshrc 2>/dev/null; then
    echo "export MLX_METAL_BUFFER_CACHE_SIZE=1024" >> ~/.zshrc
fi
if ! grep -q "MLX_METAL_USE_UNIFIED_MEMORY" ~/.zshrc 2>/dev/null; then
    echo "export MLX_METAL_USE_UNIFIED_MEMORY=1" >> ~/.zshrc
fi



echo ""
echo "✅ MLX configuration complete!"
echo ""

echo "🔄 Restart terminal to load environment variables"
echo "🚀 Ready for exo-mlx!"