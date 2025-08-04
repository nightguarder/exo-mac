<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/exo-logo-black-bg.jpg">
  <img alt="exo logo" src="/docs/exo-logo-transparent.png" width="50%" height="50%">
</picture>

# exo: Run your own AI cluster at home with everyday devices

**Apple Silicon Optimized Fork**

*Maintained as an optimized fork of [exo labs](https://github.com/exo-explore/exo) by [exo labs](https://x.com/exolabs)*

[![GitHub Repo stars](https://img.shields.io/github/stars/exo-explore/exo)](https://github.com/exo-explore/exo/stargazers)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-Optimized-000000?logo=apple)](https://support.apple.com/en-us/116943)
[![MLX](https://img.shields.io/badge/MLX-Accelerated-FF6B00)](https://github.com/ml-explore/mlx)

</div>

---

**Unify your Apple devices into one powerful GPU: iPhone, iPad, Mac, and more!**

<div align="center">
  <h3>🍎 This fork focuses exclusively on Apple Silicon optimization with enhanced MLX and HuggingFace support</h3>
</div>

## 🚀 Apple Silicon Performance

This fork of the original [exo project](https://github.com/exo-explore/exo) by [exo labs](https://x.com/exolabs) is **optimized specifically for Apple Silicon devices**. Experience lightning-fast AI inference with native Metal acceleration and unified memory architecture.

### Why This Fork?

- **🚀 60-80 tokens/sec** on M2 Pro with DeepSeek models
- **⚡ MLX framework** - Apple's native AI acceleration
- **💾 Unified memory** - seamless GPU/CPU sharing
- **🔋 Energy efficient** - optimized for laptops and mobile
- **🛡️ Dual engines** - MLX primary + HuggingFace fallback

## Get Involved

This is an **experimental fork** focused on Apple Silicon optimization. The original [exo labs](https://x.com/exolabs) team maintains the main project with broader platform support.

**Original Project**: [github.com/exo-explore/exo](https://github.com/exo-explore/exo)
**Community**: [Discord](https://discord.gg/EUnjGpsmWw) | [Telegram](https://t.me/+Kh-KqHTzFYg3MGNk) | [X](https://x.com/exolabs)

This fork welcomes contributions for:
- 🍎 Apple Silicon performance improvements
- 📱 iOS/iPadOS support (A-series chips)  
- 🔧 MLX engine enhancements
- 🛡️ HuggingFace integration improvements

## Features

### Wide Model Support

This fork enhances the original exo's model support with optimized Apple Silicon inference:

- **MLX Models** (Primary): LLaMA, Mistral, Qwen, DeepSeek, Phi - all with Metal acceleration
- **HuggingFace Models** (Fallback): CPU-only inference for maximum compatibility
- **Vision Models**: LlaVA and other multimodal models

### Dynamic Model Partitioning

Like the original exo, this fork [optimally splits up models](exo/topology/ring_memory_weighted_partitioning_strategy.py) based on the current network topology and device resources available. This enables you to run larger models than you would be able to on any single device.

### Automatic Device Discovery

This fork maintains exo's [automatic discovery](https://github.com/exo-explore/exo/blob/945f90f676182a751d2ad7bcf20987ab7fe0181e/exo/orchestration/node.py#L154) capabilities with Apple-optimized networking. Zero manual configuration.

### ChatGPT-compatible API

Enhanced [ChatGPT-compatible API](exo/api/chatgpt_api.py) with improved Apple Silicon performance. It's a [one-line change](examples/chatgpt_api.sh) in your application to run models on your own hardware.

### Device Equality

Following exo's philosophy, this fork does not use a master-worker architecture. Instead, devices [connect p2p](https://github.com/exo-explore/exo/blob/945f90f676182a751d2ad7bcf20987ab7fe0181e/exo/orchestration/node.py#L161). As long as a device is connected somewhere in the network, it can be used to run models.

Supports the same [partitioning strategies](exo/topology/partitioning_strategy.py) as the original exo to split up a model across devices. The default is [ring memory weighted partitioning](exo/topology/ring_memory_weighted_partitioning_strategy.py) optimized for Apple Silicon unified memory.

!["A screenshot of exo running 5 nodes](docs/exo-screenshot.jpg)

## Installation

### Prerequisites

- **Apple Silicon device** (M1, M2, M3, or M4) - this fork is optimized exclusively for Apple Silicon
- **Python>=3.12.0** (required because of [asyncio issues](https://github.com/exo-explore/exo/issues/5) in previous versions)
- **macOS Sonoma 14.0+** (Sequoia recommended for best MLX performance)

### Hardware Requirements

The only requirement to run exo is to have enough memory across all your Apple devices to fit the entire model into memory. For example, if you are running llama 3.1 8B (fp16), you need 16GB of memory across all devices. These Apple Silicon configurations work great:

- **2 x M3 MacBook Airs (8GB each)** - Perfect for 7B models
- **1 x M2 MacBook Pro (16GB)** - Great for single-device 7B models  
- **1 x Mac Studio M2 Ultra (64GB)** - Can run 70B models solo
- **3 x Mac Mini M4 (8GB each)** - Distributed 7B+ models

exo is designed to run on devices with heterogeneous capabilities. You can mix different Apple Silicon devices - M1 MacBook Air with M4 Mac Studio, etc.

### From source

```bash
# Clone this Apple Silicon optimized fork
git clone https://github.com/nightguarder/exo-local.git
cd exo-local
pip install -e .
# alternatively, with venv
source install.sh
```

### Troubleshooting

- MLX has an [install guide](https://ml-explore.github.io/mlx/build/html/install.html) with troubleshooting steps for Apple Silicon.

### Performance

For optimal performance on Apple Silicon:

1. **Upgrade to the latest version of macOS Sequoia** for best MLX support.
2. **Run `./configure_mlx.sh`** - This runs commands to optimize GPU memory allocation on Apple Silicon.
3. **Close memory-intensive apps** when running large models.

## Quick Start

### Single Command Setup

```bash
# Start exo with optimal settings
exo
```

### With Custom API Port

```bash
# Start with ChatGPT-compatible API on port 8000
python -m exo.main --chatgpt-api-port 8000
```

**That's it!** Your Apple devices will automatically find each other and create a distributed AI cluster.

- **Web Interface**: http://localhost:52415
- **ChatGPT API**: http://localhost:8000/v1/chat/completions (if using custom port)

### Test Your Setup

```bash
# Test with optimized DeepSeek model
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-distill-qwen-1.5b",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50
  }'
```

## Documentation

### Example Usage on Multiple Apple Silicon Devices

#### Device 1:
```sh
exo
```

#### Device 2:
```sh
exo
```

That's it! No configuration required - exo will automatically discover the other Apple devices.

exo starts a ChatGPT-like WebUI (powered by [tinygrad tinychat](https://github.com/tinygrad/tinygrad/tree/master/examples/tinychat)) on http://localhost:52415

For developers, exo also starts a ChatGPT-compatible API endpoint on http://localhost:52415/v1/chat/completions. Examples with curl:

#### DeepSeek R1 Distill (Recommended for Apple Silicon):
```sh
curl http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "deepseek-r1-distill-qwen-1.5b",
     "messages": [{"role": "user", "content": "What is the meaning of exo?"}],
     "temperature": 0.7
   }'
```

#### Llama 3.2 3B:
```sh
curl http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "llama-3.2-3b",
     "messages": [{"role": "user", "content": "What is the meaning of exo?"}],
     "temperature": 0.7
   }'
```

#### Llama 3.1 70B (Distributed across multiple Apple devices):
```sh
curl http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "llama-3.1-70b",
     "messages": [{"role": "user", "content": "What is the meaning of exo?"}],
     "temperature": 0.7
   }'
```

#### Llava 1.5 7B (Vision Language Model):
```sh
curl http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "llava-1.5-7b-hf",
     "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What are these?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "http://images.cocodataset.org/val2017/000000039769.jpg"
            }
          }
        ]
      }
    ],
     "temperature": 0.0
   }'
```

### Example Usage with Mixed Platforms (Apple Silicon + Others)

This fork maintains compatibility with the original exo's multi-platform support:

#### Device 1 (Apple Silicon with MLX):
```sh
exo
```

#### Device 2 (Linux with tinygrad):
```sh
exo
```

Note: **MLX** and **tinygrad** are interoperable! The Apple Silicon device will use MLX for optimal performance while other devices can use tinygrad.

### Single Device Usage

```sh
# Quick model test
exo run llama-3.2-3b

# With custom prompt
exo run deepseek-r1-distill-qwen-1.5b --prompt "Explain quantum computing for beginners"
```

### Model Storage

Models by default are stored in `~/.cache/exo/downloads`.

You can set a different model storage location by setting the `EXO_HOME` env var.

## Model Downloading

Models are downloaded from Hugging Face. If you are running exo in a country with strict internet censorship, you may need to download the models manually and put them in the `~/.cache/exo/downloads` directory.

To download models from a proxy endpoint, set the `HF_ENDPOINT` environment variable. For example, to run exo with the huggingface mirror endpoint:

```sh
HF_ENDPOINT=https://hf-mirror.com exo
```

## Debugging

Enable debug logs with the DEBUG environment variable (0-9).

```sh
DEBUG=9 exo
```

For enhanced debugging with this fork's dual engine support:

```sh
# General debugging
DEBUG=9 exo

# MLX-specific debugging
MLX_DEBUG=1 exo

# HuggingFace engine debugging  
HF_DEBUG=true exo
```

## Performance Optimization

This fork includes several Apple Silicon optimizations:

```sh
# Run after installation for optimal performance
./configure_mlx.sh

# Environment variables for tuning
export MLX_MEMORY_POOL=1         # Enable MLX memory pooling
export OMP_NUM_THREADS=4         # CPU thread optimization
```

### Apple Silicon Performance Tips

1. **Update to macOS Sequoia** for best MLX performance
2. **Run ./configure_mlx.sh** after installation  
3. **Close memory-intensive apps** when running large models
4. **Use MLX models** (deepseek, llama, qwen) for best performance
5. **HuggingFace models** work as CPU fallback when needed

## Formatting

We use [yapf](https://github.com/google/yapf) to format the code. To format the code, first install the formatting requirements:

```sh
pip3 install -e '.[formatting]'
```

Then run the formatting script:

```sh
python3 format.py ./exo
```

## Known Issues

- On certain versions of Python on macOS, certificates may not installed correctly, potentially causing SSL errors (e.g., when accessing huggingface.co). To resolve this, run the `Install Certificates` command, typically as follows:

```sh
/Applications/Python\ 3.12/Install\ Certificates.command
```

- 🚧 iOS implementation is planned for this Apple Silicon fork but not yet available. Focus is currently on macOS Apple Silicon optimization.

## Inference Engines

This fork supports optimized inference engines:

- ✅ **[MLX](exo/inference/mlx/sharded_inference_engine.py)** (Primary - Apple Silicon optimized)
- ✅ **[HuggingFace](exo/inference/huggingface/inference.py)** (Fallback - CPU-only)  
- ✅ **[tinygrad](exo/inference/tinygrad/inference.py)** (Multi-platform compatibility)

For technical details, see [README-DEV.md](README-DEV.md).

## Discovery Modules

- ✅ [UDP](exo/networking/udp) - Automatic local network discovery
- ✅ [Manual](exo/networking/manual) - Manual IP configuration
- ✅ [Tailscale](exo/networking/tailscale) - VPN mesh networking
- 🚧 AirDrop (planned for Apple ecosystem)

## Peer Networking Modules

- ✅ [GRPC](exo/networking/grpc) - Primary networking protocol
- 🚧 Apple-specific optimizations planned

## Contributing to This Fork

This Apple Silicon-optimized fork welcomes contributions that enhance Apple device performance and compatibility. While maintaining the spirit of the original [exo project](https://github.com/exo-explore/exo), this fork focuses specifically on:

- 🍎 **Apple Silicon performance improvements**
- 📱 **iOS/iPadOS support** (A-series chips)
- 🔧 **MLX engine enhancements**
- 🛡️ **HuggingFace integration improvements**
- 🎨 **Apple-specific UI/UX optimizations**

### Development Setup

```bash
# Clone this Apple Silicon fork
git clone https://github.com/nightguarder/exo-local.git
cd exo-local
source install.sh

# Install development dependencies
pip install pytest yapf black mypy

# See detailed development guide
open README-DEV.md
```

### Relationship to Original Project

- **Upstream**: [exo-explore/exo](https://github.com/exo-explore/exo) by [exo labs](https://x.com/exolabs)
- **This Fork**: Apple Silicon optimization with enhanced MLX + HuggingFace support
- **Goal**: Maximize performance on Apple devices while maintaining compatibility

## Roadmap

### Apple Silicon Enhancements
- **iPhone/iPad Support** - Extend MLX optimization to A17/A18 Pro chips
- **AirDrop Discovery** - Native Apple device discovery via AirDrop protocol
- **Metal Optimization** - Direct Metal Performance Shaders integration
- **Neural Engine** - Integration with Apple's dedicated AI chip
- **iOS App** - Native iOS client application

### Performance Goals
- **100+ tokens/sec** on M4 Max devices
- **Sub-second startup** for common models  
- **Real-time streaming** for conversations
- **Multi-modal support** (text + images + audio)

## License

GPL v3 License - Same as original exo project. See [LICENSE](LICENSE) for details.

## Acknowledgments

- **[exo labs](https://x.com/exolabs)** and the original [exo project](https://github.com/exo-explore/exo) team
- **[Apple ML Research](https://github.com/ml-explore/mlx)** for the MLX framework
- **[HuggingFace](https://huggingface.co)** for the transformers ecosystem
- **Apple Silicon community** for testing and feedback on this fork

---

<div align="center">

**🚀 Ready to supercharge your Apple devices with AI?**

```bash
git clone https://github.com/nightguarder/exo-local.git && cd exo-local && source install.sh && exo
```

*Transform your Apple devices into a distributed AI cluster in under 60 seconds*

**📖 For technical details and development information, see [README-DEV.md](README-DEV.md)**

</div>
