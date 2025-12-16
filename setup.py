import sys
import platform
from setuptools import find_packages, setup

# Check if running on Apple Silicon Mac (required for this fork)
if not (sys.platform.startswith("darwin") and platform.machine() == "arm64"):
    raise RuntimeError(
        "This exo-mlx fork is exclusively for Apple Silicon Macs (M1, M2, M3, M4).\n"
        f"Detected platform: {sys.platform} {platform.machine()}\n"
        "For other platforms, use the original exo repository."
    )

# Core requirements for Apple Silicon Macs only
install_requires = [
    # Core networking and API
    "aiohttp==3.10.11",
    "aiohttp_cors==0.7.0", 
    "aiofiles==24.1.0",
    "grpcio==1.70.0",
    "grpcio-tools==1.70.0",
    
    # Web interface
    "Jinja2==3.1.4",
    
    # Core ML dependencies  
    "numpy==2.0.0",
    "pydantic==2.9.2",
    
    # Apple Silicon ML frameworks (primary)
    "mlx==0.26.1",
    "mlx-lm==0.21.1",
    
    # PyTorch with Metal support (fallback)
    "torch>=2.1.0",
    "transformers==4.46.3",
    
    # Utilities
    "requests==2.32.3",
    "rich==13.7.1",
    "tqdm==4.66.4",
    "psutil==6.0.0",
    "uuid==1.30",
    "uvloop==0.21.0",
    "scapy>=2.7.0",
    
    # Development and monitoring
    "prometheus-client==0.20.0",
    "protobuf>=5.29.0",
]

# Optional extras for development
extras_require = {
    "dev": [
        "yapf==0.40.2",
        "black==23.12.1", 
        "mypy==1.8.0",
        "pytest==7.4.4",
        "pytest-asyncio==0.23.2",
    ],
    "performance": [
        "psutil>=6.0.0",
        "py-cpuinfo>=9.0.0",
    ],
}

setup(
  name="exo",
  version="0.0.1",
  packages=find_packages(),
  install_requires=install_requires,
  extras_require=extras_require,
  package_data={"exo": ["tinychat/**/*"]},
  entry_points={"console_scripts": ["exo = exo.main:run"]},
)
