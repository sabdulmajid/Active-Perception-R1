"""
setup.py — AeroRL build configuration.

Builds the optional CUDA IPC extension (aerorl_ipc_ext) when CUDA 12.4+ and
PyTorch with CUDA support are available.  Falls back gracefully if not.

Install (development, CUDA wheels):
    pip install -e .

Install (CPU / no CUDA extension):
    AERORL_NO_CUDA=1 pip install -e .

Build extension in-place (for testing):
    python setup.py build_ext --inplace
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Try to import PyTorch build helpers ──────────────────────────────────
try:
    import torch
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
    _HAS_TORCH = True
    _HAS_CUDA  = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH = False
    _HAS_CUDA  = False
    BuildExtension = None
    CUDAExtension   = None

from setuptools import setup, find_packages

# ── Package metadata ──────────────────────────────────────────────────────
HERE     = Path(__file__).parent
CSRC_DIR = HERE / "aerorl" / "extensions" / "csrc"

# ── CUDA extension configuration ─────────────────────────────────────────
_SKIP_CUDA = (
    os.environ.get("AERORL_NO_CUDA", "0") == "1"
    or not _HAS_TORCH
    or not _HAS_CUDA
)

ext_modules = []
cmdclass    = {}

if not _SKIP_CUDA:
    # CUDA 12.4 minimum: require SM 8.0+ (Ampere/Ada/Hopper/Blackwell)
    _COMPUTE_CAPABILITIES = [
        "-gencode", "arch=compute_80,code=sm_80",   # A100, A30, RTX 30
        "-gencode", "arch=compute_86,code=sm_86",   # RTX 3090 / A40
        "-gencode", "arch=compute_89,code=sm_89",   # RTX 40 / Ada
        "-gencode", "arch=compute_90,code=sm_90",   # H100 / H200
        "-gencode", "arch=compute_100,code=sm_100", # RTX PRO 6000 Blackwell
    ]

    ipc_extension = CUDAExtension(
        name="aerorl_ipc_ext",
        sources=[str(CSRC_DIR / "ipc_kv_cache.cu")],
        extra_compile_args={
            "cxx":  ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3",
                "--use_fast_math",
                "-std=c++17",
                "--ptxas-options=-v",
                # CUDA driver API for cuMemGetAddressRange
                "-lcuda",
            ] + _COMPUTE_CAPABILITIES,
        },
        libraries=["cuda"],  # link against CUDA driver library
    )

    ext_modules.append(ipc_extension)
    cmdclass["build_ext"] = BuildExtension

    print("[aerorl setup.py] Will build aerorl_ipc_ext (CUDA IPC extension).")
else:
    if _SKIP_CUDA:
        print("[aerorl setup.py] Skipping CUDA extension "
              "(AERORL_NO_CUDA=1 or no CUDA/PyTorch detected).")

# ── Read long description ─────────────────────────────────────────────────
_readme_path = HERE / "README.md"
long_description = _readme_path.read_text(encoding="utf-8") if _readme_path.exists() else ""

# ── setup() ──────────────────────────────────────────────────────────────
setup(
    name="aerorl",
    version="0.1.0",
    description=(
        "Zero-copy VLM RL library: drop-in KV cache + logits deduplication "
        "between vLLM rollout and PyTorch training."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AeroRL contributors",
    url="https://github.com/sabdulmajid/Active-Perception-R1",
    license="Apache-2.0",
    packages=find_packages(
        include=["aerorl", "aerorl.*"],
        exclude=["tests*", "benchmarks*"],
    ),
    python_requires=">=3.10",
    install_requires=[
        # Core tensor library — user must supply correct CUDA build
        "torch>=2.3.0",
        # Triton ships with recent PyTorch nightly / stable builds
        # but can also be installed separately:
        "triton>=2.3.0",
    ],
    extras_require={
        "quant": [
            "bitsandbytes>=0.43.0",
        ],
        "quant-torchao": [
            "torchao>=0.4.0",
        ],
        "transformers": [
            "transformers>=4.40.0",
        ],
        "vllm": [
            "vllm>=0.4.3",
        ],
        "all": [
            "bitsandbytes>=0.43.0",
            "transformers>=4.40.0",
            "vllm>=0.4.3",
            "accelerate>=0.30.0",
        ],
        "dev": [
            "pytest>=8.0",
            "pytest-cov",
        ],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    # Include the CUDA source in the wheel for reference
    package_data={
        "aerorl": [
            "extensions/csrc/*.cu",
            "extensions/csrc/*.h",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)
