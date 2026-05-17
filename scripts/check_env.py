# GhostKV Lab — github.com/manishklach/ghostkv-lab
# Patent: IN 202641062451
"""Detect execution environment and recommend a modern validation model."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - import availability depends on local environment
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / ".ghostkv_env_config.json"


def detect_linux_environment() -> str:
    """Return a coarse environment label based on /proc/version."""
    version_path = Path("/proc/version")
    if not version_path.exists():
        return "non_linux"
    version_text = version_path.read_text(encoding="utf-8", errors="ignore").lower()
    if "microsoft" in version_text:
        return "wsl2"
    return "linux"


def query_nvidia_smi() -> dict[str, str | float | None]:
    """Query GPU name and VRAM via nvidia-smi when available."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"gpu_name": None, "vram_gb": None}

    first_line = result.stdout.strip().splitlines()[0]
    gpu_name, memory_mib = [part.strip() for part in first_line.split(",", maxsplit=1)]
    return {
        "gpu_name": gpu_name,
        "vram_gb": round(float(memory_mib) / 1024.0, 2),
    }


def query_torch_cuda() -> dict[str, str | bool | None]:
    """Report torch CUDA availability and version information."""
    if torch is None:
        return {
            "torch_cuda_available": False,
            "torch_cuda_version": None,
            "torch_device_name": None,
            "torch_imported": False,
        }

    cuda_available = torch.cuda.is_available()
    return {
        "torch_cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "torch_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "torch_imported": True,
    }


def recommend_model(vram_gb: float | None, cuda_available: bool) -> dict[str, str | bool]:
    """Recommend a model and loading mode based on available accelerator memory."""
    if cuda_available and vram_gb is not None and vram_gb >= 16.0:
        return {
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "precision": "fp16",
            "device": "cuda",
            "load_in_4bit": False,
            "recommendation": "Use Llama-3.1-8B-Instruct in FP16.",
        }
    if cuda_available and vram_gb is not None and 8.0 <= vram_gb < 16.0:
        return {
            "model_id": "meta-llama/Llama-3.2-3B-Instruct",
            "precision": "fp16",
            "device": "cuda",
            "load_in_4bit": False,
            "recommendation": "Use Llama-3.2-3B-Instruct in FP16.",
        }
    if cuda_available and vram_gb is not None and 4.0 <= vram_gb < 8.0:
        return {
            "model_id": "meta-llama/Llama-3.2-3B-Instruct",
            "precision": "4bit",
            "device": "cuda",
            "load_in_4bit": True,
            "recommendation": "Use Llama-3.2-3B-Instruct in 4-bit mode if bitsandbytes is available.",
        }
    return {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "precision": "fp16",
        "device": "cpu",
        "load_in_4bit": False,
        "recommendation": "Use Llama-3.2-1B-Instruct on CPU.",
    }


def main() -> None:
    """Detect the local environment and persist a model recommendation."""
    environment = detect_linux_environment()
    gpu_info = query_nvidia_smi()
    torch_info = query_torch_cuda()
    recommendation = recommend_model(
        vram_gb=gpu_info["vram_gb"],
        cuda_available=bool(torch_info["torch_cuda_available"]),
    )

    payload = {
        "environment": environment,
        **gpu_info,
        **torch_info,
        **recommendation,
    }
    CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("GhostKV environment check")
    print(f"environment: {environment}")
    print(f"gpu_name: {gpu_info['gpu_name']}")
    print(f"vram_gb: {gpu_info['vram_gb']}")
    print(f"torch_cuda_available: {torch_info['torch_cuda_available']}")
    print(f"torch_cuda_version: {torch_info['torch_cuda_version']}")
    print(f"torch_imported: {torch_info['torch_imported']}")
    print(f"recommended_model_id: {recommendation['model_id']}")
    print(f"recommended_precision: {recommendation['precision']}")
    print(f"recommended_device: {recommendation['device']}")
    print(f"load_in_4bit: {recommendation['load_in_4bit']}")
    print(recommendation["recommendation"])
    print(f"wrote_config: {CONFIG_PATH}")


if __name__ == "__main__":
    main()
