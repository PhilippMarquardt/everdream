from __future__ import annotations

import os
import subprocess
import sys

import torch


def init_notebook(
    mount_drive: bool = False,
    drive_path: str = "/content/drive",
    install_gpu_extras: bool = True,
    install_moe: bool = False,
    install_hybrid: bool = False,
):
    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        drive = None

    if mount_drive and drive is not None:
        drive.mount(drive_path)

    pkgs = [
        "transformers",
        "datasets",
        "wandb",
        "pyarrow",
        "requests",
        "tokenizers",
        "tiktoken",
        "Jinja2",
        "filelock",
        "PyYAML",
    ]
    if install_gpu_extras:
        pkgs.append("kernels>=0.11.7")
    if install_moe:
        pkgs.extend(["git+https://github.com/tgale96/grouped_gemm@main", "megablocks"])
    if install_hybrid:
        pkgs.append("flash-linear-attention")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    if mount_drive:
        print(f"Drive mounted at {drive_path}")
