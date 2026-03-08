from __future__ import annotations

import os
import subprocess
import sys

import torch


def _drive_is_mounted(drive_path: str) -> bool:
    return os.path.isdir(drive_path) and os.path.isdir(os.path.join(drive_path, "MyDrive"))


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

    if mount_drive:
        if _drive_is_mounted(drive_path):
            pass
        elif drive is not None:
            try:
                drive.mount(drive_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Google Drive is not mounted and cannot be mounted from this Python process. "
                    f"Mount Drive in a notebook cell first, then rerun the script, or set runtime.mount_drive=false. "
                    f"Target mount path: {drive_path}"
                ) from exc
        else:
            raise RuntimeError(
                f"Google Drive mount requested but google.colab is unavailable in this process. "
                f"Mount Drive manually first or set runtime.mount_drive=false. "
                f"Target mount path: {drive_path}"
            )

    pkgs = [
        "transformers",
        "datasets",
        "wandb",
        "pyarrow",
        "requests",
        "rustbpe>=0.1.0",
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
