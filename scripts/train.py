import argparse
import os

from everdream.config.load import load_config
from everdream.common import autodetect_device_type, compute_cleanup, compute_init, print_banner
from everdream.runtime.notebook import init_notebook
from everdream.train.trainer import train


def main():
    parser = argparse.ArgumentParser(description='Train everdream models')
    parser.add_argument('--config', required=True, help='Path to YAML/TOML config')
    args = parser.parse_args()

    print_banner()
    cfg = load_config(args.config)
    if cfg.runtime.hf_token:
        os.environ["HF_TOKEN"] = cfg.runtime.hf_token
    if cfg.runtime.wandb_api_key:
        os.environ["WANDB_API_KEY"] = cfg.runtime.wandb_api_key
    if cfg.runtime.notebook:
        init_notebook(
            mount_drive=cfg.runtime.mount_drive,
            drive_path=cfg.runtime.drive_path,
            install_gpu_extras=cfg.runtime.install_gpu_extras,
            install_moe=cfg.runtime.install_moe,
            install_hybrid=cfg.runtime.install_hybrid,
        )
    device_type = autodetect_device_type() if cfg.runtime.device_type == '' else cfg.runtime.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0
    try:
        train(cfg, device=device, master_process=master_process)
    finally:
        compute_cleanup()


if __name__ == '__main__':
    main()
