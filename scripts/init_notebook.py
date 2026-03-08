from everdream.runtime.notebook import init_notebook


if __name__ == "__main__":
    init_notebook(mount_drive=True, install_gpu_extras=True, install_moe=True)
