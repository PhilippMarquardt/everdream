from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from multiprocessing import Pool

import pyarrow.parquet as pq
import requests

from everdream.config.schema import DatasetConfig
from everdream.runtime.distributed import get_base_dir, print0


def resolve_dataset_dir(spec: DatasetConfig) -> Path:
    if spec.local_dir:
        return Path(spec.local_dir)
    base = Path(get_base_dir()) / "datasets" / spec.name
    base.mkdir(parents=True, exist_ok=True)
    return base


def list_parquet_files(spec: DatasetConfig) -> list[Path]:
    data_dir = resolve_dataset_dir(spec)
    files = sorted(p for p in data_dir.glob(spec.shard_glob) if p.suffix == ".parquet")
    if spec.max_shards is not None:
        files = files[:spec.max_shards]
    if not files:
        raise FileNotFoundError(f"No parquet shards found for dataset {spec.name} in {data_dir}")
    return files


def build_prefetch_filenames(spec: DatasetConfig) -> list[str]:
    if spec.max_shard_index is None and spec.num_train_shards != 0:
        raise ValueError(f"Dataset {spec.name} requires max_shard_index to prefetch remote shards")
    if spec.val_shard_index is not None:
        val_idx = spec.val_shard_index
    else:
        val_idx = spec.max_shard_index
    assert val_idx is not None
    if spec.num_train_shards == -1:
        assert spec.max_shard_index is not None
        train_count = spec.max_shard_index
    else:
        train_count = spec.num_train_shards
    filenames = [spec.filename_template.format(index=i) for i in range(train_count)]
    filenames.append(spec.filename_template.format(index=val_idx))
    return sorted(set(filenames))


def document_batches(spec: DatasetConfig, split: str, start: int = 0, step: int = 1):
    parquet_paths = list_parquet_files(spec)
    if split == "val":
        parquet_paths = parquet_paths[-1:]
    else:
        parquet_paths = parquet_paths[:-1] or parquet_paths
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            yield rg.column("text").to_pylist()


def _download_one(args: tuple[str, str, str, str]) -> bool:
    name, url, filepath, hf_token = args
    if os.path.exists(filepath):
        return True
    tmp = filepath + ".tmp"
    headers = {}
    if hf_token and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {hf_token}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=60, headers=headers)
            response.raise_for_status()
            with open(tmp, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            os.replace(tmp, filepath)
            return True
        except Exception as exc:
            for candidate in (tmp, filepath):
                if os.path.exists(candidate):
                    try:
                        os.remove(candidate)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
            else:
                print0(f"Failed to download {name}: {url} ({type(exc).__name__}: {exc})")
    return False


def download_dataset(spec: DatasetConfig, filenames: list[str], workers: int = 4) -> None:
    if not spec.source.startswith("http"):
        raise ValueError(f"Dataset {spec.name} does not use a remote source URL")
    target_dir = resolve_dataset_dir(spec)
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    tasks = [(spec.name, f"{spec.source.rstrip('/')}/{name}", str(target_dir / name), hf_token) for name in filenames]
    with Pool(processes=workers) as pool:
        pool.map(_download_one, tasks)


def ensure_dataset_ready(spec: DatasetConfig) -> None:
    target_dir = resolve_dataset_dir(spec)
    target_dir.mkdir(parents=True, exist_ok=True)
    if any(target_dir.glob(spec.shard_glob)):
        return
    if not spec.source.startswith("http"):
        raise FileNotFoundError(f"No parquet shards found for dataset {spec.name} in {target_dir}")
    filenames = build_prefetch_filenames(spec)
    print0(f"Prefetching {len(filenames)} shards for dataset {spec.name} into {target_dir}")
    download_dataset(spec, filenames, workers=spec.num_download_workers)
    if not any(target_dir.glob(spec.shard_glob)):
        raise FileNotFoundError(f"Dataset prefetch completed but no parquet shards found for dataset {spec.name} in {target_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download parquet shards for an everdream dataset")
    parser.add_argument("--name", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--filenames", nargs="+", required=True)
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    spec = DatasetConfig(name=args.name, source=args.source, local_dir=args.local_dir)
    download_dataset(spec, args.filenames, workers=args.workers)
