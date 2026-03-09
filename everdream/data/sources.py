from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from multiprocessing import Pool

import pyarrow.parquet as pq
import requests
from datasets import load_dataset
from huggingface_hub import hf_hub_download

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
    if spec.hf_data_dir:
        ds = load_dataset(spec.source, data_dir=spec.hf_data_dir, split=spec.split, streaming=True)
        batch = []
        for idx, row in enumerate(ds):
            if idx < start or ((idx - start) % step) != 0:
                continue
            field = spec.text_field
            if field not in row:
                cols = ", ".join(sorted(row.keys()))
                raise ValueError(
                    f"Dataset {spec.name} does not contain the configured text field '{field}'. "
                    f"Available fields: {cols}."
                )
            batch.append(row[field])
            if len(batch) >= 1024:
                yield batch
                batch = []
        if batch:
            yield batch
        return

    parquet_paths = list_parquet_files(spec)
    if split == "val":
        parquet_paths = parquet_paths[-1:]
    else:
        parquet_paths = parquet_paths[:-1] or parquet_paths
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            field = spec.text_field
            if field not in rg.schema.names:
                cols = ", ".join(rg.schema.names)
                if spec.name.startswith("stack_edu"):
                    raise ValueError(
                        f"Dataset {spec.name} does not contain the configured text field '{field}'. "
                        f"Available columns: {cols}. "
                        f"Stack-Edu parquet on Hugging Face contains metadata/SWHIDs, not file contents. "
                        f"It cannot be used directly as a text corpus in everdream without a separate content-fetch step."
                    )
                raise ValueError(
                    f"Dataset {spec.name} does not contain the configured text field '{field}'. "
                    f"Available columns: {cols}."
                )
            yield rg.column(field).to_pylist()


def _download_one(args: tuple[str, str, str, str]) -> bool:
    name, url, filepath, hf_token = args
    if os.path.exists(filepath):
        return True
    tmp = filepath + ".tmp"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            if "huggingface.co/datasets/" in url:
                prefix = "https://huggingface.co/datasets/"
                repo_and_path = url.split(prefix, 1)[1]
                repo_id, _, file_path = repo_and_path.partition("/resolve/main/")
                if not repo_id or not file_path:
                    raise ValueError(f"Unsupported Hugging Face dataset URL: {url}")
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    repo_type="dataset",
                    token=hf_token or None,
                    local_dir=os.path.dirname(filepath),
                    local_dir_use_symlinks=False,
                    force_download=False,
                )
                if downloaded != filepath and os.path.abspath(downloaded) != os.path.abspath(filepath):
                    os.replace(downloaded, filepath)
            else:
                response = requests.get(url, stream=True, timeout=60)
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
    if spec.hf_data_dir:
        return
    target_dir = resolve_dataset_dir(spec)
    target_dir.mkdir(parents=True, exist_ok=True)
    if not spec.source.startswith("http"):
        if not any(target_dir.glob(spec.shard_glob)):
            raise FileNotFoundError(f"No parquet shards found for dataset {spec.name} in {target_dir}")
        return
    filenames = build_prefetch_filenames(spec)
    missing = [name for name in filenames if not (target_dir / name).exists()]
    if not missing:
        return
    print0(
        f"Prefetching {len(missing)}/{len(filenames)} missing shards for dataset {spec.name} into {target_dir}"
    )
    download_dataset(spec, missing, workers=spec.num_download_workers)
    remaining_missing = [name for name in filenames if not (target_dir / name).exists()]
    if remaining_missing:
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
