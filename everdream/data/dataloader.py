from __future__ import annotations

from dataclasses import dataclass
import random

import torch

from everdream.config.schema import DatasetConfig
from everdream.data.sources import document_batches
from everdream.runtime.distributed import get_dist_info


@dataclass
class DataState:
    epoch: int = 1
    draws: int = 0


def _weighted_document_batches(
    dataset_specs: list[DatasetConfig],
    split: str,
    tokenizer_batch_size: int,
    seed: int,
):
    ddp, ddp_rank, _, ddp_world_size = get_dist_info()
    rng = random.Random(seed + (ddp_rank if ddp else 0))
    iterators = [
        document_batches(spec, split=split, start=ddp_rank if ddp else 0, step=ddp_world_size if ddp else 1)
        for spec in dataset_specs
    ]
    pending_docs = [[] for _ in dataset_specs]
    weights = [spec.weight for spec in dataset_specs]
    state = DataState()

    while True:
        idx = rng.choices(range(len(dataset_specs)), weights=weights, k=1)[0]
        spec = dataset_specs[idx]
        docs = pending_docs[idx]
        while len(docs) < tokenizer_batch_size:
            try:
                docs.extend(next(iterators[idx]))
            except StopIteration:
                iterators[idx] = document_batches(spec, split=split, start=ddp_rank if ddp else 0, step=ddp_world_size if ddp else 1)
                docs.extend(next(iterators[idx]))
                state.epoch += 1
        state.draws += 1
        batch_docs = docs[:tokenizer_batch_size]
        pending_docs[idx] = docs[tokenizer_batch_size:]
        yield spec.name, batch_docs, {"epoch": state.epoch, "draws": state.draws}


def tokenizing_weighted_data_loader_bos_bestfit(
    tokenizer,
    dataset_specs: list[DatasetConfig],
    B: int,
    T: int,
    split: str,
    seed: int = 42,
    tokenizer_threads: int = 4,
    tokenizer_batch_size: int = 128,
    device: str = "cuda",
    buffer_size: int = 1000,
):
    row_capacity = T + 1
    bos_token = tokenizer.get_bos_token_id()
    ddp, ddp_rank, _, ddp_world_size = get_dist_info()
    rng = random.Random(seed + (ddp_rank if ddp else 0))
    iterators = [
        document_batches(spec, split=split, start=ddp_rank if ddp else 0, step=ddp_world_size if ddp else 1)
        for spec in dataset_specs
    ]
    pending_docs = [[] for _ in dataset_specs]
    dataset_weights = [float(spec.weight) for spec in dataset_specs]
    source_epochs = [1 for _ in dataset_specs]
    source_buffers: list[list[list[int]]] = [[] for _ in dataset_specs]
    source_row_queues: list[list[list[int]]] = [[] for _ in dataset_specs]
    use_cuda = device == "cuda"
    row_queue_target = max(8, min(128, buffer_size // 8))

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    last_state = {"epoch": 1, "draws": 0, "source": None}

    def next_doc_batch(dataset_idx: int):
        nonlocal last_state
        docs = pending_docs[dataset_idx]
        spec = dataset_specs[dataset_idx]
        while len(docs) < tokenizer_batch_size:
            try:
                docs.extend(next(iterators[dataset_idx]))
            except StopIteration:
                iterators[dataset_idx] = document_batches(spec, split=split, start=ddp_rank if ddp else 0, step=ddp_world_size if ddp else 1)
                docs.extend(next(iterators[dataset_idx]))
                source_epochs[dataset_idx] += 1
                last_state["epoch"] = max(source_epochs)
        pending_docs[dataset_idx] = docs[tokenizer_batch_size:]
        last_state["draws"] += 1
        return docs[:tokenizer_batch_size]

    def refill_source(dataset_idx: int):
        doc_batch = next_doc_batch(dataset_idx)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        source_buffers[dataset_idx].extend(token_lists)

    def pack_row_for_source(dataset_idx: int):
        row = [0] * row_capacity
        pos = 0
        docs = source_buffers[dataset_idx]
        while pos < row_capacity:
            while len(docs) < buffer_size:
                refill_source(dataset_idx)
                docs = source_buffers[dataset_idx]
            remaining = row_capacity - pos
            best_idx = -1
            best_len = 0
            for i, doc in enumerate(docs):
                doc_len = len(doc)
                if doc_len <= remaining and doc_len > best_len:
                    best_idx = i
                    best_len = doc_len
            if best_idx >= 0:
                doc = docs.pop(best_idx)
                doc_len = len(doc)
                row[pos:pos + doc_len] = doc
                pos += doc_len
                continue
            shortest_idx = min(range(len(docs)), key=lambda i: len(docs[i]))
            doc = docs[shortest_idx]
            row[pos:pos + remaining] = doc[:remaining]
            pos += remaining
            remainder = doc[remaining:]
            if remainder:
                docs[shortest_idx] = remainder
            else:
                docs.pop(shortest_idx)
        return row

    def refill_row_queue(dataset_idx: int):
        queue = source_row_queues[dataset_idx]
        while len(queue) < row_queue_target:
            queue.append(pack_row_for_source(dataset_idx))
        rng.shuffle(queue)

    while True:
        for row_idx in range(B):
            dataset_idx = rng.choices(range(len(dataset_specs)), weights=dataset_weights, k=1)[0]
            if not source_row_queues[dataset_idx]:
                refill_row_queue(dataset_idx)
            row = source_row_queues[dataset_idx].pop()
            row_buffer[row_idx].copy_(torch.tensor(row, dtype=torch.long))
            last_state["source"] = dataset_specs[dataset_idx].name
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, last_state
