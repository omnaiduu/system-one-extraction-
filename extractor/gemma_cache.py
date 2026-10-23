"""Cache Gemma 4 E2B last-layer states without loading the 10GB checkpoint.

The text tower is about 2.5GB. The token and per-layer embedding tables are
much larger, and this machine cannot hold them. Each document is tokenized
first. Only the rows for tokens that actually appear are copied into a small
embedding table. Layer weights are copied from the safetensors file one tensor
at a time.

The document is the causal prefix, so one forward covers every question.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, Gemma4TextModel

from extractor.labels import (
    char_span_to_tokens,
    document_token_indexes,
    expand_document,
    pack_prompt,
    question_token_indexes,
)

SNAPSHOT = Path(
    "/home/ubuntu/.cache/huggingface/hub/models--google--gemma-4-E2B-it/"
    "snapshots/3e22461f65e89153144f8adb70e3b8c2cc9845a7"
)
WEIGHTS = SNAPSHOT / "model.safetensors"
PREFIX = "model.language_model."
CACHE = Path("outputs/cache/docs")

SOURCES = (
    ("train", Path("data/audited/train.jsonl")),
    ("val", Path("data/audited/val.jsonl")),
    ("test", Path("data/audited/test.jsonl")),
    ("generalization", Path("data/generalization/locked.jsonl")),
)


def source_paths() -> list[tuple[str, Path]]:
    paths = list(SOURCES)
    extra = Path("data/round2/audited.jsonl")
    if extra.exists():
        paths.append(("train", extra))
    exam = Path("data/exam2/locked.jsonl")
    if exam.exists():
        paths.append(("exam2", exam))
    exam3 = Path("data/exam3/locked.jsonl")
    if exam3.exists():
        paths.append(("exam3", exam3))
    round3 = Path("data/round3/audited.jsonl")
    if round3.exists():
        paths.append(("train", round3))
    exam4 = Path("data/exam4/locked.jsonl")
    if exam4.exists():
        paths.append(("exam4", exam4))
    return paths


def load_rows() -> list[dict]:
    rows = []
    seen = set()
    for split, path in source_paths():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in seen:
                raise RuntimeError(f"duplicate document id {row['id']}")
            seen.add(row["id"])
            row["_split"] = split
            rows.append(row)
    return rows


def encode_rows(rows: list[dict]) -> tuple[list[dict], list[int]]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT)
    encoded = []
    used: set[int] = set()
    for row in rows:
        questions = expand_document(row)
        prompt, spans = pack_prompt(row["text"], questions)
        if not prompt.startswith(row["text"]):
            raise RuntimeError(f"{row['id']} document is not the prompt prefix")
        token = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
        offsets = [(int(start), int(end)) for start, end in token["offset_mapping"]]
        ids = [int(item) for item in token["input_ids"]]
        allowed = document_token_indexes(offsets, len(row["text"]))
        packed = []
        for question, (start, end) in zip(questions, spans):
            positions = question_token_indexes(offsets, start, end)
            if not positions:
                raise RuntimeError(f"{row['id']} {question['field']} has no question tokens")
            item = {
                "field": question["field"],
                "head": question["head"],
                "kind": question["kind"],
                "q_index": positions[-1],
            }
            if question["head"] == "point":
                item["missing"] = bool(question["missing"])
                item["value"] = question["value"]
                item["gold_start"] = question["start"]
                item["gold_end"] = question["end"]
                if question["missing"]:
                    item["start"] = 0
                    item["end"] = 0
                else:
                    start_at, end_at = char_span_to_tokens(offsets, question["start"], question["end"], allowed)
                    token_start = offsets[allowed[start_at]][0]
                    token_end = offsets[allowed[end_at]][1]
                    covered = row["text"][token_start:token_end]
                    if covered.strip() != question["value"]:
                        raise RuntimeError(
                            f"{row['id']} {question['field']} token slice {covered!r} != {question['value']!r}"
                        )
                    item["start"] = start_at
                    item["end"] = end_at
                    item["token_start"] = token_start
                    item["token_end"] = token_end
            elif question["head"] == "number":
                item["target"] = question["target"]
            elif question["head"] == "calendar":
                item["target"] = question["target"]
            elif question["head"] == "spread":
                item["value"] = question["value"]
                item["norm"] = question["norm"]
                item["min"] = question["min"]
                item["max"] = question["max"]
                item["step"] = question["step"]
            else:
                raise RuntimeError(question["head"])
            packed.append(item)
        encoded.append(
            {
                "id": row["id"],
                "split": row["_split"],
                "text": row["text"],
                "input_ids": ids,
                "doc_positions": allowed,
                "doc_offsets": [offsets[index] for index in allowed],
                "questions": packed,
            }
        )
        used.update(ids)
    return encoded, sorted(used)


def build_model(vocab: int) -> Gemma4TextModel:
    torch.set_default_dtype(torch.bfloat16)
    config = AutoConfig.from_pretrained(SNAPSHOT)
    text = config.text_config
    text.vocab_size = vocab
    text.vocab_size_per_layer_input = vocab
    text.pad_token_id = None
    text._attn_implementation = "eager"
    model = Gemma4TextModel(text)
    model.eval()
    model.requires_grad_(False)
    return model


def _copy_tensor(destination, source) -> None:
    if source.shape != destination.shape:
        if source.numel() == destination.numel():
            source = source.reshape(destination.shape)
        else:
            raise RuntimeError(f"shape {tuple(source.shape)} != {tuple(destination.shape)}")
    destination.copy_(source.to(dtype=destination.dtype))


def load_weights(model: Gemma4TextModel, token_ids: list[int]) -> None:
    with safe_open(WEIGHTS, framework="pt") as handle:
        keys = set(handle.keys())
        for name, param in model.named_parameters():
            if name in {"embed_tokens.weight", "embed_tokens_per_layer.weight"}:
                continue
            source = PREFIX + name
            if source not in keys:
                raise RuntimeError(f"missing checkpoint tensor {source}")
            _copy_tensor(param, handle.get_tensor(source))
        for name, buf in model.named_buffers():
            source = PREFIX + name
            if source in keys:
                _copy_tensor(buf, handle.get_tensor(source))
        token_table = handle.get_slice(PREFIX + "embed_tokens.weight")
        ple_table = handle.get_slice(PREFIX + "embed_tokens_per_layer.weight")
        for row, token_id in enumerate(token_ids):
            _copy_tensor(model.embed_tokens.weight[row], token_table[token_id])
            _copy_tensor(model.embed_tokens_per_layer.weight[row], ple_table[token_id])


def lookup_table(token_ids: list[int]) -> torch.Tensor:
    table = torch.full((262144,), -1, dtype=torch.long)
    for row, token_id in enumerate(token_ids):
        table[token_id] = row
    return table


def forward_document(model: Gemma4TextModel, table: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    mapped = table[torch.tensor(token_ids, dtype=torch.long)]
    if (mapped < 0).any():
        raise RuntimeError("a token was not copied into the tiny vocabulary")
    with torch.inference_mode():
        hidden = model(input_ids=mapped.view(1, -1), use_cache=False).last_hidden_state[0]
    return hidden.to(dtype=torch.float16).cpu()


def cache_path(split: str, doc_id: str) -> Path:
    safe = doc_id.replace("/", "_")
    return CACHE / f"{split}__{safe}.pt"


def write_cache(encoded: list[dict], limit: int) -> None:
    chosen = encoded[:limit] if limit else encoded
    pending = [item for item in chosen if not cache_path(item["split"], item["id"]).exists()]
    if not pending:
        print(f"cache already has {len(chosen)} documents", flush=True)
        return
    token_ids = sorted({token for item in encoded for token in item["input_ids"]})
    print(f"building text tower for {len(token_ids)} tokens, {len(pending)} forwards left", flush=True)
    model = build_model(len(token_ids))
    load_weights(model, token_ids)
    table = lookup_table(token_ids)
    CACHE.mkdir(parents=True, exist_ok=True)
    for item in pending:
        clock = time.perf_counter()
        hidden = forward_document(model, table, item["input_ids"])
        doc_states = hidden[item["doc_positions"]]
        questions = []
        for question in item["questions"]:
            saved = dict(question)
            saved["q_state"] = hidden[question["q_index"]]
            del saved["q_index"]
            questions.append(saved)
        record = {
            "id": item["id"],
            "split": item["split"],
            "text": item["text"],
            "doc_states": doc_states,
            "doc_offsets": item["doc_offsets"],
            "questions": questions,
        }
        destination = cache_path(item["split"], item["id"])
        torch.save(record, destination)
        elapsed = time.perf_counter() - clock
        print(f"cached {item['split']} {item['id']} tokens={len(item['input_ids'])} seconds={elapsed:.1f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="cache only the first N documents; 0 means all")
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)
    rows = load_rows()
    encoded, token_ids = encode_rows(rows)
    print(f"documents={len(encoded)} unique_tokens={len(token_ids)}", flush=True)
    write_cache(encoded, args.limit)


if __name__ == "__main__":
    main()
