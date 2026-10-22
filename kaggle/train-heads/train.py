"""Train pointer, number, and null heads on the notes Gemma already wrote.

The notes are the attached dataset. This run does not generate text.
The number head looks at the document tokens itself. The pointer score
ignores one leading or trailing space.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


OUT = Path("/kaggle/working")
STATUS = OUT / "status.json"
METRICS = OUT / "metrics.json"
HEADS = OUT / "heads.pt"
EPOCHS = 30
LR = 1e-3
SEED = 22
MAX_LEN = 256
SUFFIX = "\nQuestion: order id\nQuestion: price plus tax\nQuestion: tracking number"


def status(payload):
    payload["t"] = round(time.time(), 1)
    print(json.dumps(payload), flush=True)
    STATUS.write_text(json.dumps(payload, indent=2) + "\n")


def load_rows():
    found = list(Path("/kaggle/input").rglob("accepted.jsonl"))
    if len(found) != 1:
        raise RuntimeError(f"expected one accepted.jsonl, found {found}")
    rows = [json.loads(line) for line in found[0].read_text().splitlines() if line.strip()]
    status({"event": "rows", "n": len(rows), "path": str(found[0])})
    return rows


def marker_token(offsets, text: str, name: str):
    pos = text.rfind(name)
    if pos < 0:
        return None
    last_char = pos + len(name) - 1
    chosen = None
    for index, (start, end) in enumerate(offsets):
        if start <= last_char < end:
            chosen = index
    return chosen


def encode_example(tokenizer, row: dict):
    state = row["state"]
    full = state + SUFFIX
    encoded = tokenizer(
        full,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LEN,
    )
    offsets = encoded["offset_mapping"]
    doc = [index for index, (start, end) in enumerate(offsets) if end <= len(state) and end > start]
    if not doc:
        return None
    span_start, span_end = row["span"]
    covered = [
        index
        for index in doc
        if not (offsets[index][1] <= span_start or offsets[index][0] >= span_end)
    ]
    if not covered:
        return None
    markers = {}
    for name, key in (
        ("order id", "order"),
        ("price plus tax", "total"),
        ("tracking number", "track"),
    ):
        at = marker_token(offsets, full, name)
        if at is None:
            return None
        markers[key] = at
    return {
        "ids": encoded["input_ids"],
        "doc": doc,
        "start": covered[0],
        "end": covered[-1],
        "markers": markers,
        "offsets": offsets,
    }


def hidden_states(model, tokenizer, rows):
    cache = []
    for row in rows:
        spec = encode_example(tokenizer, row)
        if spec is None:
            continue
        ids = torch.tensor([spec["ids"]], device=next(model.parameters()).device)
        with torch.inference_mode():
            out = model(input_ids=ids, output_hidden_states=True)
            hidden = out.hidden_states[-1][0].float().cpu()
        doc = spec["doc"]
        cache.append({
            "doc_h": hidden[doc].half(),
            "q_order": hidden[spec["markers"]["order"]].half(),
            "q_total": hidden[spec["markers"]["total"]].half(),
            "q_track": hidden[spec["markers"]["track"]].half(),
            "start": doc.index(spec["start"]),
            "end": doc.index(spec["end"]),
            "total": row["total"],
            "p_null": row["p_null"],
            "order_id": row["order_id"],
            "state": row["state"],
            "offsets": [spec["offsets"][index] for index in doc],
        })
    status({"event": "cached", "n": len(cache)})
    return cache


class Heads(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.start_q = nn.Linear(width, width, bias=False)
        self.end_q = nn.Linear(width, width, bias=False)
        self.num_q = nn.Linear(width, width, bias=False)
        self.num_out = nn.Linear(width, 1)
        self.null = nn.Linear(width, 1)

    def pointer(self, doc_h, query):
        return doc_h @ self.start_q(query), doc_h @ self.end_q(query)

    def number(self, doc_h, query):
        weights = torch.softmax(doc_h @ self.num_q(query), dim=0)
        pooled = weights @ doc_h
        return self.num_out(pooled).squeeze(0)


def train_heads(cache):
    width = cache[0]["q_order"].shape[0]
    heads = Heads(width)
    opt = torch.optim.AdamW(heads.parameters(), lr=LR)
    rng = random.Random(SEED)
    for epoch in range(EPOCHS):
        rng.shuffle(cache)
        total_loss = 0.0
        seen = 0
        for start in range(0, len(cache), 16):
            batch = cache[start : start + 16]
            opt.zero_grad()
            loss = 0.0
            for item in batch:
                doc = item["doc_h"].float()
                start_logits, end_logits = heads.pointer(doc, item["q_order"].float())
                loss = loss + F.cross_entropy(start_logits.unsqueeze(0), torch.tensor([item["start"]]))
                loss = loss + F.cross_entropy(end_logits.unsqueeze(0), torch.tensor([item["end"]]))
                pred = heads.number(doc, item["q_total"].float())
                loss = loss + F.smooth_l1_loss(pred, torch.tensor(item["total"] / 1000.0))
                null_logit = heads.null(item["q_track"].float()).squeeze(0)
                loss = loss + F.binary_cross_entropy_with_logits(null_logit, torch.tensor(item["p_null"]))
            loss = loss / len(batch)
            loss.backward()
            opt.step()
            total_loss += float(loss) * len(batch)
            seen += len(batch)
        status({"event": "epoch", "epoch": epoch, "loss": round(total_loss / max(seen, 1), 4)})
    return heads


def slice_ok(got: str, order_id: str) -> bool:
    return got.strip() == order_id


def evaluate(heads, cache):
    exact = 0
    null_hit = 0
    abs_err = 0.0
    close = 0
    samples = []
    with torch.no_grad():
        for item in cache:
            doc = item["doc_h"].float()
            start_logits, end_logits = heads.pointer(doc, item["q_order"].float())
            start = int(start_logits.argmax())
            end = int(end_logits.argmax())
            if end < start:
                end = start
            got = item["state"][item["offsets"][start][0] : item["offsets"][end][1]]
            exact += int(slice_ok(got, item["order_id"]))
            pred_total = float(heads.number(doc, item["q_total"].float()) * 1000.0)
            abs_err += abs(pred_total - item["total"])
            close += int(abs(pred_total - item["total"]) <= 1.0)
            p_null = float(torch.sigmoid(heads.null(item["q_track"].float()).squeeze()))
            null_hit += int((p_null >= 0.5) == (item["p_null"] >= 0.5))
            if len(samples) < 6:
                samples.append({
                    "order_id": item["order_id"],
                    "predicted_slice": got,
                    "total": item["total"],
                    "predicted_total": round(pred_total, 2),
                    "p_null_true": item["p_null"],
                    "p_null_pred": round(p_null, 3),
                })
    n = max(len(cache), 1)
    return {
        "n": len(cache),
        "pointer_exact": round(exact / n, 4),
        "total_mae": round(abs_err / n, 2),
        "total_within_1": round(close / n, 4),
        "null_accuracy": round(null_hit / n, 4),
        "samples": samples,
    }


def main():
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"])
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU in this Kaggle session.")
    from transformers import AutoModelForCausalLM, AutoProcessor

    status({"event": "cuda", "device": torch.cuda.get_device_name(0)})
    rows = load_rows()
    rng = random.Random(SEED)
    rng.shuffle(rows)
    split = max(1, int(len(rows) * 0.2))
    test_rows, train_rows = rows[:split], rows[split:]
    model_id = "google/gemma-4-E2B-it"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="cuda")
    model.eval()
    status({"event": "model_loaded", "model_id": model_id})
    cache_train = hidden_states(model, processor.tokenizer, train_rows)
    cache_test = hidden_states(model, processor.tokenizer, test_rows)
    del model
    torch.cuda.empty_cache()
    heads = train_heads(cache_train)
    report = {"train": evaluate(heads, cache_train), "test": evaluate(heads, cache_test)}
    METRICS.write_text(json.dumps(report, indent=2) + "\n")
    torch.save(heads.state_dict(), HEADS)
    status({"event": "done", "test": report["test"]})


if __name__ == "__main__":
    main()
