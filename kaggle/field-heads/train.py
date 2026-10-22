"""Kaggle GPU run: Gemma E2B writes notes, code labels them, heads train.

Gemma is frozen. Three heads train on its hidden states:
a pointer for the order id, a number head for price plus tax, a null head
for a tracking number that is sometimes absent.
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
ACCEPTED = OUT / "accepted.jsonl"
METRICS = OUT / "metrics.json"
HEADS = OUT / "heads.pt"

ATTEMPTS = 700
GEN_BUDGET_S = 50 * 60
MAX_NEW = 100
MAX_LEN = 256
EPOCHS = 12
LR = 1e-3
SEED = 22


def status(payload):
    payload["t"] = round(time.time(), 1)
    print(json.dumps(payload), flush=True)
    STATUS.write_text(json.dumps(payload, indent=2) + "\n")


def appears_once(text: str, token: str) -> bool:
    count = 0
    start = 0
    while True:
        index = text.find(token, start)
        if index < 0:
            break
        before = text[index - 1] if index else " "
        after = text[index + len(token)] if index + len(token) < len(text) else " "
        if not before.isdigit() and not after.isdigit() and before not in "-/" and after not in "-/":
            count += 1
        start = index + len(token)
    return count == 1


def slots(rng: random.Random) -> dict:
    kind = rng.choice(["A", "INV", "OD", "ZZ"])
    order_id = f"{kind}-{rng.randint(1000, 99999)}"
    price = rng.randint(80, 9000) + rng.choice([0.25, 0.5, 0.75])
    tax = round(price * rng.choice([0.05, 0.12, 0.18]), 2)
    tracking = None
    if rng.random() < 0.55:
        tracking = f"TRK{rng.randint(100000, 999999)}"
    return {
        "order_id": order_id,
        "price": f"{price:.2f}",
        "tax": f"{tax:.2f}",
        "total": round(price + tax, 2),
        "tracking": tracking,
    }


def prompt_for(row: dict) -> str:
    tracking_rule = (
        f"Include this tracking number exactly once: {row['tracking']}"
        if row["tracking"]
        else "Do not include a tracking number."
    )
    return (
        "Write a short customer note of two or three sentences.\n"
        f"Include this order id exactly once: {row['order_id']}\n"
        f"Include this price exactly once, digits only like this: {row['price']}\n"
        f"Include this tax exactly once, digits only like this: {row['tax']}\n"
        "Do not write the total.\n"
        f"{tracking_rule}\n"
        "Reply with the note only."
    )


def accept(text: str, row: dict) -> bool:
    if not text or len(text) < 40 or len(text) > 500:
        return False
    if "Field:" in text or "Q_ORDER" in text:
        return False
    if f"{row['total']:.2f}" in text:
        return False
    if not appears_once(text, row["order_id"]):
        return False
    if not appears_once(text, row["price"]) or not appears_once(text, row["tax"]):
        return False
    if row["tracking"] is None:
        return "TRK" not in text
    return appears_once(text, row["tracking"])


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    candidates = ["google/gemma-4-E2B-it", "google/gemma-4-E2B"]
    last = None
    for model_id in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="cuda",
            )
            model.eval()
            status({"event": "model_loaded", "model_id": model_id, "class": type(model).__name__})
            return model_id, tokenizer, model
        except Exception as exc:  # noqa: BLE001
            last = f"{model_id}: {type(exc).__name__}: {exc}"
            status({"event": "model_load_failed", "detail": last[:500]})
    raise RuntimeError(last or "no model loaded")


def generate_note(model, tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    except Exception:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
    ids = ids.to(model.device)
    with torch.inference_mode():
        out = model.generate(
            ids,
            max_new_tokens=MAX_NEW,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    new = out[0, ids.shape[-1] :]
    return tokenizer.decode(new, skip_special_tokens=True).strip()


def generate_rows(model, tokenizer):
    rng = random.Random(SEED)
    started = time.time()
    kept = []
    dropped = 0
    for attempt in range(ATTEMPTS):
        if time.time() - started > GEN_BUDGET_S:
            break
        row = slots(rng)
        try:
            text = generate_note(model, tokenizer, prompt_for(row))
        except Exception as exc:  # noqa: BLE001
            status({"event": "generate_error", "detail": str(exc)[:300]})
            break
        if accept(text, row):
            start = text.index(row["order_id"])
            kept.append({
                "state": text,
                "order_id": row["order_id"],
                "span": [start, start + len(row["order_id"])],
                "price": float(row["price"]),
                "tax": float(row["tax"]),
                "total": row["total"],
                "tracking": row["tracking"],
                "p_null": 1.0 if row["tracking"] is None else 0.0,
            })
        else:
            dropped += 1
        if attempt % 25 == 0:
            status({
                "event": "generating",
                "attempt": attempt,
                "kept": len(kept),
                "dropped": dropped,
            })
    status({"event": "generated", "kept": len(kept), "dropped": dropped})
    return kept


def marker_token(offsets, text: str, name: str) -> int | None:
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
    full = state + "\nQ_ORDER\nQ_TOTAL\nQ_TRACK"
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
    for name in ("Q_ORDER", "Q_TOTAL", "Q_TRACK"):
        at = marker_token(offsets, full, name)
        if at is None:
            return None
        markers[name] = at
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
        ids = torch.tensor([spec["ids"]], device=model.device)
        with torch.inference_mode():
            out = model(input_ids=ids, output_hidden_states=True)
            hidden = out.hidden_states[-1][0].float().cpu()
        doc = spec["doc"]
        cache.append({
            "doc_h": hidden[doc].half(),
            "doc_index": doc,
            "q_order": hidden[spec["markers"]["Q_ORDER"]].half(),
            "q_total": hidden[spec["markers"]["Q_TOTAL"]].half(),
            "q_track": hidden[spec["markers"]["Q_TRACK"]].half(),
            "start": doc.index(spec["start"]),
            "end": doc.index(spec["end"]),
            "total": row["total"],
            "p_null": row["p_null"],
            "order_id": row["order_id"],
            "state": row["state"],
            "offsets": [spec["offsets"][index] for index in doc],
        })
    status({"event": "cached", "n": len(cache), "hidden": int(cache[0]["q_order"].shape[0]) if cache else 0})
    return cache


class Heads(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.start_q = nn.Linear(width, width, bias=False)
        self.end_q = nn.Linear(width, width, bias=False)
        self.number = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
        self.null = nn.Linear(width, 1)

    def pointer(self, doc_h, query):
        start = doc_h @ self.start_q(query)
        end = doc_h @ self.end_q(query)
        return start, end


def batches(items, size):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def train_heads(cache):
    width = cache[0]["q_order"].shape[0]
    heads = Heads(width)
    opt = torch.optim.AdamW(heads.parameters(), lr=LR)
    rng = random.Random(SEED)
    for epoch in range(EPOCHS):
        rng.shuffle(cache)
        total_loss = 0.0
        seen = 0
        for batch in batches(cache, 16):
            opt.zero_grad()
            loss = 0.0
            for item in batch:
                doc = item["doc_h"].float()
                start_logits, end_logits = heads.pointer(doc, item["q_order"].float())
                loss = loss + F.cross_entropy(start_logits.unsqueeze(0), torch.tensor([item["start"]]))
                loss = loss + F.cross_entropy(end_logits.unsqueeze(0), torch.tensor([item["end"]]))
                pred = heads.number(item["q_total"].float()).squeeze(0)
                loss = loss + F.smooth_l1_loss(pred, torch.tensor(item["total"] / 1000.0))
                null_logit = heads.null(item["q_track"].float()).squeeze(0)
                loss = loss + F.binary_cross_entropy_with_logits(
                    null_logit, torch.tensor(item["p_null"])
                )
            loss = loss / len(batch)
            loss.backward()
            opt.step()
            total_loss += float(loss) * len(batch)
            seen += len(batch)
        status({"event": "epoch", "epoch": epoch, "loss": round(total_loss / max(seen, 1), 4)})
    return heads


def evaluate(heads, cache):
    exact = 0
    null_hit = 0
    abs_err = 0.0
    samples = []
    with torch.no_grad():
        for item in cache:
            doc = item["doc_h"].float()
            start_logits, end_logits = heads.pointer(doc, item["q_order"].float())
            start = int(start_logits.argmax())
            end = int(end_logits.argmax())
            if end < start:
                end = start
            char_start = item["offsets"][start][0]
            char_end = item["offsets"][end][1]
            got = item["state"][char_start:char_end]
            exact += int(got == item["order_id"])
            pred_total = float(heads.number(item["q_total"].float()).squeeze() * 1000.0)
            abs_err += abs(pred_total - item["total"])
            p_null = float(torch.sigmoid(heads.null(item["q_track"].float()).squeeze()))
            null_hit += int((p_null >= 0.5) == (item["p_null"] >= 0.5))
            if len(samples) < 8:
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
        "null_accuracy": round(null_hit / n, 4),
        "samples": samples,
    }


def main():
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate"])
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU in this Kaggle session.")
    status({"event": "cuda", "device": torch.cuda.get_device_name(0)})
    _, tokenizer, model = load_model()
    rows = generate_rows(model, tokenizer)
    ACCEPTED.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if len(rows) < 40:
        raise RuntimeError(f"Only {len(rows)} notes passed the checks.")
    rng = random.Random(SEED)
    rng.shuffle(rows)
    split = max(1, int(len(rows) * 0.2))
    test_rows, train_rows = rows[:split], rows[split:]
    cache_train = hidden_states(model, tokenizer, train_rows)
    cache_test = hidden_states(model, tokenizer, test_rows)
    del model
    torch.cuda.empty_cache()
    heads = train_heads(cache_train)
    report = {"train": evaluate(heads, cache_train), "test": evaluate(heads, cache_test)}
    METRICS.write_text(json.dumps(report, indent=2) + "\n")
    torch.save(heads.state_dict(), HEADS)
    status({"event": "done", "test": report["test"]})


if __name__ == "__main__":
    main()
