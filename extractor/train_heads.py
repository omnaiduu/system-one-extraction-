"""Train the four heads on cached train states.

The epoch is the earliest one with the highest training field exact-match.
Training loss is not the stop signal: the spread head's Gaussian loss can
fall without bound by shrinking its width. Validation is only four documents,
so it is not used to rank epochs. It only picks the missing threshold, from
{0.5, 0.3, 0.7}. Test and the generalization exam are scored once, after that
choice is frozen.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from extractor.heads import (
    CalendarHead,
    NumberHead,
    PointHead,
    SpreadHead,
    calendar_loss,
    number_loss,
    point_loss,
    spread_loss,
)
from extractor.score_heads import score_document, summarize

CACHE = Path("outputs/cache/docs")
OUT = Path("outputs")
THRESHOLDS = (0.5, 0.3, 0.7)
EPOCHS = 80
MAX_EPOCHS = 120
SEED = 7
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01


def load_records() -> list[dict]:
    paths = sorted(CACHE.glob("*.pt"))
    if not paths:
        raise RuntimeError(f"no cached states in {CACHE}")
    return [torch.load(path, weights_only=False) for path in paths]


def build_heads(width: int) -> dict:
    return {
        "point": PointHead(width),
        "number": NumberHead(width),
        "calendar": CalendarHead(width),
        "spread": SpreadHead(width),
    }


def _batch_target(question: dict, key: str) -> torch.Tensor:
    value = question["target"][key]
    return torch.tensor([value], dtype=torch.long)


def loss_on(heads: dict, record: dict, question: dict) -> torch.Tensor:
    query = question["q_state"].float().unsqueeze(0)
    if question["head"] == "point":
        document = record["doc_states"].float().unsqueeze(0)
        outputs = heads["point"](document, query)
        missing = torch.tensor([1 if question["missing"] else 0], dtype=torch.long)
        start = torch.tensor([question["start"]], dtype=torch.long)
        end = torch.tensor([question["end"]], dtype=torch.long)
        return point_loss(outputs, missing, start, end)
    if question["head"] == "number":
        outputs = heads["number"](query)
        target = {
            "sign": _batch_target(question, "sign"),
            "whole": torch.tensor([question["target"]["whole"]], dtype=torch.long),
            "frac": torch.tensor([question["target"]["frac"]], dtype=torch.long),
        }
        return number_loss(outputs, target)
    if question["head"] == "calendar":
        outputs = heads["calendar"](query)
        target = {
            name: None if question["target"][name] is None else torch.tensor([question["target"][name]], dtype=torch.long)
            for name in ("year", "month", "day", "hour", "minute")
        }
        return calendar_loss(outputs, target, question["kind"])
    outputs = heads["spread"](query)
    return spread_loss(outputs, torch.tensor([question["norm"]], dtype=torch.float32))


def evaluate(heads: dict, records: list[dict], threshold: float) -> tuple[list[dict], dict]:
    scored = []
    with torch.no_grad():
        for record in records:
            scored.append(score_document(heads, record, threshold))
    return scored, summarize(scored)


def val_key(summary: dict) -> tuple:
    return (summary["fields"]["exact"] or 0.0, summary["documents"]["exact"] or 0.0)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    records = load_records()
    splits = {}
    for record in records:
        splits.setdefault(record["split"], []).append(record)
    required = ["train", "val", "test", "generalization"]
    if args.round == 2:
        required.append("exam2")
    for name in required:
        if name not in splits:
            raise RuntimeError(f"cached states are missing the {name} split")
    width = int(splits["train"][0]["doc_states"].shape[-1])
    heads = build_heads(width)
    initial = {name: {key: value.detach().clone() for key, value in head.state_dict().items()} for name, head in heads.items()}
    parameters = [param for head in heads.values() for param in head.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history = []
    best_fields = None
    best_epoch = None
    best_state = None
    train_questions = [
        (record, question) for record in splits["train"] for question in record["questions"]
    ]
    epoch = 0
    limit = EPOCHS
    while epoch < limit:
        epoch += 1
        random.shuffle(train_questions)
        total = 0.0
        for head in heads.values():
            head.train()
        for record, question in train_questions:
            optimizer.zero_grad()
            loss = loss_on(heads, record, question)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total += float(loss.detach())
        for head in heads.values():
            head.eval()
        train_loss = total / len(train_questions)
        _, train_summary = evaluate(heads, splits["train"], 0.5)
        _, val_summary = evaluate(heads, splits["val"], 0.5)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_fields": train_summary["fields"]["exact"],
            "val_fields_at_0.5": val_summary["fields"]["exact"],
        }
        history.append(row)
        train_fields = train_summary["fields"]["exact"]
        if best_fields is None or train_fields > best_fields:
            best_fields = train_fields
            best_epoch = epoch
            best_state = {name: {key: value.detach().clone() for key, value in head.state_dict().items()} for name, head in heads.items()}
        print(
            f"epoch {epoch} loss {train_loss:.4f} train_fields {train_summary['fields']['exact']:.3f} "
            f"val_fields {val_summary['fields']['exact']:.3f} best_epoch {best_epoch}",
            flush=True,
        )
        if epoch == limit and best_epoch >= limit - 5 and limit < MAX_EPOCHS:
            limit = min(MAX_EPOCHS, limit + 20)
            print(f"train loss still falling, extending to {limit} epochs", flush=True)

    def apply(state):
        for name, head in heads.items():
            head.load_state_dict(state[name])
            head.eval()

    apply(best_state)
    threshold_scores = {}
    chosen_threshold = None
    chosen_key = None
    for threshold in THRESHOLDS:
        _, summary = evaluate(heads, splits["val"], threshold)
        threshold_scores[str(threshold)] = {"fields": summary["fields"], "documents": summary["documents"]}
        key = val_key(summary)
        if chosen_key is None or key > chosen_key:
            chosen_key = key
            chosen_threshold = threshold
    selection = {
        "epoch": best_epoch,
        "train_fields": best_fields,
        "threshold": chosen_threshold,
        "val_at_threshold": threshold_scores[str(chosen_threshold)],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    scored_splits = [name for name in splits if not (args.round == 2 and name == "generalization")]
    apply(initial)
    before = {}
    for name in scored_splits:
        _, before[name] = evaluate(heads, splits[name], chosen_threshold)
    apply(best_state)
    after = {}
    predictions = {}
    for name in scored_splits:
        scored, after[name] = evaluate(heads, splits[name], chosen_threshold)
        predictions[name] = scored
    payload = {
        "seed": SEED,
        "epochs_run": epoch,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "selection": selection,
        "threshold_scores_on_val": threshold_scores,
        "selection_rule": "earliest epoch with the highest training field exact-match. If that epoch is within the last 5 of the budget, extend by 20 up to 120. Then the missing threshold in {0.5, 0.3, 0.7} with the highest validation field exact-match, then document exact-match. Ties keep the earlier threshold in that order. Training loss is not used to pick the epoch, because the spread loss can fall without bound.",
        "before": before,
        "after": after,
        "history": history,
        "weights": "public Hugging Face google/gemma-4-E2B-it; no Kaggle token was on this machine",
        "readout": "last question token, because the document is a causal prefix",
        "point_rule": "token span exact; character offsets are also recorded and are often one space wider",
    }
    if args.round == 2:
        payload["round"] = 2
        payload["claim_split"] = "exam2"
        payload["note"] = "exam2 is the fresh exam. The first generalization exam is not the improvement claim."
        score_name = "scores_round2.json"
        prediction_name = "predictions_round2.json"
        weight_name = "heads_round2.pt"
    else:
        score_name = "scores.json"
        prediction_name = "predictions.json"
        weight_name = "heads.pt"
    (OUT / score_name).write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / prediction_name).write_text(json.dumps(predictions, indent=2) + "\n")
    torch.save(
        {
            "heads": best_state,
            "width": width,
            "selection": selection,
            "seed": SEED,
            "round": args.round,
        },
        OUT / weight_name,
    )
    print(f"wrote outputs/{score_name} outputs/{weight_name}", flush=True)


if __name__ == "__main__":
    main()
