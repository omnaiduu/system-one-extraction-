"""Train one highlighter on the documents we already have.

The training texts are the same 200. The 24-document exams already on disk are
scored after the epoch is frozen. No new documents are written. Forecast
fields are left out: the answer is not in the text, so the highlighter has
nothing to mark. The program turns the marked spans into the typed value.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch

from extractor.gemma_cache import HIGHLIGHT_CACHE
from extractor.heads import PointHead, point_loss
from extractor.highlight import dumps_public, group_questions, judge_group, summarize_highlighter

OUT = Path("outputs")
THRESHOLDS = (0.5, 0.3, 0.7)
EPOCHS = 80
MAX_EPOCHS = 120
SEED = 7
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
SCORED_SPLITS = ("train", "val", "test", "generalization", "exam2", "exam3", "exam4")


def load_records() -> list[dict]:
    paths = sorted(HIGHLIGHT_CACHE.glob("*.pt"))
    if not paths:
        raise RuntimeError(f"no highlighter states in {HIGHLIGHT_CACHE}")
    return [torch.load(path, weights_only=False) for path in paths]


def predict_span(head: PointHead, question: dict, doc_states: torch.Tensor, threshold: float) -> dict:
    query = question["q_state"].detach().float().unsqueeze(0)
    outputs = head(doc_states.detach().float().unsqueeze(0), query)
    missing = bool(torch.sigmoid(outputs["missing"])[0] >= threshold)
    start = int(outputs["start"].argmax(dim=-1)[0])
    end = int(outputs["end"].argmax(dim=-1)[0])
    return {"missing": missing, "start": start, "end": end}


def score_record(head: PointHead, record: dict, threshold: float) -> dict:
    offsets = [tuple(pair) for pair in record["doc_offsets"]]
    fields = []
    for group in group_questions(record["questions"]):
        predictions = [predict_span(head, question, record["doc_states"], threshold) for question in group]
        fields.append(judge_group(group, predictions, offsets, record["text"]))
    return {
        "id": record["id"],
        "split": record["split"],
        "skipped": list(record.get("skipped", [])),
        "fields": fields,
        "document_span": bool(fields) and all(field["span_exact"] for field in fields),
        "document_value": bool(fields) and all(field["value_exact"] for field in fields),
    }


def evaluate(head: PointHead, records: list[dict], threshold: float) -> tuple[list[dict], dict]:
    scored = []
    with torch.no_grad():
        for record in records:
            scored.append(score_record(head, record, threshold))
    return scored, summarize_highlighter(scored)


def loss_on(head: PointHead, record: dict, question: dict) -> torch.Tensor:
    query = question["q_state"].float().unsqueeze(0)
    document = record["doc_states"].float().unsqueeze(0)
    outputs = head(document, query)
    missing = torch.tensor([1 if question["missing"] else 0], dtype=torch.long)
    start = torch.tensor([question["start"]], dtype=torch.long)
    end = torch.tensor([question["end"]], dtype=torch.long)
    return point_loss(outputs, missing, start, end)


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    records = load_records()
    splits: dict[str, list[dict]] = {}
    for record in records:
        splits.setdefault(record["split"], []).append(record)
    for name in ("train", "val", "exam4"):
        if name not in splits:
            raise RuntimeError(f"cached states are missing the {name} split")
    width = int(splits["train"][0]["doc_states"].shape[-1])
    head = PointHead(width)
    initial = {key: value.detach().clone() for key, value in head.state_dict().items()}
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_questions = [
        (record, question) for record in splits["train"] for question in record["questions"]
    ]
    history = []
    best_fields = None
    best_epoch = None
    best_state = None
    epoch = 0
    limit = EPOCHS
    while epoch < limit:
        epoch += 1
        random.shuffle(train_questions)
        head.train()
        total = 0.0
        for record, question in train_questions:
            optimizer.zero_grad()
            loss = loss_on(head, record, question)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
        head.eval()
        train_loss = total / len(train_questions)
        _, train_summary = evaluate(head, splits["train"], 0.5)
        _, val_summary = evaluate(head, splits["val"], 0.5)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_fields_span": train_summary["fields_span"]["exact"],
            "train_fields_value": train_summary["fields_value"]["exact"],
            "val_fields_span": val_summary["fields_span"]["exact"],
        }
        history.append(row)
        train_fields = train_summary["fields_span"]["exact"]
        if best_fields is None or train_fields > best_fields:
            best_fields = train_fields
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
        print(
            f"epoch {epoch} loss {train_loss:.4f} "
            f"train_span {train_summary['fields_span']['correct']}/{train_summary['fields_span']['total']} "
            f"train_value {train_summary['fields_value']['correct']}/{train_summary['fields_value']['total']} "
            f"val_span {val_summary['fields_span']['correct']}/{val_summary['fields_span']['total']} "
            f"best_epoch {best_epoch}",
            flush=True,
        )
        if epoch == limit and best_epoch >= limit - 5 and limit < MAX_EPOCHS:
            limit = min(MAX_EPOCHS, limit + 20)
            print(f"train span exact still rising, extending to {limit} epochs", flush=True)

    head.load_state_dict(best_state)
    head.eval()
    threshold_scores = {}
    chosen_threshold = None
    chosen_key = None
    for threshold in THRESHOLDS:
        _, summary = evaluate(head, splits["val"], threshold)
        threshold_scores[str(threshold)] = {
            "fields_span": summary["fields_span"],
            "documents_span": summary["documents_span"],
        }
        key = (summary["fields_span"]["exact"] or 0.0, summary["documents_span"]["exact"] or 0.0)
        if chosen_key is None or key > chosen_key:
            chosen_key = key
            chosen_threshold = threshold

    def apply(state):
        head.load_state_dict(state)
        head.eval()

    scored_names = [name for name in SCORED_SPLITS if name in splits]
    apply(initial)
    before = {}
    for name in scored_names:
        _, before[name] = evaluate(head, splits[name], chosen_threshold)
    apply(best_state)
    after = {}
    predictions = {}
    for name in scored_names:
        scored, after[name] = evaluate(head, splits[name], chosen_threshold)
        predictions[name] = scored
    payload = {
        "seed": SEED,
        "epochs_run": epoch,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "selection": {
            "epoch": best_epoch,
            "train_fields_span": best_fields,
            "threshold": chosen_threshold,
            "val_at_threshold": threshold_scores[str(chosen_threshold)],
        },
        "threshold_scores_on_val": threshold_scores,
        "selection_rule": (
            "earliest epoch with the highest training field span exact-match. "
            "If that epoch is within the last 5 of the budget, extend by 20 up to 120. "
            "Then the missing threshold in {0.5, 0.3, 0.7} with the highest validation "
            "field span exact-match, then document span exact-match. Ties keep the earlier threshold."
        ),
        "before": before,
        "after": after,
        "history": history,
        "weights": "public Hugging Face google/gemma-4-E2B-it",
        "readout": (
            "one point head. The question-end state is the same causal state as before: "
            "it has seen the text and the question. The program reads the marked spans."
        ),
        "data": (
            "training texts are the existing 200. Exams are the four 24-document sets already on disk. "
            "No new documents were written. Forecast fields are not scored."
        ),
        "claim_split": "exam4",
        "note": (
            "exam4 is the latest 24 already written. generalization, exam2, and exam3 are the earlier "
            "24s, scored again with this head. They were not used to pick the epoch."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scores_highlighter.json").write_text(dumps_public(payload))
    (OUT / "predictions_highlighter.json").write_text(dumps_public(predictions))
    torch.save(
        {
            "point": best_state,
            "width": width,
            "selection": payload["selection"],
            "seed": SEED,
        },
        OUT / "heads_highlighter.pt",
    )
    exam = after["exam4"]
    print(
        f"exam4 span {exam['fields_span']['correct']}/{exam['fields_span']['total']} "
        f"value {exam['fields_value']['correct']}/{exam['fields_value']['total']}",
        flush=True,
    )
    print("wrote outputs/scores_highlighter.json outputs/heads_highlighter.pt", flush=True)


if __name__ == "__main__":
    main()
