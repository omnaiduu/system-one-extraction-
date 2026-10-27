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

from extractor.gemma_cache import HIGHLIGHT_BLOCKED, HIGHLIGHT_CACHE
from extractor.heads import PointHead, choose_span, choose_spans, point_loss
from extractor.highlight import dumps_public, group_questions, judge_group, summarize_highlighter

OUT = Path("outputs")
THRESHOLDS = (0.5, 0.3, 0.7)
EPOCHS = 80
MAX_EPOCHS = 120
SEED = 7
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
SCORED_SPLITS = ("train", "val", "test", "generalization", "exam2", "exam3", "exam4")


def load_records(cache: Path, blocked: bool) -> list[dict]:
    paths = sorted(cache.glob("*.pt"))
    if not paths:
        raise RuntimeError(f"no highlighter states in {cache}")
    records = [torch.load(path, weights_only=False) for path in paths]
    if blocked and not all(record.get("blocked") for record in records):
        raise RuntimeError("blocked training needs question-blocked states")
    return records


def span_limits(records: list[dict]) -> dict[str, int]:
    """Longest gold span of each operation on the training documents, in tokens."""
    limits: dict[str, int] = {}
    for record in records:
        if record["split"] != "train":
            continue
        for question in record["questions"]:
            if question["missing"]:
                continue
            length = int(question["end"]) - int(question["start"]) + 1
            if length < 1:
                raise RuntimeError(f"{record['id']} {question['field']} has an empty gold span")
            op = question["recipe"]["op"]
            limits[op] = max(limits.get(op, 1), length)
    if not limits:
        raise RuntimeError("training records have no spans to measure")
    return limits


def predict_span(
    head: PointHead,
    question: dict,
    doc_states: torch.Tensor,
    threshold: float,
    limits: dict[str, int],
) -> dict:
    query = question["q_state"].detach().float().unsqueeze(0)
    outputs = head(doc_states.detach().float().unsqueeze(0), query)
    missing = bool(torch.sigmoid(outputs["missing"])[0] >= threshold)
    op = question["recipe"]["op"]
    if op in limits:
        limit = limits[op]
    elif op == "missing":
        limit = int(doc_states.shape[0])
    else:
        raise RuntimeError(f"no training span for {op}")
    start, end = choose_span(outputs["start"][0], outputs["end"][0], limit)
    return {"missing": missing, "start": start, "end": end}


def score_record(head: PointHead, record: dict, threshold: float, limits: dict[str, int]) -> dict:
    offsets = [tuple(pair) for pair in record["doc_offsets"]]
    fields = []
    for group in group_questions(record["questions"]):
        predictions = [
            predict_span(head, question, record["doc_states"], threshold, limits) for question in group
        ]
        op = group[0]["recipe"]["op"]
        if op in {"list", "sum"}:
            _chain_present_parts(head, group, record["doc_states"], predictions, limits[op])
        fields.append(judge_group(group, predictions, offsets, record["text"]))
    return {
        "id": record["id"],
        "split": record["split"],
        "skipped": list(record.get("skipped", [])),
        "fields": fields,
        "document_span": bool(fields) and all(field["span_exact"] for field in fields),
        "document_value": bool(fields) and all(field["value_exact"] for field in fields),
    }


def _chain_present_parts(
    head: PointHead,
    group: list[dict],
    doc_states: torch.Tensor,
    predictions: list[dict],
    limit: int,
) -> None:
    """List and sum parts keep their missing bits. Present parts are placed in text order."""
    present = [index for index, prediction in enumerate(predictions) if not prediction["missing"]]
    if len(present) < 2:
        return
    starts = []
    ends = []
    for index in present:
        query = group[index]["q_state"].detach().float().unsqueeze(0)
        outputs = head(doc_states.detach().float().unsqueeze(0), query)
        starts.append(outputs["start"][0])
        ends.append(outputs["end"][0])
    for index, (start, end) in zip(present, choose_spans(starts, ends, limit)):
        predictions[index]["start"] = start
        predictions[index]["end"] = end


def evaluate(
    head: PointHead,
    records: list[dict],
    threshold: float,
    limits: dict[str, int],
) -> tuple[list[dict], dict]:
    scored = []
    with torch.no_grad():
        for record in records:
            scored.append(score_record(head, record, threshold, limits))
    return scored, summarize_highlighter(scored)


def load_point(path: Path) -> tuple[PointHead, dict]:
    blob = torch.load(path, weights_only=False)
    state = dict(blob["point"])
    if "missing.weight" in state:
        state["missing_question.weight"] = state.pop("missing.weight")
        state["missing_question.bias"] = state.pop("missing.bias")
        state["missing_match.weight"] = torch.zeros(1, 2)
        state["missing_match.bias"] = torch.zeros(1)
    head = PointHead(int(blob["width"]))
    head.load_state_dict(state)
    head.eval()
    return head, blob


def rescore(records: list[dict], blocked: bool) -> None:
    """Score a saved head again. The weights and the missing threshold stay as they were."""
    suffix = "_blocked" if blocked else ""
    head, blob = load_point(OUT / f"heads_highlighter{suffix}.pt")
    limits = span_limits(records)
    threshold = float(blob["selection"]["threshold"])
    splits: dict[str, list[dict]] = {}
    for record in records:
        splits.setdefault(record["split"], []).append(record)
    predictions = {}
    after = {}
    longer = {}
    for name in SCORED_SPLITS:
        if name not in splits:
            continue
        scored, after[name] = evaluate(head, splits[name], threshold, limits)
        predictions[name] = scored
        extra = 0
        for record in splits[name]:
            for question in record["questions"]:
                if question["missing"]:
                    continue
                length = int(question["end"]) - int(question["start"]) + 1
                if length > limits[question["recipe"]["op"]]:
                    extra += 1
        longer[name] = extra
    payload = {
        "weights_file": f"outputs/heads_highlighter{suffix}.pt",
        "threshold": threshold,
        "threshold_note": "Kept from the saved run. The cap and the chain were not used to choose the epoch or the threshold.",
        "span_limit_rule": (
            "For each operation, a span may be at most as many tokens as the longest gold span of that "
            "operation on the training documents. A list or a sum places its present parts in one chain: "
            "each later part starts after the previous part's end, and the chain is the one with the "
            "highest total of start score plus end score. Other operations still pick each span on its own. "
            "The missing score is unchanged."
        ),
        "span_limits": limits,
        "gold_spans_longer_than_the_training_max": longer,
        "blocked": blocked,
        "after": after,
        "readout": (
            "same saved point head. Each span stays inside the longest training span of that operation. "
            "List and sum parts that are marked present are placed in text order, each one starting "
            "after the previous one ends. The program then reads the marked characters."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    score_path = OUT / f"scores_highlighter{suffix}_chained.json"
    pred_path = OUT / f"predictions_highlighter{suffix}_chained.json"
    score_path.write_text(dumps_public(payload))
    pred_path.write_text(dumps_public(predictions))
    train = after["train"]
    exam = after["exam4"]
    print(
        f"train span {train['fields_span']['correct']}/{train['fields_span']['total']} "
        f"value {train['fields_value']['correct']}/{train['fields_value']['total']}",
        flush=True,
    )
    print(
        f"exam4 span {exam['fields_span']['correct']}/{exam['fields_span']['total']} "
        f"value {exam['fields_value']['correct']}/{exam['fields_value']['total']}",
        flush=True,
    )
    for split_name, summary in (("train", train), ("exam4", exam)):
        for op in ("list", "sum"):
            row = summary["by_op_value"][op]
            print(f"{split_name} {op} {row['correct']}/{row['total']}", flush=True)
    print(f"wrote {score_path} {pred_path}", flush=True)


def loss_on(head: PointHead, record: dict, question: dict) -> torch.Tensor:
    query = question["q_state"].float().unsqueeze(0)
    document = record["doc_states"].float().unsqueeze(0)
    outputs = head(document, query)
    missing = torch.tensor([1 if question["missing"] else 0], dtype=torch.long)
    start = torch.tensor([question["start"]], dtype=torch.long)
    end = torch.tensor([question["end"]], dtype=torch.long)
    return point_loss(outputs, missing, start, end)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--blocked", action="store_true", help="train on question states that cannot see the text")
    parser.add_argument("--rescore", action="store_true", help="score the saved head with the training span cap")
    args = parser.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    cache = HIGHLIGHT_BLOCKED if args.blocked else HIGHLIGHT_CACHE
    records = load_records(cache, args.blocked)
    if args.rescore:
        rescore(records, args.blocked)
        return
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
    limits = span_limits(records)
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
        _, train_summary = evaluate(head, splits["train"], 0.5, limits)
        _, val_summary = evaluate(head, splits["val"], 0.5, limits)
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
        _, summary = evaluate(head, splits["val"], threshold, limits)
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
        _, before[name] = evaluate(head, splits[name], chosen_threshold, limits)
    apply(best_state)
    after = {}
    predictions = {}
    for name in scored_names:
        scored, after[name] = evaluate(head, splits[name], chosen_threshold, limits)
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
            "one point head. Question tokens cannot attend to the text. "
            "The missing score reads the question state and the best start and end match. "
            "The marked span is the best start-plus-end pair no longer than the longest "
            "training span of that operation. List and sum parts marked present are placed in "
            "text order. The program reads the marked spans."
            if args.blocked
            else
            "one point head. The question-end state is the same causal state as before: "
            "it has seen the text and the question. The marked span is the best start-plus-end "
            "pair no longer than the longest training span of that operation. List and sum parts "
            "marked present are placed in text order. The program reads the marked spans."
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
    suffix = "_blocked" if args.blocked else ""
    (OUT / f"scores_highlighter{suffix}.json").write_text(dumps_public(payload))
    (OUT / f"predictions_highlighter{suffix}.json").write_text(dumps_public(predictions))
    torch.save(
        {
            "point": best_state,
            "width": width,
            "selection": payload["selection"],
            "seed": SEED,
            "blocked": args.blocked,
        },
        OUT / f"heads_highlighter{suffix}.pt",
    )
    exam = after["exam4"]
    print(
        f"exam4 span {exam['fields_span']['correct']}/{exam['fields_span']['total']} "
        f"value {exam['fields_value']['correct']}/{exam['fields_value']['total']}",
        flush=True,
    )
    print(f"wrote outputs/scores_highlighter{suffix}.json outputs/heads_highlighter{suffix}.pt", flush=True)


if __name__ == "__main__":
    main()
