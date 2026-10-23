"""Exact-match scoring for the four heads.

A list or a forecast is one field. It is right only when every expanded
question is right. A document is right only when every field is right.

Point gold often starts one character inside a Gemma token, because the
tokenizer glues on the preceding space. Token-exact means the head picked
the covering tokens. Text-exact means those tokens' characters, stripped,
equal the gold string. Character-offset exact is reported beside them.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP

import torch

from extractor.labels import YEAR_BASE, join_number


def snap_to_step(value: float, low: float, step: float) -> Decimal:
    origin = Decimal(str(low))
    step_d = Decimal(str(step))
    units = (Decimal(str(value)) - origin) / step_d
    snapped = units.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return origin + snapped * step_d


def calendar_string(kind: str, indexes: dict) -> str:
    if kind == "time":
        return f"{indexes['hour']:02d}:{indexes['minute']:02d}"
    year = YEAR_BASE + indexes["year"]
    day = f"{year:04d}-{indexes['month'] + 1:02d}-{indexes['day'] + 1:02d}"
    if kind == "date":
        return day
    return f"{day}T{indexes['hour']:02d}:{indexes['minute']:02d}"


def _argmax_dict(outputs: dict[str, torch.Tensor]) -> dict[str, int]:
    return {name: int(value.argmax(dim=-1).view(-1)[0]) for name, value in outputs.items()}


def predict_question(heads: dict, question: dict, doc_states: torch.Tensor, threshold: float) -> dict:
    query = question["q_state"].detach().float().unsqueeze(0)
    if question["head"] == "point":
        outputs = heads["point"](doc_states.detach().float().unsqueeze(0), query)
        missing = bool(torch.sigmoid(outputs["missing"])[0] >= threshold)
        start = int(outputs["start"].argmax(dim=-1)[0])
        end = int(outputs["end"].argmax(dim=-1)[0])
        return {"missing": missing, "start": start, "end": end}
    if question["head"] == "number":
        outputs = heads["number"](query)
        sign = int(outputs["sign"].argmax(dim=-1)[0])
        whole = [int(item) for item in outputs["whole"].argmax(dim=-1)[0]]
        frac = [int(item) for item in outputs["frac"].argmax(dim=-1)[0]]
        encoded = {"sign": sign, "whole": whole, "frac": frac, "decimals": question["target"]["decimals"]}
        return {"number": join_number(encoded)}
    if question["head"] == "calendar":
        outputs = heads["calendar"](query)
        indexes = _argmax_dict(outputs)
        return {"text": calendar_string(question["kind"], indexes)}
    outputs = heads["spread"](query)
    mean = float(outputs["mean"][0].detach())
    width = float(outputs["width"][0].detach())
    low = float(question["min"])
    high = float(question["max"])
    value = mean * (high - low) + low
    return {"value": value, "mean": mean, "width": width}


def question_correct(question: dict, prediction: dict, doc_offsets: list[tuple[int, int]], text: str) -> dict:
    if question["head"] == "point":
        gold_missing = bool(question["missing"])
        missing_ok = prediction["missing"] == gold_missing
        token_ok = False
        text_ok = False
        char_ok = False
        if not gold_missing and not prediction["missing"] and prediction["end"] >= prediction["start"]:
            token_ok = prediction["start"] == question["start"] and prediction["end"] == question["end"]
            start_char = doc_offsets[prediction["start"]][0]
            end_char = doc_offsets[prediction["end"]][1]
            char_ok = (start_char, end_char) == (question["gold_start"], question["gold_end"])
            text_ok = text[start_char:end_char].strip() == question["value"]
        return {
            "correct": missing_ok and (gold_missing or token_ok),
            "token_exact": (not gold_missing) and token_ok,
            "text_exact": (not gold_missing) and text_ok,
            "char_exact": (not gold_missing) and char_ok,
            "missing_exact": missing_ok,
            "gold_missing": gold_missing,
        }
    if question["head"] == "number":
        gold = join_number(question["target"])
        return {"correct": prediction["number"] == gold}
    if question["head"] == "calendar":
        gold = calendar_string(question["kind"], question["target"])
        return {"correct": prediction["text"] == gold}
    snapped = snap_to_step(prediction["value"], question["min"], question["step"])
    gold = snap_to_step(question["value"], question["min"], question["step"])
    z = Decimal("1.2815515655446004")
    mean = Decimal(str(prediction["mean"]))
    width = Decimal(str(prediction["width"]))
    norm = Decimal(str(question["norm"]))
    inside = (mean - z * width) <= norm <= (mean + z * width)
    return {"correct": snapped == gold, "in_band": bool(inside)}


def score_document(heads: dict, record: dict, threshold: float) -> dict:
    doc_states = record["doc_states"]
    offsets = [tuple(pair) for pair in record["doc_offsets"]]
    by_field: dict[str, list[bool]] = defaultdict(list)
    questions = []
    for question in record["questions"]:
        prediction = predict_question(heads, question, doc_states, threshold)
        judged = question_correct(question, prediction, offsets, record["text"])
        by_field[question["field"]].append(judged["correct"])
        questions.append({"field": question["field"], "head": question["head"], "kind": question["kind"], **judged})
    fields = []
    for name, parts in by_field.items():
        head = next(item["head"] for item in questions if item["field"] == name)
        kind = next(item["kind"] for item in questions if item["field"] == name)
        fields.append({"field": name, "head": head, "kind": kind, "correct": all(parts)})
    return {
        "id": record["id"],
        "split": record["split"],
        "questions": questions,
        "fields": fields,
        "document_correct": all(item["correct"] for item in fields),
    }


def summarize(scored: list[dict]) -> dict:
    fields = [field for doc in scored for field in doc["fields"]]
    questions = [question for doc in scored for question in doc["questions"]]

    def rate(items: list[dict]) -> dict:
        total = len(items)
        correct = sum(1 for item in items if item["correct"])
        return {"correct": correct, "total": total, "exact": (correct / total) if total else None}

    by_head = {}
    for head in ("point", "number", "calendar", "spread"):
        by_head[head] = rate([item for item in fields if item["head"] == head])
    by_kind = {}
    for item in fields:
        by_kind.setdefault(item["kind"], []).append(item)
    return {
        "documents": rate([{"correct": doc["document_correct"]} for doc in scored]),
        "fields": rate(fields),
        "questions": rate(questions),
        "by_head": by_head,
        "by_kind": {kind: rate(items) for kind, items in sorted(by_kind.items())},
        "point_token_exact": rate(
            [{"correct": item["token_exact"]} for item in questions if item["head"] == "point" and not item["gold_missing"]]
        ),
        "point_char_exact": rate(
            [{"correct": item["char_exact"]} for item in questions if item["head"] == "point" and not item["gold_missing"]]
        ),
        "point_text_exact": rate(
            [{"correct": item["text_exact"]} for item in questions if item["head"] == "point" and not item["gold_missing"]]
        ),
        "spread_in_band": rate(
            [{"correct": item["in_band"]} for item in questions if item["head"] == "spread"]
        ),
    }
