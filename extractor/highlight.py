"""One highlighter, then a program.

Every answer that is written in the text, or that is calculated from pieces
written in the text, becomes one or more span questions. The highlighter marks
those spans. This module reads the marked characters and applies the same rule
the auditor used: copy, sum, product, subtract, parse a date, count days, add
days, or the median of an even set of figures.

A forecast is not included. Its answer was removed from the text, so there is
no span to mark.
"""

from __future__ import annotations

import json
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from extractor.audit import (
    Drop,
    format_number,
    label_document,
    load_jsonl,
    parse_date,
    parse_datetime,
    parse_number,
    parse_time,
    quantize,
)
from extractor.labels import _point


def _recipe(op: str, kind: str, gold, **extra) -> dict:
    recipe = {"op": op, "kind": kind, "gold": gold}
    recipe.update(extra)
    return recipe


def _attach(question: dict, recipe: dict, part: int, role: str) -> dict:
    question["recipe"] = recipe
    question["part"] = part
    question["role"] = role
    return question


def _pieces(field: dict, text: str) -> list[dict]:
    kind = field["type"]
    if kind == "list":
        return [
            {"start": start, "end": end, "text": value}
            for (start, end), value in zip(field["spans"], field["value"])
        ]
    evidence = field.get("evidence")
    if isinstance(evidence, list):
        return evidence
    if field.get("start") is None:
        raise ValueError(f"{kind} has no span")
    raw = evidence if isinstance(evidence, str) else text[field["start"]:field["end"]]
    return [{"start": field["start"], "end": field["end"], "text": raw}]


def _op(field: dict) -> tuple[str, dict]:
    kind = field["type"]
    compute = field.get("compute")
    if kind == "text":
        return "copy", {}
    if kind == "list":
        return "list", {}
    if kind == "number" and compute is None:
        return "read_number", {"decimals": int(field.get("decimals", 2))}
    if kind == "number" and compute in {"sum", "product", "subtract"}:
        return compute, {"decimals": int(field.get("decimals", 2))}
    if kind == "duration":
        return "days_between", {"dayfirst": bool(field.get("dayfirst", True))}
    if kind == "date" and compute is None:
        return "read_date", {"dayfirst": bool(field.get("dayfirst", True))}
    if kind == "time":
        return "read_time", {}
    if kind == "datetime":
        return "read_datetime", {"dayfirst": bool(field.get("dayfirst", True))}
    if kind == "date" and compute == "add_days":
        return "add_days", {"dayfirst": bool(field.get("dayfirst", True))}
    if kind == "estimate":
        return "median", {
            "decimals": int(field.get("decimals", 0)),
            "min": field["min"],
            "max": field["max"],
            "step": field["step"],
        }
    raise ValueError(f"no highlighter op for {kind} {compute}")


def expand_highlighter(row: dict) -> tuple[list[dict], list[str]]:
    """Point questions for every evidence span, plus the names of skipped forecasts."""
    text = row["text"]
    questions = []
    skipped = []
    for name, field in row["fields"].items():
        instruction = field["instruction"]
        kind = field["type"]
        if kind == "forecast":
            skipped.append(name)
            continue
        if kind == "text" and field.get("missing"):
            recipe = _recipe("missing", kind, None)
            questions.append(_attach(_point(name, instruction, True, None, None, None, kind), recipe, 0, "missing"))
            continue
        pieces = _pieces(field, text)
        op, extra = _op(field)
        recipe = _recipe(op, kind, field.get("value"), **extra)
        if op == "list":
            label = "Item"
        elif op in {"sum", "product", "subtract", "median"}:
            label = "Part"
        else:
            label = None
        if label is not None:
            for index, piece in enumerate(pieces, start=1):
                questions.append(_attach(
                    _point(name, f"{instruction} {label} {index}.", False, piece["start"], piece["end"], piece["text"], kind),
                    recipe,
                    index,
                    "part",
                ))
            questions.append(_attach(
                _point(name, f"{instruction} {label} {len(pieces) + 1}.", True, None, None, None, kind),
                recipe,
                len(pieces) + 1,
                "stop",
            ))
            continue
        if op == "days_between":
            roles = ("The start date.", "The end date.")
        elif op == "add_days":
            roles = ("The start date.", "The day count.")
        else:
            roles = None
        if roles is not None:
            if len(pieces) != len(roles):
                raise ValueError(f"{name} expected {len(roles)} spans, found {len(pieces)}")
            for index, (piece, role) in enumerate(zip(pieces, roles), start=1):
                questions.append(_attach(
                    _point(name, f"{instruction} {role}", False, piece["start"], piece["end"], piece["text"], kind),
                    recipe,
                    index,
                    "part",
                ))
            continue
        piece = pieces[0]
        questions.append(_attach(
            _point(name, instruction, False, piece["start"], piece["end"], piece["text"], kind),
            recipe,
            1,
            "part",
        ))
    return questions, skipped


def render_gold(recipe: dict):
    op = recipe["op"]
    gold = recipe["gold"]
    if op == "missing":
        return None
    if op == "list":
        return list(gold)
    if op in {"copy", "read_date", "read_time", "read_datetime", "add_days"}:
        return gold
    if op == "days_between":
        return int(gold)
    return format_number(Decimal(str(gold)), int(recipe["decimals"]))


def apply_recipe(recipe: dict, pieces: list[str]):
    """Turn the marked strings into the typed value. `pieces` excludes the stop question."""
    op = recipe["op"]
    if op == "missing":
        if pieces:
            raise Drop("missing_field_has_spans")
        return None
    if not pieces or any(piece is None or not str(piece).strip() for piece in pieces):
        raise Drop("blank_span")
    cleaned = [piece.strip() for piece in pieces]
    if op == "copy":
        return cleaned[0]
    if op == "list":
        return cleaned
    if op == "read_number":
        return format_number(parse_number(cleaned[0]), int(recipe["decimals"]))
    if op in {"sum", "product", "subtract"}:
        numbers = [parse_number(piece) for piece in cleaned]
        if op == "sum":
            value = sum(numbers, Decimal("0"))
        elif op == "product":
            if len(numbers) != 2:
                raise Drop("product_needs_two")
            value = numbers[0] * numbers[1]
        else:
            if len(numbers) != 2:
                raise Drop("subtract_needs_two")
            value = numbers[0] - numbers[1]
        return format_number(value, int(recipe["decimals"]))
    dayfirst = bool(recipe.get("dayfirst", True))
    if op == "read_date":
        return parse_date(cleaned[0], dayfirst).isoformat()
    if op == "read_time":
        return parse_time(cleaned[0]).strftime("%H:%M")
    if op == "read_datetime":
        return parse_datetime(cleaned[0], dayfirst).strftime("%Y-%m-%dT%H:%M")
    if op == "days_between":
        if len(cleaned) != 2:
            raise Drop("duration_needs_two_dates")
        start = parse_date(cleaned[0], dayfirst)
        end = parse_date(cleaned[1], dayfirst)
        return (end - start).days
    if op == "add_days":
        if len(cleaned) != 2:
            raise Drop("add_days_needs_two")
        start = parse_date(cleaned[0], dayfirst)
        days = int(parse_number(cleaned[1]))
        return (start + timedelta(days=days)).isoformat()
    if op == "median":
        numbers = sorted(parse_number(piece) for piece in cleaned)
        if len(numbers) < 2 or len(numbers) % 2 != 0:
            raise Drop("median_needs_even_count")
        mid = len(numbers) // 2
        median = quantize((numbers[mid - 1] + numbers[mid]) / Decimal(2), int(recipe["decimals"]))
        return format_number(median, int(recipe["decimals"]))
    raise Drop(f"bad_op:{op}")


def values_match(recipe: dict, got) -> bool:
    return got == render_gold(recipe)


def evidence_pieces(field: dict, text: str) -> list[str]:
    if field["type"] == "forecast" or (field["type"] == "text" and field.get("missing")):
        return []
    return [piece["text"] for piece in _pieces(field, text)]


def program_matches_field(field: dict, text: str) -> bool:
    if field["type"] == "forecast":
        return True
    if field["type"] == "text" and field.get("missing"):
        return field.get("value") is None
    op, extra = _op(field)
    recipe = _recipe(op, field["type"], field.get("value"), **extra)
    got = apply_recipe(recipe, evidence_pieces(field, text))
    return values_match(recipe, got)


def raw_index(root: Path = Path("data")) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for path in sorted(root.rglob("*.jsonl")):
        if "raw" not in path.parts:
            continue
        for row in load_jsonl(path):
            if row.get("_error"):
                raise RuntimeError(f"bad raw json {row['id']}")
            if row["id"] in found:
                raise RuntimeError(f"duplicate raw id {row['id']}")
            found[row["id"]] = row
    return found


def _same(left, right) -> bool:
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    if isinstance(left, float) or isinstance(right, float):
        return left == right
    return left == right


def align_row(audited: dict, raw: dict) -> dict:
    """Keep the saved row. Copy operand spans from the marks when the value still matches.

    Two saved rows have a date field the current marks no longer ask for. Those
    dates already carry character offsets, so the saved field is kept as it is.
    """
    fresh = label_document(raw)
    if fresh["id"] != audited["id"] or fresh["text"] != audited["text"]:
        raise RuntimeError(f"{audited['id']} text changed when the marks were read again")
    merged = {
        "id": audited["id"],
        "genre": audited.get("genre", fresh.get("genre", "")),
        "text": audited["text"],
        "fields": {},
    }
    for name, field in audited["fields"].items():
        other = fresh["fields"].get(name)
        if other is None:
            if field["type"] in {"duration", "forecast"} or field.get("compute") in {"add_days", "sum", "product", "subtract"}:
                if "evidence" not in field and "start" not in field and "spans" not in field:
                    raise RuntimeError(f"{audited['id']} {name} has no span and the marks no longer contain it")
            merged["fields"][name] = field
            continue
        if field["type"] != other["type"] or not _same(field.get("value"), other.get("value")):
            raise RuntimeError(f"{audited['id']} {name} value changed when the marks were read again")
        if bool(field.get("missing")) != bool(other.get("missing")):
            raise RuntimeError(f"{audited['id']} {name} missing bit changed")
        merged["fields"][name] = other
    return merged


def verify_corpus(raw_rows: list[dict] | None = None) -> int:
    rows = raw_rows if raw_rows is not None else list(raw_index().values())
    checked = 0
    for raw in rows:
        labeled = label_document(raw)
        for name, field in labeled["fields"].items():
            if not program_matches_field(field, labeled["text"]):
                raise RuntimeError(f"{labeled['id']} {name} program does not reproduce the audited value")
            checked += 1
        questions, skipped = expand_highlighter(labeled)
        if skipped != [name for name, field in labeled["fields"].items() if field["type"] == "forecast"]:
            raise RuntimeError(f"{labeled['id']} skipped the wrong fields")
        for question in questions:
            if question["missing"]:
                continue
            start, end = question["start"], question["end"]
            if labeled["text"][start:end] != question["value"]:
                raise RuntimeError(f"{labeled['id']} {question['field']} span text drifted")
    return checked


def slice_text(prediction: dict, offsets: list[tuple[int, int]], text: str) -> str | None:
    if prediction["missing"] or prediction["end"] < prediction["start"]:
        return None
    if prediction["start"] < 0 or prediction["end"] >= len(offsets):
        return None
    start_char = offsets[prediction["start"]][0]
    end_char = offsets[prediction["end"]][1]
    return text[start_char:end_char].strip()


def judge_group(questions: list[dict], predictions: list[dict], offsets: list[tuple[int, int]], text: str) -> dict:
    recipe = questions[0]["recipe"]
    flags_match = all(bool(pred["missing"]) == bool(question["missing"]) for question, pred in zip(questions, predictions))
    tokens_match = flags_match
    question_rows = []
    for question, pred in zip(questions, predictions):
        token_ok = False
        if not question["missing"] and not pred["missing"] and pred["end"] >= pred["start"]:
            token_ok = pred["start"] == question["start"] and pred["end"] == question["end"]
        if not question["missing"] and not token_ok:
            tokens_match = False
        question_rows.append({
            "part": question["part"],
            "role": question["role"],
            "missing": bool(pred["missing"]),
            "gold_missing": bool(question["missing"]),
            "token_exact": (not question["missing"]) and token_ok,
        })
    pieces = []
    predicted = []
    for question, pred in zip(questions, predictions):
        if question["role"] == "stop" or question["missing"]:
            continue
        snippet = slice_text(pred, offsets, text)
        predicted.append(snippet)
        pieces.append(snippet)
    got = None
    value_ok = False
    if flags_match:
        try:
            got = apply_recipe(recipe, pieces)
            value_ok = values_match(recipe, got)
        except (Drop, ValueError, ArithmeticError):
            got = None
            value_ok = False
    return {
        "field": questions[0]["field"],
        "kind": recipe["kind"],
        "op": recipe["op"],
        "span_exact": tokens_match,
        "value_exact": value_ok,
        "gold": render_gold(recipe),
        "got": got,
        "predicted": predicted,
        "questions": question_rows,
    }


def group_questions(questions: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []
    name = None
    for question in questions:
        if question["field"] != name:
            if current:
                groups.append(current)
            current = [question]
            name = question["field"]
        else:
            current.append(question)
    if current:
        groups.append(current)
    return groups


def summarize_highlighter(scored: list[dict]) -> dict:
    def rate(flags: list[bool]) -> dict:
        total = len(flags)
        correct = sum(1 for flag in flags if flag)
        return {"correct": correct, "total": total, "exact": (correct / total) if total else None}

    fields = [field for doc in scored for field in doc["fields"]]
    by_kind_span: dict[str, list[bool]] = {}
    by_kind_value: dict[str, list[bool]] = {}
    by_op_value: dict[str, list[bool]] = {}
    for field in fields:
        by_kind_span.setdefault(field["kind"], []).append(field["span_exact"])
        by_kind_value.setdefault(field["kind"], []).append(field["value_exact"])
        by_op_value.setdefault(field["op"], []).append(field["value_exact"])
    present = [
        question["token_exact"]
        for doc in scored
        for field in doc["fields"]
        for question in field["questions"]
        if not question["gold_missing"]
    ]
    missing = [
        question["missing"] == question["gold_missing"]
        for doc in scored
        for field in doc["fields"]
        for question in field["questions"]
        if question["gold_missing"]
    ]
    return {
        "documents_span": rate([doc["document_span"] for doc in scored]),
        "documents_value": rate([doc["document_value"] for doc in scored]),
        "fields_span": rate([field["span_exact"] for field in fields]),
        "fields_value": rate([field["value_exact"] for field in fields]),
        "span_token_exact": rate(present),
        "missing_exact": rate(missing),
        "by_kind_span": {kind: rate(flags) for kind, flags in sorted(by_kind_span.items())},
        "by_kind_value": {kind: rate(flags) for kind, flags in sorted(by_kind_value.items())},
        "by_op_value": {op: rate(flags) for op, flags in sorted(by_op_value.items())},
        "skipped_forecast_fields": sum(len(doc.get("skipped", [])) for doc in scored),
    }


def dumps_public(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
