"""Turn an audited document into one training question per head target.

List items become separate point questions, plus one extra question whose
correct answer is "missing" so the head learns when to stop. Forecast steps
become separate spread questions. Character offsets stay in document
coordinates. The document is always the prefix of the packed prompt, so those
offsets are also prompt offsets.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

INT_DIGITS = 7
YEAR_BASE = 2018
YEAR_SPAN = 16


def quantize(value: float, decimals: int) -> Decimal:
    number = Decimal(str(value))
    step = Decimal("1").scaleb(-decimals)
    return number.quantize(step, rounding=ROUND_HALF_UP)


def split_number(value: float, decimals: int) -> dict:
    quantized = quantize(value, decimals)
    negative = quantized < 0
    absolute = abs(quantized)
    if decimals == 0:
        whole = str(int(absolute))
        frac = "00"
    else:
        rendered = f"{absolute:.{decimals}f}"
        whole, frac = rendered.split(".")
        frac = (frac + "00")[:2]
    if len(whole) > INT_DIGITS:
        raise ValueError(f"number needs more than {INT_DIGITS} digits: {whole}")
    return {
        "sign": 1 if negative else 0,
        "whole": [int(char) for char in whole.zfill(INT_DIGITS)],
        "frac": [int(char) for char in frac],
        "decimals": decimals,
    }


def join_number(encoded: dict) -> Decimal:
    whole = "".join(str(digit) for digit in encoded["whole"]).lstrip("0") or "0"
    frac = "".join(str(digit) for digit in encoded["frac"])
    decimals = encoded["decimals"]
    text = whole if decimals == 0 else f"{whole}.{frac[:decimals]}"
    number = Decimal(text)
    return -number if encoded["sign"] else number


def pack_prompt(text: str, questions: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    """Return the full prompt and the character span of each question body."""
    pieces = [text]
    spans: list[tuple[int, int]] = []
    for question in questions:
        body = f"{question['field']}: {question['instruction']}"
        pieces.append("\n\nQUESTION\n")
        pieces.append(body)
        end = sum(len(piece) for piece in pieces)
        spans.append((end - len(body), end))
    prompt = "".join(pieces)
    for question, (start, end) in zip(questions, spans):
        expected = f"{question['field']}: {question['instruction']}"
        if prompt[start:end] != expected:
            raise RuntimeError("question span drifted")
    return prompt, spans


def expand_document(row: dict) -> list[dict]:
    questions = []
    for name, field in row["fields"].items():
        kind = field["type"]
        instruction = field["instruction"]
        if kind == "list":
            spans = field.get("spans") or []
            for index, span in enumerate(spans, start=1):
                questions.append(_point(name, f"{instruction} Item {index}.", False, span[0], span[1], field["value"][index - 1], "list"))
            questions.append(_point(name, f"{instruction} Item {len(spans) + 1}.", True, None, None, None, "list"))
        elif kind == "forecast":
            for index, value in enumerate(field["value"], start=1):
                questions.append(_spread(name, f"{instruction} Step {index}.", field, value))
        elif kind == "text":
            questions.append(_point(name, instruction, field.get("missing", False), field.get("start"), field.get("end"), field.get("value")))
        elif kind in {"number", "duration"}:
            decimals = int(field.get("decimals", 0 if kind == "duration" else 2))
            questions.append({
                "field": name,
                "instruction": instruction,
                "head": "number",
                "kind": kind,
                "target": split_number(field["value"], decimals),
            })
        elif kind in {"date", "time", "datetime"}:
            questions.append({
                "field": name,
                "instruction": instruction,
                "head": "calendar",
                "kind": kind,
                "target": calendar_target(kind, field["value"]),
            })
        elif kind == "estimate":
            questions.append(_spread(name, instruction, field, field["value"]))
        else:
            raise ValueError(f"unknown field type {kind}")
    return questions


def _point(name: str, instruction: str, missing: bool, start, end, value, kind: str = "text") -> dict:
    return {
        "field": name,
        "instruction": instruction,
        "head": "point",
        "kind": kind,
        "missing": missing,
        "start": start,
        "end": end,
        "value": value,
    }


def _spread(name: str, instruction: str, field: dict, value: float) -> dict:
    low = float(field["min"])
    high = float(field["max"])
    if high <= low:
        raise ValueError(f"bad range for {name}")
    return {
        "field": name,
        "instruction": instruction,
        "head": "spread",
        "kind": field["type"],
        "value": float(value),
        "min": low,
        "max": high,
        "step": float(field["step"]),
        "norm": (float(value) - low) / (high - low),
    }


def calendar_target(kind: str, value: str) -> dict:
    target = {"kind": kind, "year": None, "month": None, "day": None, "hour": None, "minute": None}
    if kind == "time":
        hour, minute = value.split(":")
        target["hour"] = int(hour)
        target["minute"] = int(minute)
        return target
    if "T" in value:
        day, clock = value.split("T")
    else:
        day, clock = value, None
    year, month, day_number = (int(part) for part in day.split("-"))
    if not YEAR_BASE <= year < YEAR_BASE + YEAR_SPAN:
        raise ValueError(f"year {year} is outside {YEAR_BASE}-{YEAR_BASE + YEAR_SPAN - 1}")
    target["year"] = year - YEAR_BASE
    target["month"] = month - 1
    target["day"] = day_number - 1
    if clock:
        hour, minute = clock.split(":")
        target["hour"] = int(hour)
        target["minute"] = int(minute)
    return target


def document_token_indexes(offsets: list[tuple[int, int]], doc_chars: int) -> list[int]:
    chosen = []
    for index, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        if end <= doc_chars:
            chosen.append(index)
    return chosen


def question_token_indexes(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    chosen = []
    for index, (token_start, token_end) in enumerate(offsets):
        if token_end <= token_start:
            continue
        if token_start >= start and token_end <= end:
            chosen.append(index)
    return chosen


def char_span_to_tokens(offsets: list[tuple[int, int]], start: int, end: int, allowed: list[int]) -> tuple[int, int]:
    """Map a document character span onto indexes inside `allowed`."""
    start_at = None
    end_at = None
    for position, token_index in enumerate(allowed):
        token_start, token_end = offsets[token_index]
        if start_at is None and token_start <= start < token_end:
            start_at = position
        if token_start < end <= token_end or token_start <= end - 1 < token_end:
            end_at = position
    if start_at is None or end_at is None or end_at < start_at:
        raise ValueError(f"span {start}:{end} does not land on a token")
    return start_at, end_at
