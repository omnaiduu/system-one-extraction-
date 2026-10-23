"""Turn marked documents into extraction rows.

A writer marks evidence that is literally in the text. This auditor removes the
marks, checks each quote, and only then creates labels. Sums, day counts, date
shifts, medians, and forecast tails are calculated here. A row is dropped when
the evidence is missing, the dates do not parse, or a calculated answer was
already written in the text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

MARK_RE = re.compile(r"\[\[([a-z][a-z0-9_]*)\]\](.*?)\[\[/\]\]", re.DOTALL)
OBS_LINE_RE = re.compile(r"^\s*\[\[obs\]\](.*?)\[\[/\]\]\s*$")

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}


@dataclass
class Mark:
    name: str
    inner: str
    start: int
    end: int


class Drop(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def parse_number(raw: str) -> Decimal:
    text = raw.strip().replace(",", "")
    text = re.sub(r"^[^\d\-.]+", "", text)
    text = re.sub(r"[^\d.\-]+$", "", text)
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        raise Drop(f"unparsable_number:{raw}")
    return Decimal(text)


def quantize(value: Decimal, decimals: int) -> Decimal:
    step = Decimal("1").scaleb(-decimals)
    return value.quantize(step, rounding=ROUND_HALF_UP)


def format_number(value: Decimal, decimals: int) -> str:
    return f"{quantize(value, decimals):.{decimals}f}"


def _month_date(day: int, month_name: str, year: int) -> date:
    month = MONTHS.get(month_name.lower())
    if month is None:
        raise Drop(f"unparsable_date:{day} {month_name} {year}")
    try:
        return date(year, month, day)
    except ValueError as exc:
        raise Drop(f"unparsable_date:{day} {month_name} {year}") from exc


def parse_date(raw: str, dayfirst: bool = True) -> date:
    text = " ".join(raw.strip().split())
    iso = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if iso:
        return _safe_date(*map(int, iso.groups()), raw)
    ymd = re.fullmatch(r"(\d{4})/(\d{2})/(\d{2})", text)
    if ymd:
        return _safe_date(*map(int, ymd.groups()), raw)
    named = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", text)
    if named:
        return _month_date(int(named.group(1)), named.group(2), int(named.group(3)))
    named_us = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})", text)
    if named_us:
        return _month_date(int(named_us.group(2)), named_us.group(1), int(named_us.group(3)))
    slash = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if slash:
        a, b, year = int(slash.group(1)), int(slash.group(2)), int(slash.group(3))
        day, month = (a, b) if dayfirst else (b, a)
        return _safe_date(year, month, day, raw)
    raise Drop(f"unparsable_date:{raw}")


def _safe_date(year: int, month: int, day: int, raw: str) -> date:
    try:
        return date(year, month, day)
    except ValueError as exc:
        raise Drop(f"unparsable_date:{raw}") from exc


def parse_time(raw: str) -> time:
    text = raw.strip().lower().replace(".", "")
    clock = re.fullmatch(r"(\d{1,2}):(\d{2})\s*([ap]m)?", text)
    if not clock:
        raise Drop(f"unparsable_time:{raw}")
    hour, minute = int(clock.group(1)), int(clock.group(2))
    suffix = clock.group(3)
    if suffix == "pm" and hour < 12:
        hour += 12
    if suffix == "am" and hour == 12:
        hour = 0
    if suffix is None and not 0 <= hour <= 23:
        raise Drop(f"unparsable_time:{raw}")
    if suffix is not None and not 1 <= int(clock.group(1)) <= 12:
        raise Drop(f"unparsable_time:{raw}")
    try:
        return time(hour, minute)
    except ValueError as exc:
        raise Drop(f"unparsable_time:{raw}") from exc


def parse_datetime(raw: str, dayfirst: bool = True) -> datetime:
    text = raw.strip()
    if "T" in text:
        day_part, time_part = text.split("T", 1)
    elif " " in text:
        day_part, time_part = text.rsplit(" ", 1)
        if ":" not in time_part:
            # "March 3, 2026 2:30 pm" has a comma date and a clock at the end.
            match = re.fullmatch(r"(.+?)\s+(\d{1,2}:\d{2}(?:\s*[ap]m)?)\s*", text, re.I)
            if not match:
                raise Drop(f"unparsable_datetime:{raw}")
            day_part, time_part = match.group(1), match.group(2)
    else:
        raise Drop(f"unparsable_datetime:{raw}")
    return datetime.combine(parse_date(day_part, dayfirst), parse_time(time_part))


def number_present(text: str, value: Decimal, decimals: int) -> bool:
    rendered = format_number(value, decimals)
    plain = rendered[:-2] if decimals and rendered.endswith(".00") else rendered.rstrip("0").rstrip(".") if "." in rendered else rendered
    candidates = {rendered, plain}
    for candidate in candidates:
        if not candidate:
            continue
        if re.search(rf"(?<![\d.]){re.escape(candidate)}(?![\d.])", text):
            return True
    return False


def strip_forecast_tail(marked: str, horizon: int) -> tuple[str, list[dict]]:
    lines = marked.split("\n")
    obs_indexes = [i for i, line in enumerate(lines) if OBS_LINE_RE.match(line)]
    if len(obs_indexes) <= horizon:
        raise Drop("forecast_history_too_short")
    hidden_at = set(obs_indexes[-horizon:])
    future = []
    kept = []
    for i, line in enumerate(lines):
        if i in hidden_at:
            inner = OBS_LINE_RE.match(line).group(1).strip()
            if "|" not in inner:
                raise Drop("bad_obs")
            when, value = inner.split("|", 1)
            future.append({"date": parse_date(when).isoformat(), "value": parse_number(value)})
            continue
        if OBS_LINE_RE.match(line) is None and "[[obs]]" in line:
            raise Drop("obs_not_on_own_line")
        kept.append(line)
    dates = [item["date"] for item in future]
    if dates != sorted(dates):
        raise Drop("forecast_dates_not_increasing")
    return "\n".join(kept), future


def find_marks(marked: str) -> tuple[str, dict[str, list[Mark]]]:
    if "[[[" in marked:
        raise Drop("broken_mark")
    pieces: list[str] = []
    grouped: dict[str, list[Mark]] = {}
    cursor = 0
    last = 0
    matches = list(MARK_RE.finditer(marked))
    if marked.count("[[/") != len(matches):
        raise Drop("broken_mark")
    for match in matches:
        prefix = marked[last:match.start()]
        if "[[" in prefix:
            raise Drop("broken_mark")
        pieces.append(prefix)
        cursor += len(prefix)
        name = match.group(1)
        inner = match.group(2)
        if "[[" in inner or not inner.strip():
            raise Drop("broken_mark")
        mark = Mark(name, inner, cursor, cursor + len(inner))
        grouped.setdefault(name, []).append(mark)
        pieces.append(inner)
        cursor += len(inner)
        last = match.end()
    tail = marked[last:]
    if "[[" in tail:
        raise Drop("broken_mark")
    pieces.append(tail)
    text = "".join(pieces)
    for group in grouped.values():
        for mark in group:
            if text[mark.start:mark.end] != mark.inner:
                raise Drop("span_mismatch")
    return text, grouped


def _need_marks(grouped: dict[str, list[Mark]], name: str, exact: int | None = None) -> list[Mark]:
    found = grouped.get(name, [])
    if not found:
        raise Drop(f"missing_mark:{name}")
    if exact is not None and len(found) != exact:
        raise Drop(f"mark_count:{name}")
    return found


def _field_base(spec: dict) -> dict:
    return {
        "type": spec["type"],
        "instruction": spec["instruction"],
    }


def label_document(obj: dict) -> dict:
    marked = obj["marked_text"]
    ask = obj["ask"]
    if not isinstance(ask, list) or not ask:
        raise Drop("empty_ask")
    names = [item["name"] for item in ask]
    if len(names) != len(set(names)):
        raise Drop("duplicate_field")

    forecast = [item for item in ask if item["type"] == "forecast"]
    if len(forecast) > 1:
        raise Drop("one_forecast_only")
    future = []
    if forecast:
        horizon = int(forecast[0].get("horizon", 2))
        if not 1 <= horizon <= 8:
            raise Drop("bad_horizon")
        marked, future = strip_forecast_tail(marked, horizon)
        if len(future) != horizon:
            raise Drop("bad_horizon")

    text, grouped = find_marks(marked)
    if not text.strip():
        raise Drop("empty_text")
    for banned in obj.get("banned", []):
        if banned and banned in text:
            raise Drop(f"banned_text:{banned}")

    fields = {}
    for spec in ask:
        kind = spec["type"]
        base = _field_base(spec)
        if kind == "text":
            fields[spec["name"]] = _label_text(text, grouped, spec, base)
        elif kind == "list":
            fields[spec["name"]] = _label_list(text, grouped, spec, base)
        elif kind == "number":
            fields[spec["name"]] = _label_number(text, grouped, spec, base)
        elif kind == "duration":
            fields[spec["name"]] = _label_duration(text, grouped, spec, base)
        elif kind in {"date", "time", "datetime"}:
            fields[spec["name"]] = _label_calendar(text, grouped, spec, base)
        elif kind == "estimate":
            fields[spec["name"]] = _label_estimate(text, grouped, spec, base)
        elif kind == "forecast":
            fields[spec["name"]] = _label_forecast(text, spec, base, future)
        else:
            raise Drop(f"bad_type:{kind}")
    return {"id": obj["id"], "genre": obj.get("genre", ""), "text": text, "fields": fields}


def _label_text(text: str, grouped: dict[str, list[Mark]], spec: dict, base: dict) -> dict:
    mark_name = spec.get("mark")
    if mark_name in (None, ""):
        for banned in spec.get("must_not_contain", []):
            if banned in text:
                raise Drop(f"null_leaked:{spec['name']}")
        return {**base, "value": None, "missing": True}
    found = _need_marks(grouped, mark_name, exact=1)
    mark = found[0]
    return {
        **base,
        "value": mark.inner,
        "missing": False,
        "start": mark.start,
        "end": mark.end,
    }


def _label_list(text: str, grouped: dict[str, list[Mark]], spec: dict, base: dict) -> dict:
    del text
    found = _need_marks(grouped, spec["mark"])
    return {
        **base,
        "items": spec.get("items", "text"),
        "missing": False,
        "value": [mark.inner for mark in found],
        "spans": [[mark.start, mark.end] for mark in found],
    }


def _marked_number(grouped: dict[str, list[Mark]], name: str) -> tuple[Mark, Decimal]:
    found = _need_marks(grouped, name, exact=1)
    return found[0], parse_number(found[0].inner)


def _label_number(text: str, grouped: dict[str, list[Mark]], spec: dict, base: dict) -> dict:
    decimals = int(spec.get("decimals", 2))
    compute = spec.get("compute")
    if compute is None:
        mark, value = _marked_number(grouped, spec["mark"])
        value = quantize(value, decimals)
        return {
            **base,
            "decimals": decimals,
            "missing": False,
            "value": float(value),
            "start": mark.start,
            "end": mark.end,
            "evidence": mark.inner,
        }
    if compute == "sum":
        parts = _need_marks(grouped, spec["parts"])
        if len(parts) < 2:
            raise Drop("need_two_parts")
        value = sum((parse_number(mark.inner) for mark in parts), Decimal("0"))
        evidence = [_evidence(mark) for mark in parts]
    elif compute == "product":
        left_mark, left = _marked_number(grouped, spec["left"])
        right_mark, right = _marked_number(grouped, spec["right"])
        value = left * right
        evidence = [_evidence(left_mark), _evidence(right_mark)]
    elif compute == "subtract":
        left_mark, left = _marked_number(grouped, spec["left"])
        right_mark, right = _marked_number(grouped, spec["right"])
        value = left - right
        evidence = [_evidence(left_mark), _evidence(right_mark)]
    else:
        raise Drop(f"bad_compute:{compute}")
    value = quantize(value, decimals)
    if number_present(text, value, decimals):
        raise Drop(f"answer_leaked:{spec['name']}")
    return {
        **base,
        "decimals": decimals,
        "compute": compute,
        "missing": False,
        "value": float(value),
        "evidence": evidence,
    }


def _evidence(mark: Mark) -> dict:
    return {"start": mark.start, "end": mark.end, "text": mark.inner}


def _label_duration(text: str, grouped: dict[str, list[Mark]], spec: dict, base: dict) -> dict:
    if spec.get("unit", "days") != "days":
        raise Drop("only_days")
    if spec.get("compute") != "days_between":
        raise Drop("bad_compute:duration")
    start = parse_date(_need_marks(grouped, spec["start"], exact=1)[0].inner, spec.get("dayfirst", True))
    end = parse_date(_need_marks(grouped, spec["end"], exact=1)[0].inner, spec.get("dayfirst", True))
    days = (end - start).days
    if days < 0:
        raise Drop("negative_duration")
    if number_present(text, Decimal(days), 0):
        raise Drop(f"answer_leaked:{spec['name']}")
    return {**base, "unit": "days", "missing": False, "value": days}


def _label_calendar(text: str, grouped: dict[str, list[Mark]], spec: dict, base: dict) -> dict:
    dayfirst = spec.get("dayfirst", True)
    compute = spec.get("compute")
    if compute is None:
        mark = _need_marks(grouped, spec["mark"], exact=1)[0]
        value = _calendar_value(spec["type"], mark.inner, dayfirst)
        return {**base, "missing": False, "value": value, "start": mark.start, "end": mark.end}
    if compute != "add_days":
        raise Drop(f"bad_compute:{compute}")
    start = parse_date(_need_marks(grouped, spec["start"], exact=1)[0].inner, dayfirst)
    days = int(parse_number(_need_marks(grouped, spec["days"], exact=1)[0].inner))
    value = (start + timedelta(days=days)).isoformat()
    if value in text:
        raise Drop(f"answer_leaked:{spec['name']}")
    return {**base, "compute": "add_days", "missing": False, "value": value}


def _calendar_value(kind: str, raw: str, dayfirst: bool) -> str:
    if kind == "date":
        return parse_date(raw, dayfirst).isoformat()
    if kind == "time":
        return parse_time(raw).strftime("%H:%M")
    if kind == "datetime":
        return parse_datetime(raw, dayfirst).strftime("%Y-%m-%dT%H:%M")
    raise Drop(f"bad_type:{kind}")


def _label_estimate(text: str, grouped: dict[str, list[Mark]], spec: dict, base: dict) -> dict:
    comps = _need_marks(grouped, spec.get("mark", "comp"))
    if len(comps) < 4 or len(comps) % 2 != 0:
        raise Drop("estimate_needs_even_comps")
    values = [parse_number(mark.inner) for mark in comps]
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = quantize((ordered[mid - 1] + ordered[mid]) / Decimal(2), int(spec.get("decimals", 0)))
    lo = Decimal(str(spec["min"]))
    hi = Decimal(str(spec["max"]))
    step = Decimal(str(spec["step"]))
    if not lo < median < hi or step <= 0:
        raise Drop("estimate_out_of_bounds")
    if number_present(text, median, int(spec.get("decimals", 0))):
        raise Drop(f"answer_leaked:{spec['name']}")
    return {
        **base,
        "missing": False,
        "value": float(median),
        "min": float(lo),
        "max": float(hi),
        "step": float(step),
        "evidence": [_evidence(mark) for mark in comps],
    }


def _label_forecast(text: str, spec: dict, base: dict, future: list[dict]) -> dict:
    lo = Decimal(str(spec["min"]))
    hi = Decimal(str(spec["max"]))
    for item in future:
        if not lo <= item["value"] <= hi:
            raise Drop("forecast_out_of_bounds")
        if number_present(text, item["value"], int(spec.get("decimals", 0))):
            raise Drop(f"answer_leaked:{spec['name']}")
    return {
        **base,
        "missing": False,
        "horizon": len(future),
        "min": float(lo),
        "max": float(hi),
        "step": float(spec["step"]),
        "value": [float(item["value"]) for item in future],
        "dates": [item["date"] for item in future],
    }


def audit_document(obj: dict) -> dict:
    try:
        return label_document(obj)
    except Drop:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise Drop(f"bad_record:{exc}") from exc


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            rows.append({"id": f"{path.name}:{line_no}", "genre": "", "_error": f"bad_json:{exc}"})
    return rows


def audit_paths(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    kept: list[dict] = []
    dropped: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        for obj in load_jsonl(path):
            doc_id = str(obj.get("id", path.name))
            if obj.get("_error"):
                dropped.append({"id": doc_id, "reason": obj["_error"]})
                continue
            if doc_id in seen:
                dropped.append({"id": doc_id, "reason": "duplicate_id"})
                continue
            seen.add(doc_id)
            try:
                kept.append(audit_document(obj))
            except Drop as exc:
                dropped.append({"id": doc_id, "reason": exc.reason})
    kept.sort(key=lambda row: row["id"])
    return kept, dropped


def split_rows(rows: list[dict]) -> dict[str, list[dict]]:
    train, val, test = [], [], []
    for index, row in enumerate(rows):
        slot = index % 10
        if slot == 9:
            test.append(row)
        elif slot == 8:
            val.append(row)
        else:
            train.append(row)
    return {"train": train, "val": val, "test": test}


def field_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for field in row["fields"].values():
            key = field["type"] if not field.get("missing") else f"{field['type']}:missing"
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def write_split(raw_dir: Path, out_dir: Path) -> dict:
    paths = sorted(raw_dir.glob("*.jsonl"))
    kept, dropped = audit_paths(paths)
    parts = split_rows(kept)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in parts.items():
        destination = out_dir / f"{name}.jsonl"
        destination.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    report = {
        "kept": len(kept),
        "dropped": dropped,
        "split": {name: len(rows) for name, rows in parts.items()},
        "fields": field_counts(kept),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = write_split(root / "data" / "raw", root / "data" / "audited")
    print(json.dumps({"kept": report["kept"], "split": report["split"], "fields": report["fields"], "dropped": len(report["dropped"])}, indent=2))


if __name__ == "__main__":
    main()
