"""Small readout heads on a frozen Gemma last layer.

Point picks a start token, an end token, and a missing score.
Number writes a sign and digits.
Calendar picks a legal year, month, day, hour, and minute.
Spread predicts a mean and a width inside the caller's min/max.
"""

from __future__ import annotations

import torch
from torch import nn

from extractor.labels import INT_DIGITS, YEAR_SPAN


class PointHead(nn.Module):
    def __init__(self, width: int, inner: int = 64) -> None:
        super().__init__()
        self.doc_norm = nn.LayerNorm(width)
        self.query_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, inner)
        self.start_key = nn.Linear(width, inner)
        self.end_key = nn.Linear(width, inner)
        self.missing_question = nn.Linear(width, 1)
        self.missing_match = nn.Linear(2, 1)
        self.score_scale = inner ** -0.5

    def forward(self, document: torch.Tensor, question: torch.Tensor) -> dict[str, torch.Tensor]:
        document = self.doc_norm(document)
        question = self.query_norm(question)
        query = self.query(question)
        start = torch.einsum("bh,bth->bt", query, self.start_key(document)) * self.score_scale
        end = torch.einsum("bh,bth->bt", query, self.end_key(document)) * self.score_scale
        best = torch.stack([start.amax(dim=-1), end.amax(dim=-1)], dim=-1)
        missing = self.missing_question(question) + self.missing_match(best)
        return {"start": start, "end": end, "missing": missing.squeeze(-1)}


def choose_span(start: torch.Tensor, end: torch.Tensor, limit: int) -> tuple[int, int]:
    """Pick the start and end that score best together, no longer than `limit` tokens.

    Independent argmax can put the end before the start, or on a token far past
    the longest span this operation had in training. The score of a span is the
    start score plus the end score. Ties keep the earlier start, then the earlier end.
    """
    if start.ndim != 1 or end.shape != start.shape:
        raise ValueError(f"start and end must be one document, got {tuple(start.shape)} and {tuple(end.shape)}")
    if limit < 1:
        raise ValueError(f"span limit must be at least 1, got {limit}")
    count = int(start.numel())
    if count == 0:
        raise ValueError("document has no tokens")
    scores = start[:, None] + end[None, :]
    index = torch.arange(count)
    length = index[None, :] - index[:, None] + 1
    allowed = (length >= 1) & (length <= limit)
    scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    chosen = int(scores.argmax())
    start_at, end_at = divmod(chosen, count)
    return start_at, end_at


def choose_spans(
    starts: list[torch.Tensor],
    ends: list[torch.Tensor],
    limit: int,
) -> list[tuple[int, int]]:
    """Place several spans in order. Each later span starts after the previous end.

    The score is the sum of each span's start score plus end score. Each span is
    at most `limit` tokens. Ties keep the earlier start, then the earlier end.
    A single span matches choose_span. If the document cannot hold the chain,
    each span falls back to choose_span on its own.
    """
    if not starts or len(starts) != len(ends):
        raise ValueError("starts and ends must be the same non-empty list")
    count = int(starts[0].numel())
    for start, end in zip(starts, ends):
        if start.shape != (count,) or end.shape != (count,):
            raise ValueError(f"span logits must have shape ({count},)")
    if limit < 1:
        raise ValueError(f"span limit must be at least 1, got {limit}")
    if count == 0:
        raise ValueError("document has no tokens")
    if count < len(starts):
        return [choose_span(start, end, limit) for start, end in zip(starts, ends)]

    neg = torch.finfo(torch.float32).min / 8
    index = torch.arange(count)
    length = index[None, :] - index[:, None] + 1
    allowed = (length >= 1) & (length <= limit)
    layers = []
    previous = None
    for start, end in zip(starts, ends):
        if previous is None:
            best_before = torch.zeros(count, dtype=torch.float32)
            prev_for_start = None
        else:
            best_before = _scores_ending_before(previous, neg)
            prev_for_start = _earliest_end_before(previous)
        scores = best_before[:, None] + start.float()[:, None] + end.float()[None, :]
        scores = scores.masked_fill(~allowed, neg)
        layers.append(
            {
                "start_at": scores.argmax(dim=0),
                "best": scores.max(dim=0).values,
                "prev_for_start": prev_for_start,
            }
        )
        previous = layers[-1]["best"]
    if float(previous.max()) <= float(neg) / 2:
        return [choose_span(start, end, limit) for start, end in zip(starts, ends)]

    end_at = int(previous.argmax())
    chosen: list[tuple[int, int]] = []
    for layer in reversed(layers):
        start_at = int(layer["start_at"][end_at])
        chosen.append((start_at, end_at))
        if layer["prev_for_start"] is None:
            break
        end_at = int(layer["prev_for_start"][start_at])
    chosen.reverse()
    if len(chosen) != len(starts):
        raise RuntimeError("span chain did not place every part")
    return chosen


def _scores_ending_before(best: torch.Tensor, neg: float) -> torch.Tensor:
    """best[t] is a chain score ending at t. Result[a] is the best score ending before a."""
    out = torch.empty_like(best)
    out[0] = neg
    if best.numel() > 1:
        out[1:] = torch.cummax(best, dim=0).values[:-1]
    return out


def _earliest_end_before(best: torch.Tensor) -> torch.Tensor:
    """Result[a] is the earliest end whose score is the max of best[:a]."""
    count = int(best.numel())
    result = torch.zeros(count, dtype=torch.long)
    values = best.tolist()
    running_idx = 0
    running_max = values[0]
    for position in range(1, count):
        result[position] = running_idx
        if values[position] > running_max:
            running_max = values[position]
            running_idx = position
    return result


class DocumentReader(nn.Module):
    """Pool document tokens with weights that depend on the question.

    A date or an amount lives in the document. The question only says which
    field is being asked. Adding this pool to the question vector lets the
    number, calendar, and spread heads see those tokens.
    """

    def __init__(self, width: int, inner: int = 64) -> None:
        super().__init__()
        self.doc_norm = nn.LayerNorm(width)
        self.query_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, inner)
        self.key = nn.Linear(width, inner)
        self.value = nn.Linear(width, width)
        self.scale = inner ** -0.5

    def forward(self, document: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        document = self.doc_norm(document)
        question = self.query_norm(question)
        scores = torch.einsum("bh,bth->bt", self.query(question), self.key(document)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bt,bth->bh", weights, self.value(document))


class NumberHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.reader = DocumentReader(width)
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 2 + INT_DIGITS * 10 + 20))

    def forward(self, document: torch.Tensor, question: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.net(self.norm(question + self.reader(document, question)))
        sign = raw[:, :2]
        cursor = 2
        whole = raw[:, cursor:cursor + INT_DIGITS * 10].view(-1, INT_DIGITS, 10)
        cursor += INT_DIGITS * 10
        frac = raw[:, cursor:].view(-1, 2, 10)
        return {"sign": sign, "whole": whole, "frac": frac}


class CalendarHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.reader = DocumentReader(width)
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, YEAR_SPAN + 12 + 31 + 24 + 60))

    def forward(self, document: torch.Tensor, question: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.net(self.norm(question + self.reader(document, question)))
        cuts = [YEAR_SPAN, 12, 31, 24, 60]
        pieces = {}
        cursor = 0
        for name, size in zip(("year", "month", "day", "hour", "minute"), cuts):
            pieces[name] = raw[:, cursor:cursor + size]
            cursor += size
        return pieces


class SpreadHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.reader = DocumentReader(width)
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, document: torch.Tensor, question: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.net(self.norm(question + self.reader(document, question)))
        mean = torch.sigmoid(raw[:, 0])
        width_score = torch.nn.functional.softplus(raw[:, 1]) + 1e-3
        return {"mean": mean, "width": width_score}


def point_loss(outputs: dict[str, torch.Tensor], missing: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs["missing"], missing.float())
    present = missing == 0
    if present.any():
        loss = loss + torch.nn.functional.cross_entropy(outputs["start"][present], start[present])
        loss = loss + torch.nn.functional.cross_entropy(outputs["end"][present], end[present])
    return loss


def number_loss(outputs: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> torch.Tensor:
    loss = torch.nn.functional.cross_entropy(outputs["sign"], target["sign"])
    loss = loss + torch.nn.functional.cross_entropy(outputs["whole"].transpose(1, 2), target["whole"])
    loss = loss + torch.nn.functional.cross_entropy(outputs["frac"].transpose(1, 2), target["frac"])
    return loss


def calendar_loss(outputs: dict[str, torch.Tensor], target: dict[str, torch.Tensor | None], kind: str) -> torch.Tensor:
    loss = outputs["year"].sum() * 0
    if kind in {"date", "datetime"}:
        loss = loss + torch.nn.functional.cross_entropy(outputs["year"], target["year"])
        loss = loss + torch.nn.functional.cross_entropy(outputs["month"], target["month"])
        loss = loss + torch.nn.functional.cross_entropy(outputs["day"], target["day"])
    if kind in {"time", "datetime"}:
        loss = loss + torch.nn.functional.cross_entropy(outputs["hour"], target["hour"])
        loss = loss + torch.nn.functional.cross_entropy(outputs["minute"], target["minute"])
    return loss


def spread_loss(outputs: dict[str, torch.Tensor], norm: torch.Tensor) -> torch.Tensor:
    mean = outputs["mean"]
    var = outputs["width"] ** 2
    return (0.5 * (torch.log(var) + (norm - mean) ** 2 / var)).mean()
