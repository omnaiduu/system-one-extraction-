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
        self.missing = nn.Linear(width, 1)
        self.score_scale = inner ** -0.5

    def forward(self, document: torch.Tensor, question: torch.Tensor) -> dict[str, torch.Tensor]:
        document = self.doc_norm(document)
        question = self.query_norm(question)
        query = self.query(question)
        start = torch.einsum("bh,bth->bt", query, self.start_key(document)) * self.score_scale
        end = torch.einsum("bh,bth->bt", query, self.end_key(document)) * self.score_scale
        return {"start": start, "end": end, "missing": self.missing(question).squeeze(-1)}


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
