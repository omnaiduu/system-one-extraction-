from decimal import Decimal

import torch

from extractor.heads import CalendarHead, NumberHead, PointHead, SpreadHead
from extractor.score_heads import calendar_string, question_correct, score_document, snap_to_step


def test_snap_uses_half_up_on_the_callers_grid():
    assert snap_to_step(10.5, 0, 1) == Decimal("11")
    assert snap_to_step(25375, 0, 1) == Decimal("25375")


def test_calendar_string_builds_iso_values():
    assert calendar_string("date", {"year": 8, "month": 3, "day": 18, "hour": 0, "minute": 0}) == "2026-04-19"
    assert calendar_string("time", {"hour": 14, "minute": 30, "year": 0, "month": 0, "day": 0}) == "14:30"
    assert calendar_string("datetime", {"year": 8, "month": 3, "day": 1, "hour": 14, "minute": 30}) == "2026-04-02T14:30"


def test_point_token_match_can_differ_from_the_character_offsets():
    question = {
        "head": "point",
        "kind": "text",
        "missing": False,
        "start": 1,
        "end": 1,
        "gold_start": 5,
        "gold_end": 12,
        "value": "HT-3391",
    }
    offsets = [(0, 4), (4, 12)]
    judged = question_correct(question, {"missing": False, "start": 1, "end": 1}, offsets, "Bill HT-3391")
    assert judged["correct"] is True
    assert judged["char_exact"] is False


def test_a_list_field_needs_every_item_and_the_stop():
    width = 8
    heads = {
        "point": PointHead(width),
        "number": NumberHead(width),
        "calendar": CalendarHead(width),
        "spread": SpreadHead(width),
    }
    heads["point"].missing.weight.data.zero_()
    heads["point"].missing.bias.data.fill_(-20)
    record = {
        "id": "demo",
        "split": "val",
        "text": "jar lid",
        "doc_states": torch.zeros(2, width),
        "doc_offsets": [(0, 3), (4, 7)],
        "questions": [
            {
                "field": "items",
                "head": "point",
                "kind": "list",
                "missing": True,
                "start": 0,
                "end": 0,
                "value": None,
                "gold_start": None,
                "gold_end": None,
                "q_state": torch.zeros(width),
            },
            {
                "field": "items",
                "head": "point",
                "kind": "list",
                "missing": False,
                "start": 0,
                "end": 0,
                "value": "jar",
                "gold_start": 0,
                "gold_end": 3,
                "q_state": torch.zeros(width),
            },
        ],
    }
    scored = score_document(heads, record, 0.5)
    assert len(scored["fields"]) == 1
    assert scored["fields"][0]["correct"] is False
