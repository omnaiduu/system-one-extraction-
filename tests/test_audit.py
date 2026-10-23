import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from extractor.audit import Drop, audit_document, number_present, parse_date, write_split


def test_parse_date_formats():
    assert parse_date("2026-04-02") == date(2026, 4, 2)
    assert parse_date("2 Apr 2026") == date(2026, 4, 2)
    assert parse_date("2 April 2026") == date(2026, 4, 2)
    assert parse_date("April 2, 2026") == date(2026, 4, 2)
    assert parse_date("02/04/2026", dayfirst=True) == date(2026, 4, 2)
    assert parse_date("04/02/2026", dayfirst=False) == date(2026, 4, 2)
    assert parse_date("2 Sept 2026") == date(2026, 9, 2)


def test_number_leak_uses_token_boundaries():
    assert number_present("old ref 12 only", Decimal("12"), 0)
    assert not number_present("room 120 stays", Decimal("12"), 0)


def test_support_message_labels_are_found_by_code():
    row = audit_document(
        {
            "id": "support-demo",
            "genre": "whatsapp",
            "banned": ["AWB"],
            "marked_text": (
                "Riya, blender [[order_id]]BL-441[[/]] came on [[delivered_on]]2 Apr 2026[[/]]. "
                "[[damage]]jar cracked[[/]] and [[damage]]lid missing[[/]]. "
                "I sent [[part]]1899[[/]] plus [[part]]342[[/]] by UPI. "
                "Writing this [[written_on]]11 Apr 2026[[/]]. Old ref BX-12 was another order."
            ),
            "ask": [
                {"name": "order_id", "type": "text", "instruction": "Order id", "mark": "order_id"},
                {"name": "tracking_number", "type": "text", "instruction": "Tracking number", "mark": None, "must_not_contain": ["AWB"]},
                {"name": "damage", "type": "list", "instruction": "Each kind of damage", "mark": "damage", "items": "text"},
                {"name": "delivered_on", "type": "date", "instruction": "Delivery date", "mark": "delivered_on"},
                {"name": "amount_paid", "type": "number", "instruction": "Total sent", "decimals": 2, "compute": "sum", "parts": "part"},
                {"name": "days_since_delivery", "type": "duration", "instruction": "Days from delivery to the message", "unit": "days", "compute": "days_between", "start": "delivered_on", "end": "written_on"},
            ],
        }
    )
    assert row["text"][row["fields"]["order_id"]["start"]:row["fields"]["order_id"]["end"]] == "BL-441"
    assert "[[" not in row["text"]
    assert row["fields"]["amount_paid"]["value"] == 2241.0
    assert row["fields"]["days_since_delivery"]["value"] == 9
    assert row["fields"]["tracking_number"]["value"] is None
    assert row["fields"]["damage"]["value"] == ["jar cracked", "lid missing"]
    assert row["fields"]["delivered_on"]["value"] == "2026-04-02"
    assert "BX-12" in row["text"]


def test_leaked_sum_is_dropped():
    with pytest.raises(Drop) as caught:
        audit_document(
            {
                "id": "leak",
                "marked_text": "Paid [[part]]10[[/]] and [[part]]5[[/]]. Total 15.",
                "ask": [
                    {"name": "amount_paid", "type": "number", "instruction": "Total", "decimals": 2, "compute": "sum", "parts": "part"},
                ],
            }
        )
    assert caught.value.reason.startswith("answer_leaked")


def test_due_date_is_shifted_by_code():
    row = audit_document(
        {
            "id": "due",
            "marked_text": "Issued [[issued]]2 Sep 2026[[/]], net [[net_days]]30[[/]].",
            "ask": [
                {
                    "name": "due_on",
                    "type": "date",
                    "instruction": "Issue date plus the net days",
                    "compute": "add_days",
                    "start": "issued",
                    "days": "net_days",
                }
            ],
        }
    )
    assert row["fields"]["due_on"]["value"] == "2026-10-02"
    assert "2026-10-02" not in row["text"]


def test_estimate_median_of_even_comps():
    row = audit_document(
        {
            "id": "house",
            "marked_text": "Comps [[comp]]410000[[/]] [[comp]]450000[[/]] [[comp]]470000[[/]] [[comp]]520000[[/]]. Subject is nearby.",
            "ask": [
                {
                    "name": "sale_price",
                    "type": "estimate",
                    "instruction": "Middle of the comparable prices",
                    "mark": "comp",
                    "decimals": 0,
                    "min": 100000,
                    "max": 900000,
                    "step": 1000,
                }
            ],
        }
    )
    assert row["fields"]["sale_price"]["value"] == 460000.0


def test_forecast_tail_is_removed_from_the_text():
    marked = "\n".join(
        [
            "Cups sold",
            "[[obs]]2026-06-01|40[[/]]",
            "[[obs]]2026-06-02|42[[/]]",
            "[[obs]]2026-06-03|38[[/]]",
            "[[obs]]2026-06-04|55[[/]]",
            "[[obs]]2026-06-05|61[[/]]",
            "[[obs]]2026-06-06|63[[/]]",
        ]
    )
    row = audit_document(
        {
            "id": "cups",
            "marked_text": marked,
            "ask": [
                {
                    "name": "next_cups",
                    "type": "forecast",
                    "instruction": "Next two days of cup sales",
                    "horizon": 2,
                    "decimals": 0,
                    "min": 0,
                    "max": 200,
                    "step": 1,
                }
            ],
        }
    )
    assert row["fields"]["next_cups"]["value"] == [61.0, 63.0]
    assert "61" not in row["text"]
    assert "63" not in row["text"]
    assert "40" in row["text"]


def test_write_split_keeps_only_valid_rows(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    good = {
        "id": "ok",
        "marked_text": "Code [[order_id]]Z-1[[/]]",
        "ask": [{"name": "order_id", "type": "text", "instruction": "Order", "mark": "order_id"}],
    }
    bad = {"id": "bad", "marked_text": "no marks", "ask": [{"name": "order_id", "type": "text", "instruction": "Order", "mark": "order_id"}]}
    (raw / "a.jsonl").write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n", encoding="utf-8")
    report = write_split(raw, tmp_path / "out")
    assert report["kept"] == 1
    assert report["dropped"][0]["reason"].startswith("missing_mark")
    assert (tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8").strip()
