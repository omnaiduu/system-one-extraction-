import torch

from extractor.audit import audit_document
from extractor.gemma_cache import apply_question_block
from extractor.highlight import apply_recipe, expand_highlighter, program_matches_field, verify_corpus


def _support():
    return audit_document(
        {
            "id": "support-demo",
            "genre": "whatsapp",
            "marked_text": (
                "Riya, blender [[order_id]]BL-441[[/]] came on [[delivered_on]]2 Apr 2026[[/]]. "
                "[[damage]]jar cracked[[/]] and [[damage]]lid missing[[/]]. "
                "I sent [[part]]1899[[/]] plus [[part]]342[[/]] by UPI. "
                "Writing this [[written_on]]11 Apr 2026[[/]]. Shift [[shift]]3[[/]]."
            ),
            "ask": [
                {"name": "order_id", "type": "text", "instruction": "Order id", "mark": "order_id"},
                {"name": "damage", "type": "list", "instruction": "Each kind of damage", "mark": "damage", "items": "text"},
                {"name": "amount_paid", "type": "number", "instruction": "Total sent", "decimals": 2, "compute": "sum", "parts": "part"},
                {
                    "name": "days_since_delivery",
                    "type": "duration",
                    "instruction": "Days from delivery to the message",
                    "unit": "days",
                    "compute": "days_between",
                    "start": "delivered_on",
                    "end": "written_on",
                },
                {
                    "name": "due_on",
                    "type": "date",
                    "instruction": "Delivery date plus 3 days",
                    "compute": "add_days",
                    "start": "delivered_on",
                    "days": "shift",
                },
            ],
        }
    )


def test_program_reads_the_marked_pieces():
    row = _support()
    assert program_matches_field(row["fields"]["amount_paid"], row["text"])
    assert program_matches_field(row["fields"]["days_since_delivery"], row["text"])
    questions, skipped = expand_highlighter(row)
    assert skipped == []
    paid = [question["instruction"] for question in questions if question["field"] == "amount_paid"]
    assert paid == ["Total sent Part 1.", "Total sent Part 2.", "Total sent Part 3."]
    assert questions[-1]["recipe"]["op"] == "add_days"
    days = [question for question in questions if question["field"] == "days_since_delivery"]
    assert [question["instruction"] for question in days] == [
        "Days from delivery to the message The start date.",
        "Days from delivery to the message The end date.",
    ]
    assert apply_recipe(days[0]["recipe"], ["2 Apr 2026", "11 Apr 2026"]) == 9


def test_due_date_uses_the_date_and_the_day_count():
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
    assert [piece["text"] for piece in row["fields"]["due_on"]["evidence"]] == ["2 Sep 2026", "30"]
    assert program_matches_field(row["fields"]["due_on"], row["text"])


def test_question_block_leaves_the_text_rows_and_closes_the_text_columns():
    mask = torch.zeros(1, 1, 6, 6)
    mask[:, :, :, 4:] = -1
    blocked = apply_question_block(mask, doc_tokens=4)
    assert torch.equal(blocked[:, :, :4, :], mask[:, :, :4, :])
    assert torch.equal(blocked[:, :, 4:, 4:], mask[:, :, 4:, 4:])
    assert torch.all(blocked[:, :, 4:, :4] == torch.finfo(mask.dtype).min)


def test_every_saved_mark_reproduces_its_audited_value():
    checked = verify_corpus()
    assert checked > 1000
