from decimal import Decimal

import pytest

from extractor.labels import (
    char_span_to_tokens,
    document_token_indexes,
    expand_document,
    join_number,
    pack_prompt,
    question_token_indexes,
    split_number,
)


def test_round_trip_money_and_large_whole_numbers():
    for value, decimals in ((2413.25, 2), (15, 0), (212462.5, 1), (-3.5, 1)):
        encoded = split_number(value, decimals)
        assert join_number(encoded) == Decimal(str(value)) if decimals else join_number(encoded) == Decimal(int(value))


def test_expand_list_adds_a_stop_question_and_forecast_steps():
    row = {
        "fields": {
            "damage": {
                "type": "list",
                "instruction": "Each kind of damage",
                "value": ["jar", "lid"],
                "spans": [[1, 4], [8, 11]],
            },
            "next": {
                "type": "forecast",
                "instruction": "Next days",
                "value": [61, 63],
                "min": 0,
                "max": 100,
                "step": 1,
            },
        }
    }
    questions = expand_document(row)
    assert [question["instruction"] for question in questions if question["head"] == "point"] == [
        "Each kind of damage Item 1.",
        "Each kind of damage Item 2.",
        "Each kind of damage Item 3.",
    ]
    assert questions[2]["missing"] is True
    assert [question["value"] for question in questions if question["head"] == "spread"] == [61, 63]


def test_prompt_keeps_the_document_as_a_prefix():
    text = "blouse HT-3391 is here"
    questions = [{"field": "job", "instruction": "Job code"}]
    prompt, spans = pack_prompt(text, questions)
    assert prompt.startswith(text)
    start, end = spans[0]
    assert prompt[start:end] == "job: Job code"


def test_character_span_lands_on_the_token_that_contains_it():
    offsets = [(0, 0), (0, 6), (7, 14), (15, 17)]
    allowed = document_token_indexes(offsets, doc_chars=14)
    assert allowed == [1, 2]
    start, end = char_span_to_tokens(offsets, 7, 14, allowed)
    assert (start, end) == (1, 1)
    question = question_token_indexes(offsets, 15, 17)
    assert question == [3]


def test_span_that_misses_every_token_is_an_error():
    with pytest.raises(ValueError):
        char_span_to_tokens([(0, 3)], 10, 12, [0])
