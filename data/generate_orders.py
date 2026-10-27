"""Write order texts whose labels are known exactly.

Each order id is inserted by this script, then found with a string search.
A second code is often inserted too, and that code is not the label.
Some texts have no order id. The wording and the position change on purpose.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


OPENINGS = [
    "Hello",
    "Update",
    "Note",
    "Customer wrote",
    "Desk log",
    "Message received",
    "Shipment note",
    "Accounts copy",
]

FACTS = [
    "the crate arrived with a cracked corner",
    "payment was made at the counter",
    "the packet was left at the gate",
    "delivery was signed by Ravi",
    "the box was short one item",
    "the courier called before arriving",
    "the seal was intact",
    "a refund was requested the same day",
]

CLOSINGS = [
    "Please reply today.",
    "File this with the receipt.",
    "No other reference was given.",
    "Confirm when it is booked.",
    "The customer is waiting.",
]


def code(rng: random.Random, prefix: str | None = None) -> str:
    kind = prefix or rng.choice(["A", "INV", "AB", "OD", "ZZ", "REF"])
    number = rng.randint(10, 99999)
    style = rng.choice(["dash", "slash", "plain", "year"])
    if style == "dash":
        return f"{kind}-{number}"
    if style == "slash":
        return f"{kind}/{number}"
    if style == "year":
        return f"{kind}/2026/{number}"
    return f"{kind}{number}"


def payment(rng: random.Random) -> str:
    price = rng.randint(20, 8000) + rng.choice([0, 0.5, 0.25])
    tax = round(price * rng.choice([0.05, 0.12, 0.18]), 2)
    unit = rng.choice(["Rs", "INR", "₹"])
    return f"Paid {unit} {price:.2f} and tax {unit} {tax:.2f}."


def place(parts: list[str], rng: random.Random, piece: str) -> list[str]:
    parts = list(parts)
    parts.insert(rng.randint(0, len(parts)), piece)
    return parts


def make_example(rng: random.Random, example_id: str) -> dict:
    order_id = code(rng)
    distractor = code(rng)
    while distractor == order_id or order_id in distractor or distractor in order_id:
        distractor = code(rng)

    present = rng.random() > 0.15
    parts = [
        rng.choice(OPENINGS) + ".",
        rng.choice(FACTS) + ".",
        payment(rng),
        rng.choice(CLOSINGS),
    ]
    if rng.random() < 0.7:
        decoy_word = rng.choice(["old ref", "phone ext", "batch", "do not use"])
        parts = place(parts, rng, f"{decoy_word} {distractor}.")

    if present:
        label_word = rng.choice(
            ["order", "order id", "docket", "consignment", "reference", "booking"]
        )
        parts = place(parts, rng, f"{label_word} {order_id}.")

    text = " ".join(parts)
    if present:
        start = text.index(order_id)
        end = start + len(order_id)
        if text[start:end] != order_id:
            raise RuntimeError(text)
        if distractor in text and text.index(distractor) == start:
            raise RuntimeError("label collided with distractor")
        field = {"type": "text", "value": order_id, "span": [start, end]}
    else:
        if order_id in text:
            raise RuntimeError("absent id was written into the text")
        field = {"type": "text", "value": None, "span": None}

    return {"id": example_id, "state": text, "fields": {"order_id": field}}


def write_split(path: Path, count: int, seed: int) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            example = make_example(rng, f"{path.stem}-{index:04d}")
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/orders"))
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--val", type=int, default=200)
    parser.add_argument("--seed", type=int, default=22)
    args = parser.parse_args()
    write_split(args.out / "train.jsonl", args.train, args.seed)
    write_split(args.out / "val.jsonl", args.val, args.seed + 1)


if __name__ == "__main__":
    main()
