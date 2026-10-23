import torch

from extractor.heads import NumberHead, PointHead, number_loss, point_loss


def test_point_head_learns_a_fixed_span():
    torch.manual_seed(0)
    head = PointHead(8, inner=8)
    document = torch.randn(16, 3, 8)
    question = torch.randn(16, 8)
    missing = torch.zeros(16, dtype=torch.long)
    start = torch.ones(16, dtype=torch.long)
    end = torch.full((16,), 2, dtype=torch.long)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.05)
    before = _point_hits(head, document, question, start, end)
    for _ in range(40):
        optimizer.zero_grad()
        loss = point_loss(head(document, question), missing, start, end)
        loss.backward()
        optimizer.step()
    after = _point_hits(head, document, question, start, end)
    assert after > before


def test_number_head_learns_one_digit_pattern():
    torch.manual_seed(1)
    head = NumberHead(4)
    question = torch.randn(32, 4)
    target = {
        "sign": torch.zeros(32, dtype=torch.long),
        "whole": torch.zeros(32, 7, dtype=torch.long),
        "frac": torch.zeros(32, 2, dtype=torch.long),
    }
    target["whole"][:, -1] = 4
    optimizer = torch.optim.Adam(head.parameters(), lr=0.05)
    before = _number_hits(head, question, target)
    for _ in range(30):
        optimizer.zero_grad()
        loss = number_loss(head(question), target)
        loss.backward()
        optimizer.step()
    after = _number_hits(head, question, target)
    assert after > before


def _point_hits(head, document, question, start, end) -> int:
    output = head(document, question)
    got_start = output["start"].argmax(-1)
    got_end = output["end"].argmax(-1)
    return int(((got_start == start) & (got_end == end)).sum())


def _number_hits(head, question, target) -> int:
    output = head(question)
    whole = output["whole"].argmax(-1)
    return int((whole == target["whole"]).all(-1).sum())
