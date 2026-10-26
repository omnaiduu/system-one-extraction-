import torch

from extractor.heads import NumberHead, PointHead, number_loss, point_loss


def test_missing_score_reads_the_best_match():
    torch.manual_seed(0)
    head = PointHead(8, inner=4)
    head.missing_question.weight.data.zero_()
    head.missing_question.bias.data.zero_()
    head.missing_match.weight.data.copy_(torch.tensor([[1.0, 0.0]]))
    head.missing_match.bias.data.zero_()
    document = torch.randn(2, 5, 8)
    question = torch.randn(2, 8)
    output = head(document, question)
    assert torch.allclose(output["missing"], output["start"].amax(dim=-1))


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
    document = torch.randn(32, 3, 4)
    question = torch.randn(32, 4)
    target = {
        "sign": torch.zeros(32, dtype=torch.long),
        "whole": torch.zeros(32, 7, dtype=torch.long),
        "frac": torch.zeros(32, 2, dtype=torch.long),
    }
    target["whole"][:, -1] = 4
    optimizer = torch.optim.Adam(head.parameters(), lr=0.05)
    before = _number_hits(head, document, question, target)
    for _ in range(40):
        optimizer.zero_grad()
        loss = number_loss(head(document, question), target)
        loss.backward()
        optimizer.step()
    after = _number_hits(head, document, question, target)
    assert after > before


def test_number_head_reads_the_document_tokens():
    torch.manual_seed(2)
    head = NumberHead(8)
    question = torch.randn(1, 8)
    first = head(torch.randn(1, 4, 8), question)["whole"]
    second = head(torch.randn(1, 4, 8), question)["whole"]
    assert not torch.allclose(first, second)


def _point_hits(head, document, question, start, end) -> int:
    output = head(document, question)
    got_start = output["start"].argmax(-1)
    got_end = output["end"].argmax(-1)
    return int(((got_start == start) & (got_end == end)).sum())


def _number_hits(head, document, question, target) -> int:
    output = head(document, question)
    whole = output["whole"].argmax(-1)
    return int((whole == target["whole"]).all(-1).sum())
