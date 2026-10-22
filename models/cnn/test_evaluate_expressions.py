"""Tests for the expression scoring helpers (no model needed)."""
import pytest

from evaluate_expressions import evaluate, lev, score, summarise


@pytest.mark.parametrize("expr,value", [
    ("3+4", 7), ("2+3*4", 14), ("18-6/3", 16), ("100-25*3", 25),
    ("(4*(15-(3+2)))/8", 5), ("((9-4)*(3+2))/5", 5),
])
def test_evaluate_follows_precedence_and_brackets(expr, value):
    assert evaluate(expr) == pytest.approx(value)


@pytest.mark.parametrize("bad", ["", "3+", "(3+4", "3+4)", "8/0", "**2", "3(4)"])
def test_evaluate_rejects_malformed(bad):
    assert evaluate(bad) is None


def test_lev():
    assert lev("12+3", "12+3") == 0
    assert lev("12+3", "17+3") == 1
    assert lev("12+3", "123") == 1
    assert lev("", "abc") == 3


def test_score_and_summarise():
    rows = [dict(truth="3+4", read=r, **score("3+4", r, 7)) for r in ("3+4", "3-4", "34")]
    assert [r["exact"] for r in rows] == [True, False, False]
    assert [r["length_ok"] for r in rows] == [True, True, False]
    s = summarise(rows)
    assert s["exact"] == pytest.approx(1 / 3)
    assert s["symbol_acc"] == pytest.approx(1 - 2 / 9)
    assert s["result_ok"] == pytest.approx(1 / 3)
