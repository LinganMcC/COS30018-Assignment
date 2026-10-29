"""Tests for the expression parser: precedence, brackets, symbols and errors."""

import re
import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from expression_parser import (ExpressionError, evaluate, evaluate_expression,
                               format_result, tokenise)


# Tokeniser
def test_multi_digit_numbers_are_one_token():
    assert tokenise("120+7") == [("num", 120), ("+", "+"), ("num", 7)]


def test_spaces_are_ignored():
    assert tokenise(" 3 *  4 ") == tokenise("3*4")


def test_look_alike_symbols_map_to_one_kind():
    # what the model outputs and what people type both reach the same operator
    kinds = [kind for kind, _ in tokenise("1×2x3X4*5÷6/7−8–9—1-2")]
    assert kinds == ["num", "*", "num", "*", "num", "*", "num", "*", "num",
                     "/", "num", "/", "num", "-", "num", "-", "num", "-",
                     "num", "-", "num"]


def test_unknown_symbol_is_rejected():
    with pytest.raises(ExpressionError, match="unexpected symbol 'a'"):
        tokenise("3a")


# Precedence and associativity
@pytest.mark.parametrize("text, want", [
    ("2+3*4", 14),               # * before +
    ("2*3+4", 10),
    ("20-6/2", 17),              # / before -
    ("8-3-2", 3),                # left to right, not 8-(3-2)
    ("100/10/5", 2),             # left to right, not 100/(10/5)
    ("10÷2×3", 15),              # same level: left to right
    ("26-3*(17-4)", -13),
])
def test_operator_precedence(text, want):
    assert evaluate(text) == want


# Brackets
@pytest.mark.parametrize("text, want", [
    ("(2+3)*4", 20),
    ("((7))", 7),
    ("(6×(14−(2+4)))÷4", 12),      # the README example, nested three deep
    ("((9−3)×(6+4))÷5", 12),
    ("70−((9+3)×5)", 10),
])
def test_brackets(text, want):
    assert evaluate(text) == want


# Unary signs
@pytest.mark.parametrize("text, want", [
    ("-5+2", -3),
    ("2*-3", -6),
    ("26 - -3", 29),
    ("-(2+3)", -5),
    ("+4", 4),
])
def test_unary_plus_and_minus(text, want):
    assert evaluate(text) == want


# Exact arithmetic and formatting
def test_division_is_exact():
    # Fractions, so 1/3*3 is exactly 1 and not 0.9999999
    assert evaluate("1/3*3") == 1
    assert evaluate("10/4") == Fraction(5, 2)


@pytest.mark.parametrize("value, want", [
    (Fraction(14), "14"),
    (Fraction(-13), "-13"),
    (Fraction(5, 2), "2.5"),
    (Fraction(1, 3), "0.333333"),
    (Fraction(1, 8), "0.125"),
])
def test_format_result(value, want):
    assert format_result(value) == want


# Invalid input
@pytest.mark.parametrize("text, message", [
    ("", "empty expression"),
    ("   ", "empty expression"),
    ("3+", "ends unexpectedly"),
    ("(3+4", "missing ')'"),
    ("3+4)", "unmatched ')'"),
    ("()", "empty brackets"),
    ("3(4)", "missing operator before '('"),
    ("12 3", "missing operator before '3'"),
    ("(4+5)(2)", "missing operator before '('"),
    ("*3", "unexpected '*'"),
    ("3++*4", "unexpected '*'"),
    ("5/0", "division by zero"),
    ("5/(3-3)", "division by zero"),
])
def test_invalid_expressions_raise(text, message):
    with pytest.raises(ExpressionError, match=re.escape(message)):
        evaluate(text)


def test_misread_expressions_from_the_evaluation_fail_cleanly():
    # real outputs of the model on bad segmentations (expression_parser_test.py)
    for text in ["5-*((0+6)-40", "80/(2+6)+5-", "((3+6)*7)8", "2+(30/6))*6"]:
        with pytest.raises(ExpressionError):
            evaluate(text)


def test_very_deep_nesting_is_an_error_not_a_crash():
    text = "(" * 5000 + "1" + ")" * 5000
    with pytest.raises(ExpressionError, match="nested too deeply"):
        evaluate(text)


# GUI wrapper
def test_wrapper_returns_result_on_success():
    assert evaluate_expression("(3+4)*5") == ("35", None)


def test_wrapper_returns_message_and_never_raises():
    result, error = evaluate_expression("5/0")
    assert result is None
    assert error == "division by zero"
