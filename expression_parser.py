from fractions import Fraction

MULTIPLY = {"*", "x", "X", "×"}
DIVIDE = {"/", "÷"}
MINUS = {"-", "−", "–", "—"}
PLUS = {"+"}


class ExpressionError(Exception):
    """Raised for anything that is not a valid, computable expression."""

# Tokeniser
def tokenise(text):
    """Return a list of (kind, value) tokens.
    kind is one of: 'num', '+', '-', '*', '/', '(', ')'."""
    tokens = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
        elif ch.isdigit():
            j = i
            while j < len(text) and text[j].isdigit():
                j += 1
            tokens.append(("num", int(text[i:j])))
            i = j
        elif ch in PLUS:
            tokens.append(("+", ch))
            i += 1
        elif ch in MINUS:
            tokens.append(("-", ch))
            i += 1
        elif ch in MULTIPLY:
            tokens.append(("*", ch))
            i += 1
        elif ch in DIVIDE:
            tokens.append(("/", ch))
            i += 1
        elif ch in "()":
            tokens.append((ch, ch))
            i += 1
        else:
            raise ExpressionError(f"unexpected symbol '{ch}'")
    return tokens

# Parser / evaluator
class _Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos][0] if self.pos < len(self.tokens) else None

    def take(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self):
        if not self.tokens:
            raise ExpressionError("empty expression")
        value = self.expr()
        if self.pos < len(self.tokens):
            kind, val = self.tokens[self.pos]
            if kind == ")":
                raise ExpressionError("unmatched ')'")
            if kind in ("num", "("):
                raise ExpressionError("missing operator before "
                                      f"'{val}'")
            raise ExpressionError(f"unexpected '{val}'")
        return value

    def expr(self):
        value = self.term()
        while self.peek() in ("+", "-"):
            op = self.take()[0]
            rhs = self.term()
            value = value + rhs if op == "+" else value - rhs
        return value

    def term(self):
        value = self.unary()
        while self.peek() in ("*", "/"):
            op = self.take()[0]
            rhs = self.unary()
            if op == "*":
                value = value * rhs
            else:
                if rhs == 0:
                    raise ExpressionError("division by zero")
                value = value / rhs
        return value

    def unary(self):
        if self.peek() in ("+", "-"):
            op = self.take()[0]
            operand = self.unary()
            return operand if op == "+" else -operand
        return self.primary()

    def primary(self):
        kind = self.peek()
        if kind is None:
            raise ExpressionError("expression ends unexpectedly "
                                  "(missing number or ')')")
        if kind == "num":
            return Fraction(self.take()[1])
        if kind == "(":
            self.take()
            if self.peek() == ")":
                raise ExpressionError("empty brackets '()'")
            value = self.expr()
            if self.peek() != ")":
                raise ExpressionError("missing ')'")
            self.take()
            return value
        # an operator or ')' where a number should be
        val = self.tokens[self.pos][1]
        raise ExpressionError(f"unexpected '{val}' (a number was expected)")


def evaluate(text):
    """Evaluate `text` and return an exact Fraction. Raises ExpressionError."""
    try:
        return _Parser(tokenise(text)).parse()
    except RecursionError:
        raise ExpressionError("expression is nested too deeply")


def format_result(value):
    """Integers print as integers; other values as a trimmed decimal."""
    if value.denominator == 1:
        return str(value.numerator)
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def evaluate_expression(text):
    """GUI-friendly wrapper: returns (result_string, None) on success or
    (None, error_message) on failure. Never raises."""
    try:
        return format_result(evaluate(text)), None
    except ExpressionError as e:
        return None, str(e)


if __name__ == "__main__":
    # quick self-test:  python expression_parser.py
    cases = {
        "26-3*(17-4)": "-13",
        "2+3*4": "14",
        "(2+3)*4": "20",
        "10/4": "2.5",
        "10÷2×3": "15",
        "8-3-2": "3",
        "100/10/5": "2",
        "-5+2": "-3",
        "2*-3": "-6",
        "26 - -3": "29",
        "((7))": "7",
        "1/3*3": "1",
        "1/3": "0.333333",
    }
    bad = ["", "3+", "(3+4", "3+4)", "()", "3(4)", "12 3", "5/0", "*3", "3++*4", "3a", "(4+5)(2)"]
    ok = True
    for expr, want in cases.items():
        got, err = evaluate_expression(expr)
        flag = "ok " if got == want else "BAD"
        ok &= got == want
        print(f"{flag} {expr!r:18} -> {got!r} (expected {want!r})")
    for expr in bad:
        got, err = evaluate_expression(expr)
        flag = "ok " if err else "BAD"
        ok &= bool(err)
        print(f"{flag} {expr!r:18} -> error: {err}")
    print("ALL PASSED" if ok else "FAILURES")
