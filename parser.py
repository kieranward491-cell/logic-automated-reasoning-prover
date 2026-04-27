# parser.py

from torch import var
from formulas import *


class ParseError(Exception):
    pass


# ── Tokeniser ────────────────────────────────────────────────────────────────

SYMBOLS = ['∀', '∃', '¬', '∧', '∨', '→', '⊤', '⊥', '(', ')', '.', ',']


def tokenise(text: str) -> list[str]:
    tokens = []
    i = 0

    while i < len(text):
        ch = text[i]

        if ch.isspace():
            i += 1
            continue

        # ASCII fallbacks
        if text[i:i+2] == '->':
            tokens.append('→')
            i += 2
            continue
        if text[i:i+2] == '/\\':
            tokens.append('∧')
            i += 2
            continue
        if text[i:i+2] == '\\/':
            tokens.append('∨')
            i += 2
            continue
        if ch in ('~', '!'):
            tokens.append('¬')
            i += 1
            continue

        # Unicode symbols
        if ch in SYMBOLS:
            tokens.append(ch)
            i += 1
            continue

        # Identifiers
        if ch.isalnum() or ch == '_':
            j = i
            while j < len(text) and (text[j].isalnum() or text[j] == '_'):
                j += 1
            tokens.append(text[i:j])
            i = j
            continue

        raise ParseError(f"Unexpected character: '{ch}'")

    return tokens


# ── Parser ───────────────────────────────────────────────────────────────────

class Parser:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.pos = 0
        self.bound_vars: list[str] = []

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected: str | None = None) -> str:
        token = self.peek()
        if token is None:
            if expected is None:
                raise ParseError("Unexpected end of input")
            raise ParseError(f"Unexpected end of input, expected '{expected}'")
        if expected is not None and token != expected:
            raise ParseError(f"Expected '{expected}', got '{token}'")
        self.pos += 1
        return token

    def at_end(self) -> bool:
        return self.pos >= len(self.tokens)

    # formula ::= implies
    def parse_formula(self) -> Formula:
        return self.parse_implies()

    # implies ::= or ('→' implies)?
    # Right-associative: P → Q → R = P → (Q → R)
    def parse_implies(self) -> Formula:
        left = self.parse_or()
        if self.peek() == '→':
            self.consume('→')
            right = self.parse_implies()
            return Implies(left, right)
        return left

    # or ::= and ('∨' and)*
    def parse_or(self) -> Formula:
        left = self.parse_and()
        while self.peek() == '∨':
            self.consume('∨')
            right = self.parse_and()
            left = Or(left, right)
        return left

    # and ::= unary ('∧' unary)*
    def parse_and(self) -> Formula:
        left = self.parse_unary()
        while self.peek() == '∧':
            self.consume('∧')
            right = self.parse_unary()
            left = And(left, right)
        return left

    # unary ::= '¬' unary | quantifier | atom
    def parse_unary(self) -> Formula:
        if self.peek() == '¬':
            self.consume('¬')
            return Not(self.parse_unary())

        if self.peek() in ('∀', '∃'):
            return self.parse_quantifier()

        return self.parse_atom()

    # quantifier ::= ('∀' | '∃') identifier '.' formula
    # After
    def parse_quantifier(self) -> Formula:
        quantifier = self.consume()
        var_name = self.consume()

        if not (var_name[0].isalpha() or var_name[0] == '_'):
         raise ParseError(f"Invalid variable name '{var_name}'")

        var = Variable(var_name)
        self.consume('.')

        self.bound_vars.append(var_name)
        body = self.parse_unary()   # ← only consume one unary expression
        self.bound_vars.pop()

        if quantifier == '∀':
            return ForAll(var, body)
        return Exists(var, body)

    # atom ::= '⊤' | '⊥' | identifier '(' term_list ')' | identifier | '(' formula ')'
    def parse_atom(self) -> Formula:
        token = self.peek()

        if token == '⊤':
            self.consume()
            return Truth()

        if token == '⊥':
            self.consume()
            return Falsehood()

        if token == '(':
            self.consume('(')
            formula = self.parse_formula()
            self.consume(')')
            return formula

        if token is None:
            raise ParseError("Unexpected end of input")

        name = self.consume()

        # Predicate application
        if self.peek() == '(':
            self.consume('(')
            args = self.parse_term_list()
            self.consume(')')
            return Predicate(name, tuple(args))

        # Otherwise propositional atom
        return Atom(name)

    # term_list ::= term (',' term)*
    def parse_term_list(self) -> list[Variable | Constant]:
        terms = []

        while True:
            term_name = self.consume()

            if not (term_name[0].isalnum() or term_name[0] == '_'):
                raise ParseError(f"Invalid term name '{term_name}'")

            if term_name in self.bound_vars:
                terms.append(Variable(term_name))
            else:
                terms.append(Constant(term_name))

            if self.peek() == ',':
                self.consume(',')
            else:
                break

        return terms


# ── Public API ───────────────────────────────────────────────────────────────

def parse_formula(text: str) -> Formula:
    tokens = tokenise(text)
    parser = Parser(tokens)
    formula = parser.parse_formula()

    if not parser.at_end():
        raise ParseError(f"Unexpected token after formula: '{parser.peek()}'")

    return formula


def parse_file(path: str) -> list[Formula]:
    formulae = []

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                formulae.append(parse_formula(line))
            except ParseError as e:
                raise ParseError(f"Line {lineno}: {e}") from e

    return formulae


# ── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        "P → P",
        "(P ∧ Q) → P",
        "P → (P ∨ Q)",
        "¬¬P → P",
        "∀x.P(x) → P(a)",
        "P(a) → ∃x.P(x)",
        "∀x.(P(x) → Q(x))",
        "∃x.P(x) → ∀x.P(x)",
        "∀x.∀y.(P(x) → P(y))",
        "(P ∧ Q) → (Q ∧ P)",
        "⊤",
        "⊥ → P",
        "P /\\ Q -> P",
        "P \\/ Q -> Q \\/ P",
        "~P -> P -> P",
        "∀x.(P(x) → Q(a))",
        "∀x.∃y.R(x,y,a,b)",
        "P(a,b,c)",
    ]

    print("Parser tests:")
    passed = 0
    failed = 0

    for text in tests:
        try:
            formula = parse_formula(text)
            print(f"[PASS] {text!r:35s} => {formula}")
            passed += 1
        except ParseError as e:
            print(f"[FAIL] {text!r:35s} => ERROR: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} passed")