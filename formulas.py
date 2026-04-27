from dataclasses import dataclass, field
from typing import List, Tuple, Union

class Formula:
    pass

@dataclass(frozen=True)
class Atom(Formula):
    name: str

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class Truth(Formula):
    def __str__(self):
        return "⊤"


@dataclass(frozen=True)
class Falsehood(Formula):
    def __str__(self):
        return "⊥"


@dataclass(frozen=True)
class Not(Formula):
    formula: Formula

    def __str__(self):
        return f"¬({self.formula})"


@dataclass(frozen=True)
class And(Formula):
    left: Formula
    right: Formula

    def __str__(self):
        return f"({self.left} ∧ {self.right})"


@dataclass(frozen=True)
class Or(Formula):
    left: Formula
    right: Formula

    def __str__(self):
        return f"({self.left} ∨ {self.right})"


@dataclass(frozen=True)
class Implies(Formula):
    left: Formula
    right: Formula

    def __str__(self):
        return f"({self.left} → {self.right})"


@dataclass(frozen=True)
class Sequent:
    left: Tuple[Formula, ...]
    right: Tuple[Formula, ...]

    def __str__(self):
        left_str = ", ".join(str(f) for f in self.left) if self.left else "·"
        right_str = ", ".join(str(f) for f in self.right) if self.right else "·"
        return f"{left_str} ⊢ {right_str}"


@dataclass
class Branch:
    sequents: List[Sequent]
    closed: bool = False
    failed: bool = False

    def top(self) -> Sequent:
        return self.sequents[-1]

    def add(self, sequent: Sequent):
        self.sequents.append(sequent)

@dataclass(frozen=True)
class Variable:
    name: str

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class Constant:
    name: str

    def __str__(self):
        return self.name
    
@dataclass(frozen=True)
class Predicate(Formula):
    name: str
    args: Tuple[Union[Variable, Constant], ...]

    def __str__(self):
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.name}({args_str})"


@dataclass(frozen=True)
class ForAll(Formula):
    var: Variable
    body: Formula

    def __str__(self):
        return f"∀{self.var}.{self.body}"


@dataclass(frozen=True)
class Exists(Formula):
    var: Variable
    body: Formula

    def __str__(self):
        return f"∃{self.var}.{self.body}"
    
def substitute(formula: Formula, var: Variable, term: Union[Variable, Constant]) -> Formula:
    if isinstance(formula, Predicate):
        new_args = tuple(term if arg == var else arg for arg in formula.args)
        return Predicate(formula.name, new_args)

    elif isinstance(formula, And):
        return And(
            substitute(formula.left, var, term),
            substitute(formula.right, var, term),
        )

    elif isinstance(formula, Or):
        return Or(
            substitute(formula.left, var, term),
            substitute(formula.right, var, term),
        )

    elif isinstance(formula, Implies):
        return Implies(
            substitute(formula.left, var, term),
            substitute(formula.right, var, term),
        )

    elif isinstance(formula, Not):
        return Not(substitute(formula.formula, var, term))

    elif isinstance(formula, ForAll):
        if formula.var == var:
            return formula  # avoid variable capture
        return ForAll(formula.var, substitute(formula.body, var, term))

    elif isinstance(formula, Exists):
        if formula.var == var:
            return formula
        return Exists(formula.var, substitute(formula.body, var, term))

    return formula

def collect_constants(formula: Formula) -> set:
    constants = set()

    if isinstance(formula, Predicate):
        for arg in formula.args:
            if isinstance(arg, Constant):
                constants.add(arg)

    elif isinstance(formula, Not):
        constants |= collect_constants(formula.formula)

    elif isinstance(formula, And):
        constants |= collect_constants(formula.left)
        constants |= collect_constants(formula.right)

    elif isinstance(formula, Or):
        constants |= collect_constants(formula.left)
        constants |= collect_constants(formula.right)

    elif isinstance(formula, Implies):
        constants |= collect_constants(formula.left)
        constants |= collect_constants(formula.right)

    elif isinstance(formula, ForAll):
        constants |= collect_constants(formula.body)

    elif isinstance(formula, Exists):
        constants |= collect_constants(formula.body)

    return constants

def remove_one(items: Tuple[Formula, ...], target: Formula) -> Tuple[Formula, ...]:
    items = list(items)
    items.remove(target)
    return tuple(items)


def replace_one_with_many(
    items: Tuple[Formula, ...],
    target: Formula,
    replacements: List[Formula]
) -> Tuple[Formula, ...]:
    items = list(items)
    items.remove(target)
    items.extend(replacements)
    return tuple(items)

def is_identity(sequent: Sequent) -> bool:
    return any(f in sequent.right for f in sequent.left)


def has_top_right(sequent: Sequent) -> bool:
    return any(isinstance(f, Truth) for f in sequent.right)


def has_bottom_left(sequent: Sequent) -> bool:
    return any(isinstance(f, Falsehood) for f in sequent.left)


def is_trivial(sequent: Sequent) -> bool:
    return is_identity(sequent) or has_top_right(sequent) or has_bottom_left(sequent)