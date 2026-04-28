# formulas.py
#
# Shared data model for the first-order logic prover.
# Defines the AST node types, sequent and branch structures,
# and utility functions used by both the baseline and improved provers.

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union


# ── Formula AST ───────────────────────────────────────────────────────────────
# All formula types inherit from Formula. Dataclasses are frozen (immutable
# and hashable) so they can be stored in sets and used as dictionary keys.

class Formula:
    pass


@dataclass(frozen=True)
class Atom(Formula):
    """A propositional atom, e.g. P, Q."""
    name: str
    def __str__(self): return self.name

@dataclass(frozen=True)
class Truth(Formula):
    """The logical constant ⊤ (true)."""
    def __str__(self): return "⊤"

@dataclass(frozen=True)
class Falsehood(Formula):
    """The logical constant ⊥ (false)."""
    def __str__(self): return "⊥"

@dataclass(frozen=True)
class Not(Formula):
    """Negation: ¬A."""
    formula: Formula
    def __str__(self): return f"¬({self.formula})"

@dataclass(frozen=True)
class And(Formula):
    """Conjunction: A ∧ B."""
    left: Formula
    right: Formula
    def __str__(self): return f"({self.left} ∧ {self.right})"

@dataclass(frozen=True)
class Or(Formula):
    """Disjunction: A ∨ B."""
    left: Formula
    right: Formula
    def __str__(self): return f"({self.left} ∨ {self.right})"

@dataclass(frozen=True)
class Implies(Formula):
    """Implication: A → B."""
    left: Formula
    right: Formula
    def __str__(self): return f"({self.left} → {self.right})"


# ── First-order logic terms ───────────────────────────────────────────────────

@dataclass(frozen=True)
class Variable:
    """A logical variable, e.g. x, y. Bound by a quantifier."""
    name: str
    def __str__(self): return self.name

@dataclass(frozen=True)
class Constant:
    """A logical constant, e.g. a, b, c0. Not bound by any quantifier."""
    name: str
    def __str__(self): return self.name

@dataclass(frozen=True)
class Predicate(Formula):
    """A predicate applied to a list of terms, e.g. P(x, a)."""
    name: str
    args: Tuple[Union[Variable, Constant], ...]
    def __str__(self):
        return f"{self.name}({', '.join(str(a) for a in self.args)})"

@dataclass(frozen=True)
class ForAll(Formula):
    """Universal quantification: ∀x.A."""
    var: Variable
    body: Formula
    def __str__(self): return f"∀{self.var}.{self.body}"

@dataclass(frozen=True)
class Exists(Formula):
    """Existential quantification: ∃x.A."""
    var: Variable
    body: Formula
    def __str__(self): return f"∃{self.var}.{self.body}"


# ── Sequent ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Sequent:
    """
    A sequent Γ ⊢ Δ where Γ (left) and Δ (right) are tuples of formulas.
    Frozen so sequents can be stored in sets for duplicate detection.
    """
    left: Tuple[Formula, ...]
    right: Tuple[Formula, ...]

    def __str__(self):
        l = ", ".join(str(f) for f in self.left) if self.left else "·"
        r = ", ".join(str(f) for f in self.right) if self.right else "·"
        return f"{l} ⊢ {r}"


# ── Branch ────────────────────────────────────────────────────────────────────
# Baseline branch — simple shared state, no isolation between branches.
# The improved prover overrides this with a branch-local version.

@dataclass
class Branch:
    """
    A single branch in the proof tree, represented as a sequence of sequents
    from the root to the current frontier.
    """
    sequents: List[Sequent]
    closed: bool = False   # True if the branch has been closed by a trivial rule
    failed: bool = False   # True if no rule applies and the branch cannot close

    def top(self) -> Sequent:
        """Return the current sequent at the top of this branch."""
        return self.sequents[-1]

    def add(self, sequent: Sequent):
        """Append a new sequent to this branch."""
        self.sequents.append(sequent)


# ── Substitution ──────────────────────────────────────────────────────────────

def substitute(formula: Formula, var: Variable, term: Union[Variable, Constant]) -> Formula:
    """
    Substitute all free occurrences of var with term in formula.
    Avoids variable capture by skipping quantifiers that rebind var.
    """
    if isinstance(formula, Predicate):
        return Predicate(formula.name, tuple(term if a == var else a for a in formula.args))
    elif isinstance(formula, And):
        return And(substitute(formula.left, var, term), substitute(formula.right, var, term))
    elif isinstance(formula, Or):
        return Or(substitute(formula.left, var, term), substitute(formula.right, var, term))
    elif isinstance(formula, Implies):
        return Implies(substitute(formula.left, var, term), substitute(formula.right, var, term))
    elif isinstance(formula, Not):
        return Not(substitute(formula.formula, var, term))
    elif isinstance(formula, ForAll):
        if formula.var == var: return formula  # var is rebound, stop here
        return ForAll(formula.var, substitute(formula.body, var, term))
    elif isinstance(formula, Exists):
        if formula.var == var: return formula  # var is rebound, stop here
        return Exists(formula.var, substitute(formula.body, var, term))
    return formula


# ── Constant collection ───────────────────────────────────────────────────────

def collect_constants(formula: Formula) -> set:
    """
    Recursively collect all constants appearing in a formula.
    Used to initialise the term pool before proof search begins.
    """
    if isinstance(formula, Predicate):
        return {a for a in formula.args if isinstance(a, Constant)}
    elif isinstance(formula, Not):
        return collect_constants(formula.formula)
    elif isinstance(formula, (And, Or, Implies)):
        return collect_constants(formula.left) | collect_constants(formula.right)
    elif isinstance(formula, (ForAll, Exists)):
        return collect_constants(formula.body)
    return set()


# ── Sequent helpers ───────────────────────────────────────────────────────────

def remove_one(items: Tuple, target) -> Tuple:
    """Remove the first occurrence of target from a tuple."""
    lst = list(items)
    lst.remove(target)
    return tuple(lst)

def replace_one_with_many(items: Tuple, target, replacements: List) -> Tuple:
    """Replace the first occurrence of target in a tuple with a list of replacements."""
    lst = list(items)
    lst.remove(target)
    lst.extend(replacements)
    return tuple(lst)


# ── Trivial closure checks ────────────────────────────────────────────────────

def is_identity(s: Sequent) -> bool:
    """True if any formula appears on both sides of the sequent (id rule)."""
    return any(f in s.right for f in s.left)

def has_top_right(s: Sequent) -> bool:
    """True if ⊤ appears on the right (⊤R rule)."""
    return any(isinstance(f, Truth) for f in s.right)

def has_bottom_left(s: Sequent) -> bool:
    """True if ⊥ appears on the left (⊥L rule)."""
    return any(isinstance(f, Falsehood) for f in s.left)

def is_trivial(s: Sequent) -> bool:
    """True if the sequent can be closed immediately by a trivial rule."""
    return is_identity(s) or has_top_right(s) or has_bottom_left(s)
