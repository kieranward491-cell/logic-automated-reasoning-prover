from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from formulas import *


def apply_and_left(s: Sequent) -> Optional[Sequent]:
    for f in s.left:
        if isinstance(f, And):
            return Sequent(replace_one_with_many(s.left, f, [f.left, f.right]), s.right)
    return None

def apply_or_right(s: Sequent) -> Optional[Sequent]:
    for f in s.right:
        if isinstance(f, Or):
            return Sequent(s.left, replace_one_with_many(s.right, f, [f.left, f.right]))
    return None

def apply_implies_right(s: Sequent) -> Optional[Sequent]:
    for f in s.right:
        if isinstance(f, Implies):
            return Sequent(
                tuple(list(s.left) + [f.left]),
                tuple(list(remove_one(s.right, f)) + [f.right])
            )
    return None

def apply_not_left(s: Sequent) -> Optional[Sequent]:
    for f in s.left:
        if isinstance(f, Not):
            return Sequent(remove_one(s.left, f), tuple(list(s.right) + [f.formula]))
    return None

def apply_not_right(s: Sequent) -> Optional[Sequent]:
    for f in s.right:
        if isinstance(f, Not):
            return Sequent(tuple(list(s.left) + [f.formula]), remove_one(s.right, f))
    return None

def apply_and_right(s: Sequent) -> Optional[Tuple[Sequent, Sequent]]:
    for f in s.right:
        if isinstance(f, And):
            base = list(remove_one(s.right, f))
            return (Sequent(s.left, tuple(base + [f.left])),
                    Sequent(s.left, tuple(base + [f.right])))
    return None

def apply_or_left(s: Sequent) -> Optional[Tuple[Sequent, Sequent]]:
    for f in s.left:
        if isinstance(f, Or):
            base = list(remove_one(s.left, f))
            return (Sequent(tuple(base + [f.left]), s.right),
                    Sequent(tuple(base + [f.right]), s.right))
    return None

def apply_implies_left(s: Sequent) -> Optional[Tuple[Sequent, Sequent]]:
    for f in s.left:
        if isinstance(f, Implies):
            base = list(remove_one(s.left, f))
            return (Sequent(tuple(base), tuple(list(s.right) + [f.left])),
                    Sequent(tuple(base + [f.right]), s.right))
    return None


# ∀R and ∃L: always fresh constant, consume the formula
def apply_forall_right(s: Sequent, terms: set, counter: int):
    for f in s.right:
        if isinstance(f, ForAll):
            c = Constant(f"c{counter}")
            terms.add(c)
            return Sequent(s.left, replace_one_with_many(s.right, f, [substitute(f.body, f.var, c)])), counter + 1
    return None, counter

def apply_exists_left(s: Sequent, terms: set, counter: int):
    for f in s.left:
        if isinstance(f, Exists):
            c = Constant(f"c{counter}")
            terms.add(c)
            return Sequent(replace_one_with_many(s.left, f, [substitute(f.body, f.var, c)]), s.right), counter + 1
    return None, counter


# ∀L and ∃R: try existing term if available, otherwise fresh — keep the formula
def apply_forall_left(s: Sequent, terms: set, counter: int):
    for f in s.left:
        if isinstance(f, ForAll):
            if terms:
                t = next(iter(terms))
            else:
                t = Constant(f"c{counter}")
                counter += 1
                terms.add(t)
            instantiated = substitute(f.body, f.var, t)
            return Sequent(tuple(list(s.left) + [instantiated]), s.right), counter
    return None, counter

def apply_exists_right(s: Sequent, terms: set, counter: int):
    for f in s.right:
        if isinstance(f, Exists):
            if terms:
                t = next(iter(terms))
            else:
                t = Constant(f"c{counter}")
                counter += 1
                terms.add(t)
            instantiated = substitute(f.body, f.var, t)
            return Sequent(s.left, tuple(list(s.right) + [instantiated])), counter
    return None, counter

def apply_non_branching_rule(s: Sequent, terms: set, counter: int):
    for rule in [apply_and_left, apply_or_right, apply_implies_right,
                 apply_not_left, apply_not_right]:
        result = rule(s)
        if result:
            return result, counter

    res, counter = apply_forall_right(s, terms, counter)
    if res: return res, counter

    res, counter = apply_exists_left(s, terms, counter)
    if res: return res, counter

    # Only attempt ∀L/∃R if no branching rule applies (Algorithm 2 priority 4/5)
    if apply_branching_rule(s) is not None:
        return None, counter

    res, counter = apply_forall_left(s, terms, counter)
    if res: return res, counter

    res, counter = apply_exists_right(s, terms, counter)
    if res: return res, counter

    return None, counter

def apply_branching_rule(s: Sequent) -> Optional[Tuple[Sequent, Sequent]]:
    for rule in [apply_and_right, apply_or_left, apply_implies_left]:
        result = rule(s)
        if result is not None:
            return result
    return None


def prove(formula: Formula, max_steps: int = 1000, debug: bool = False) -> bool:
    initial = Sequent((), (formula,))
    branches = [Branch([initial])]
    terms = collect_constants(formula)
    counter = 0
    steps = 0

    while branches and steps < max_steps:
        open_branch = next((b for b in branches if not b.closed and not b.failed), None)
        if open_branch is None:
            break

        current = open_branch.top()

        if is_trivial(current):
            open_branch.closed = True
            steps += 1
            continue

        next_seq, counter = apply_non_branching_rule(current, terms, counter)
        if next_seq is not None:
            open_branch.add(next_seq)
            steps += 1
            continue

        split = apply_branching_rule(current)
        if split is not None:
            s1, s2 = split
            parent_path = open_branch.sequents[:]
            open_branch.add(s1)
            branches.append(Branch(parent_path + [s2]))
            steps += 1
            continue

        open_branch.failed = True
        steps += 1

    return all(branch.closed for branch in branches)

if __name__ == "__main__":
    RUN_TESTS = False

    if RUN_TESTS:
        P_atom = Atom("P")
        Q_atom = Atom("Q")
        x = Variable("x")
        y = Variable("y")
        a = Constant("a")
        b = Constant("b")

        def P(t): return Predicate("P", (t,))
        def Q(t): return Predicate("Q", (t,))
        def R(t): return Predicate("R", (t,))

        tests = [
            # ── Tier 1: propositional (baseline should pass cleanly) ──────────────
            ("T1-01  P → P",
            Implies(P_atom, P_atom), True),

            ("T1-02  (P ∧ Q) → P",
            Implies(And(P_atom, Q_atom), P_atom), True),

            ("T1-03  P → (P ∨ Q)",
            Implies(P_atom, Or(P_atom, Q_atom)), True),

            ("T1-04  ¬¬P → P",
            Implies(Not(Not(P_atom)), P_atom), True),

            ("T1-05  P → ¬¬P",
            Implies(P_atom, Not(Not(P_atom))), True),

            ("T1-06  ((P → Q) ∧ P) → Q  (modus ponens)",
            Implies(And(Implies(P_atom, Q_atom), P_atom), Q_atom), True),

            ("T1-07  P → (Q → P)",
            Implies(P_atom, Implies(Q_atom, P_atom)), True),

            ("T1-08  ((P → Q) ∧ (Q → P_atom)) → (P ↔ Q)  (as ∧)",
            Implies(
                And(Implies(P_atom, Q_atom), Implies(Q_atom, P_atom)),
                And(Implies(P_atom, Q_atom), Implies(Q_atom, P_atom))
            ), True),

            ("T1-09  P ∧ ¬P  (unsatisfiable)",
            And(P_atom, Not(P_atom)), False),

            ("T1-10  (P ∨ ¬P)  (excluded middle)",
            Or(P_atom, Not(P_atom)), True),

            # ── Tier 2: simple FOL (baseline should pass) ─────────────────────────
            ("T2-01  ∀x P(x) → P(a)",
            Implies(ForAll(x, P(x)), P(a)), True),

            ("T2-02  P(a) → ∃x P(x)",
            Implies(P(a), Exists(x, P(x))), True),

            ("T2-03  ∃x P(x) → ∃x P(x)",
            Implies(Exists(x, P(x)), Exists(x, P(x))), True),

            ("T2-04  ∀x P(x) → ∀x P(x)",
            Implies(ForAll(x, P(x)), ForAll(x, P(x))), True),

            ("T2-05  P(a) → ∀x P(x)  (non-theorem)",
            Implies(P(a), ForAll(x, P(x))), False),

            ("T2-06  ∃x P(x) → ∀x P(x)  (non-theorem)",
            Implies(Exists(x, P(x)), ForAll(x, P(x))), False),

            # ── Tier 3: FOL requiring multiple instantiations ─────────────────────
            # Baseline may pass these but will wastefully instantiate with
            # every term in the shared pool rather than just what's needed
            ("T3-01  ∀x P(x) → P(a) ∧ P(b)",
            Implies(ForAll(x, P(x)), And(P(a), P(b))), True),

            ("T3-02  (∀x P(x)) ∧ (∀x Q(x)) → P(a) ∧ Q(a)",
            Implies(
                And(ForAll(x, P(x)), ForAll(x, Q(x))),
                And(P(a), Q(a))
            ), True),

            ("T3-03  ∀x (P(x) → Q(x)), P(a) → Q(a)  (universal instantiation)",
            Implies(
                And(ForAll(x, Implies(P(x), Q(x))), P(a)),
                Q(a)
            ), True),

            ("T3-04  ∃x P(x), ∀x (P(x) → Q(x)) → ∃x Q(x)",
            Implies(
                And(Exists(x, P(x)), ForAll(x, Implies(P(x), Q(x)))),
                Exists(x, Q(x))
            ), True),

            # ── Tier 4: cases the baseline cannot handle within max_steps ─────────
            # These are expected FAIL — your improvements should fix them
            ("T4-01  ∀x P(x) → P(a) ∧ P(b) ∧ P(c0)  (3 instantiations needed)",
            Implies(
                ForAll(x, P(x)),
                And(And(P(a), P(b)), Predicate("P", (Constant("c0"),)))
            ), True),

            ("T4-02  ∀x (P(x) ∨ Q(x)), ¬P(a) → Q(a)  (instantiate then branch)",
            Implies(
                And(ForAll(x, Or(P(x), Q(x))), Not(P(a))),
                Q(a)
            ), True),

            ("T4-03  ∀x ∀y (P(x) → P(y)) → (P(a) → P(b))  (nested quantifiers)",
            Implies(
                ForAll(x, ForAll(y, Implies(P(x), P(y)))),
                Implies(P(a), P(b))
            ), True),
        ]

        passed = 0
        failed = 0
        for name, formula, expected in tests:
            result = prove(formula)
            status = "PASS" if result == expected else "FAIL"
            if status == "PASS":
                passed += 1
            else:
                failed += 1
            print(f"[{status}] {name}")

        print(f"\n{passed}/{passed + failed} tests passed")
        print("Note: T4 failures are expected for the baseline —")
        print("      they exist to demonstrate where improvements are needed.")