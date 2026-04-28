# assignment1_baseline.py
#
# Baseline backward proof search prover for first-order logic.
# Implements Algorithm 2 from [1] over the sequent calculus LK'.
#
# Rule priority (following Algorithm 2):
#   1. Trivial closure (id, ⊤R, ⊥L)
#   2. Non-branching propositional rules (∧L, ∨R, →R, ¬L, ¬R)
#   3. Deterministic quantifier rules (∀R, ∃L) — fresh constant, consume formula
#   4. Branching rules (∧R, ∨L, →L)
#   5. Non-deterministic quantifier rules (∀L, ∃R) — instantiate with existing or fresh term
#
# Limitations (addressed in assignment1_improved.py):
#   - Shared term pool across all branches
#   - No used-terms tracking — same term may be re-instantiated repeatedly
#   - No duplicate sequent detection

from formulas import *


# ── Propositional non-branching rules ────────────────────────────────────────
# Each rule matches a formula on the appropriate side of the sequent,
# removes it, and returns the resulting sequent.

def apply_and_left(s: Sequent) -> Optional[Sequent]:
    """∧L: replace A ∧ B on the left with A, B."""
    for f in s.left:
        if isinstance(f, And):
            return Sequent(replace_one_with_many(s.left, f, [f.left, f.right]), s.right)
    return None

def apply_or_right(s: Sequent) -> Optional[Sequent]:
    """∨R: replace A ∨ B on the right with A, B."""
    for f in s.right:
        if isinstance(f, Or):
            return Sequent(s.left, replace_one_with_many(s.right, f, [f.left, f.right]))
    return None

def apply_implies_right(s: Sequent) -> Optional[Sequent]:
    """→R: move antecedent to left, consequent stays right."""
    for f in s.right:
        if isinstance(f, Implies):
            return Sequent(
                tuple(list(s.left) + [f.left]),
                tuple(list(remove_one(s.right, f)) + [f.right])
            )
    return None

def apply_not_left(s: Sequent) -> Optional[Sequent]:
    """¬L: move negated formula to the right."""
    for f in s.left:
        if isinstance(f, Not):
            return Sequent(remove_one(s.left, f), tuple(list(s.right) + [f.formula]))
    return None

def apply_not_right(s: Sequent) -> Optional[Sequent]:
    """¬R: move negated formula to the left."""
    for f in s.right:
        if isinstance(f, Not):
            return Sequent(tuple(list(s.left) + [f.formula]), remove_one(s.right, f))
    return None


# ── Branching rules ───────────────────────────────────────────────────────────
# Each rule splits the current sequent into two child sequents.

def apply_and_right(s: Sequent) -> Optional[Tuple[Sequent, Sequent]]:
    """∧R: split A ∧ B on the right into two branches, one for each conjunct."""
    for f in s.right:
        if isinstance(f, And):
            base = list(remove_one(s.right, f))
            return (Sequent(s.left, tuple(base + [f.left])),
                    Sequent(s.left, tuple(base + [f.right])))
    return None

def apply_or_left(s: Sequent) -> Optional[Tuple[Sequent, Sequent]]:
    """∨L: split A ∨ B on the left into two branches, one for each disjunct."""
    for f in s.left:
        if isinstance(f, Or):
            base = list(remove_one(s.left, f))
            return (Sequent(tuple(base + [f.left]), s.right),
                    Sequent(tuple(base + [f.right]), s.right))
    return None

def apply_implies_left(s: Sequent) -> Optional[Tuple[Sequent, Sequent]]:
    """→L: split A → B on the left into two branches — prove A or assume B."""
    for f in s.left:
        if isinstance(f, Implies):
            base = list(remove_one(s.left, f))
            return (Sequent(tuple(base), tuple(list(s.right) + [f.left])),
                    Sequent(tuple(base + [f.right]), s.right))
    return None


# ── Deterministic quantifier rules ───────────────────────────────────────────
# ∀R and ∃L always introduce a fresh constant and consume the formula.
# These fire before the non-deterministic ∀L/∃R rules.

def apply_forall_right(s: Sequent, terms: set, counter: int):
    """∀R: replace ∀x.A on the right with A[x/c] for a fresh constant c."""
    for f in s.right:
        if isinstance(f, ForAll):
            c = Constant(f"c{counter}")
            terms.add(c)
            return Sequent(s.left, replace_one_with_many(
                s.right, f, [substitute(f.body, f.var, c)])), counter + 1
    return None, counter

def apply_exists_left(s: Sequent, terms: set, counter: int):
    """∃L: replace ∃x.A on the left with A[x/c] for a fresh constant c."""
    for f in s.left:
        if isinstance(f, Exists):
            c = Constant(f"c{counter}")
            terms.add(c)
            return Sequent(replace_one_with_many(
                s.left, f, [substitute(f.body, f.var, c)]), s.right), counter + 1
    return None, counter


# ── Non-deterministic quantifier rules ───────────────────────────────────────
# ∀L and ∃R keep the quantified formula in the sequent so it can be
# instantiated with multiple terms. They use an existing term if available,
# otherwise generate a fresh constant.

def apply_forall_left(s: Sequent, terms: set, counter: int):
    """∀L: add A[x/t] to the left for some term t, keeping ∀x.A."""
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
    """∃R: add A[x/t] to the right for some term t, keeping ∃x.A."""
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


# ── Rule dispatch ─────────────────────────────────────────────────────────────

def apply_non_branching_rule(s: Sequent, terms: set, counter: int):
    """
    Apply the highest-priority applicable non-branching rule.
    Returns (sequent, counter) or (None, counter) if none applies.
    Priority: propositional → ∀R/∃L → (skip if branching rule exists) → ∀L/∃R
    """
    # Priority 2: propositional non-branching rules
    for rule in [apply_and_left, apply_or_right, apply_implies_right,
                 apply_not_left, apply_not_right]:
        result = rule(s)
        if result:
            return result, counter

    # Priority 2: deterministic quantifier rules
    res, counter = apply_forall_right(s, terms, counter)
    if res: return res, counter

    res, counter = apply_exists_left(s, terms, counter)
    if res: return res, counter

    # Priority 4/5: only attempt ∀L/∃R if no branching rule applies
    if apply_branching_rule(s) is not None:
        return None, counter

    res, counter = apply_forall_left(s, terms, counter)
    if res: return res, counter

    res, counter = apply_exists_right(s, terms, counter)
    if res: return res, counter

    return None, counter

def apply_branching_rule(s: Sequent) -> Optional[Tuple[Sequent, Sequent]]:
    """
    Apply the highest-priority applicable branching rule.
    Returns (s1, s2) or None if none applies.
    """
    for rule in [apply_and_right, apply_or_left, apply_implies_left]:
        result = rule(s)
        if result is not None:
            return result
    return None


# ── Main proof search loop ────────────────────────────────────────────────────

def prove(formula: Formula, max_steps: int = 1000, debug: bool = False) -> bool:
    """
    Attempt to prove a formula using backward proof search.

    Returns True if all branches close, False if any branch fails or
    the step limit is reached. The step limit is necessary because
    first-order logic is only semi-decidable.

    Args:
        formula: The formula to prove.
        max_steps: Maximum number of rule applications before giving up.
        debug: If True, print each step to stdout.
    """
    initial = Sequent((), (formula,))
    branches = [Branch([initial])]

    # Shared term pool and fresh constant counter (baseline limitation)
    terms = collect_constants(formula)
    counter = 0
    steps = 0

    while branches and steps < max_steps:
        # Select the first open branch (depth-first, left-to-right)
        open_branch = next((b for b in branches if not b.closed and not b.failed), None)
        if open_branch is None:
            break

        current = open_branch.top()

        if debug:
            print(f"Step {steps + 1}: {current}")

        # Priority 1: trivial closure
        if is_trivial(current):
            if debug:
                print("  Closed by trivial rule")
            open_branch.closed = True
            steps += 1
            continue

        # Priority 2-5: non-branching rules
        next_seq, counter = apply_non_branching_rule(current, terms, counter)
        if next_seq is not None:
            if debug:
                print(f"  Applied non-branching rule -> {next_seq}")
            open_branch.add(next_seq)
            steps += 1
            continue

        # Priority 3: branching rules
        split = apply_branching_rule(current)
        if split is not None:
            s1, s2 = split
            if debug:
                print(f"  Applied branching rule -> {s1}  |  {s2}")
            parent_path = open_branch.sequents[:]
            open_branch.add(s1)
            # Each new branch starts from the same parent path
            branches.append(Branch(parent_path + [s2]))
            steps += 1
            continue

        if debug:
            print("  No rule applies, branch failed")
        open_branch.failed = True
        steps += 1

    # Proof succeeds only if every branch was closed
    return all(branch.closed for branch in branches)


if __name__ == "__main__":
    # Set to True to run the built-in test suite
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
            # ── Tier 1: propositional ──────────────
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

            # ── Tier 2: simple FOL ─────────────────────────
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

            # ── Tier 3: ─────────────────────
            
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

            # ── Tier 4:  ─────────
            
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
