# assignment1_improved.py
#
# Improved backward proof search prover for first-order logic.
# Extends the baseline (Algorithm 2 from [1]) with the following improvements:
#
#   1. Branch-local state — each branch maintains its own terms, counter,
#      and used-terms record, preventing cross-branch contamination.
#
#   2. Rule prioritisation and delayed quantifier instantiation — branching
#      rules are prioritised over ∀L/∃R, and used-terms tracking prevents
#      repeated identical instantiations.
#
#   3. Goal-directed term selection — terms already present in the current
#      sequent are preferred when instantiating quantified formulas.
#
#   4. Duplicate detection — sequents are normalised and stored per branch
#      to prevent revisiting identical states.
#
#   5. Branch selection heuristic — the smallest open branch is processed
#      first, closing simpler goals before tackling complex ones.

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from formulas import *


# ── Improved Branch ───────────────────────────────────────────────────────────
# Overrides the shared-state Branch from formulas.py with branch-local state.
# Each branch independently tracks its terms, counter, used instantiations,
# and previously seen sequents.

@dataclass
class Branch:
    sequents: List[Sequent]
    closed: bool = False
    failed: bool = False
    terms: set = field(default_factory=set)          # constants known on this branch
    counter: int = 0                                  # fresh constant counter
    used_terms: dict = field(default_factory=dict)   # formula -> set of terms tried
    seen: set = field(default_factory=set)            # sequents visited on this branch

    def __post_init__(self):
        # Initialise seen set from the starting sequents if not provided
        if not self.seen:
            self.seen = set(self.sequents)

    def top(self) -> Sequent:
        """Return the current sequent at the top of this branch."""
        return self.sequents[-1]

    def add(self, sequent: Sequent) -> bool:
        """
        Normalise and add a sequent to this branch.
        Returns False if the sequent has already been seen (duplicate detection),
        True otherwise.
        """
        sequent = normalize_sequent(sequent)
        if sequent in self.seen:
            return False
        self.sequents.append(sequent)
        self.seen.add(sequent)
        return True


# ── Improvement helpers ───────────────────────────────────────────────────────

def normalize_sequent(sequent: Sequent) -> Sequent:
    """
    Normalise a sequent by sorting and deduplicating both sides.
    This ensures that sequents which are logically identical but
    structurally different (e.g. different formula orderings) are
    treated as the same state by duplicate detection.
    """
    left = tuple(sorted(set(sequent.left), key=str))
    right = tuple(sorted(set(sequent.right), key=str))
    return Sequent(left, right)

def collect_terms_from_sequent(sequent: Sequent) -> set:
    """Collect all constants currently appearing anywhere in the sequent."""
    terms = set()
    for formula in sequent.left:
        terms |= collect_constants(formula)
    for formula in sequent.right:
        terms |= collect_constants(formula)
    return terms

def choose_term_for_instantiation(sequent: Sequent, available_terms: set, global_terms: set):
    """
    Goal-directed term selection.
    Prefer terms already visible in the current sequent over globally known
    terms, increasing the likelihood that the instantiation is relevant.
    """
    sequent_terms = collect_terms_from_sequent(sequent)

    for term in sequent_terms:
        if term in available_terms:
            return term

    for term in global_terms:
        if term in available_terms:
            return term

    return None

def formula_mentions_target(formula: Formula, target_terms: set) -> bool:
    """Return True if the formula contains any of the target constants."""
    return bool(collect_constants(formula) & target_terms)

def branch_score(branch: Branch) -> int:
    """
    Branch selection heuristic — score a branch by the size of its
    current sequent. Smaller scores are processed first, closing
    simpler branches before tackling complex ones.
    """
    s = branch.top()
    return len(s.left) + len(s.right)

def short_side(formulas, limit=5):
    """Truncate a list of formulas for compact debug output."""
    items = [str(f) for f in formulas[:limit]]
    if len(formulas) > limit:
        items.append(f"... (+{len(formulas) - limit} more)")
    return ", ".join(items) if items else "·"

def short_sequent(sequent, limit=5):
    """Format a sequent compactly for debug output."""
    left = short_side(list(sequent.left), limit)
    right = short_side(list(sequent.right), limit)
    return f"{left} ⊢ {right}"

def fresh_constant(counter: int) -> Constant:
    """Generate a fresh constant with a unique name."""
    return Constant(f"c{counter}")


# ── Propositional non-branching rules ────────────────────────────────────────
# Unchanged from the baseline — these have no interaction with the improvements.

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

def apply_forall_right(sequent: Sequent, terms: set, counter: int):
    """∀R: replace ∀x.A on the right with A[x/c] for a fresh constant c."""
    for f in sequent.right:
        if isinstance(f, ForAll):
            new_const = fresh_constant(counter)
            terms.add(new_const)
            instantiated = substitute(f.body, f.var, new_const)
            new_right = replace_one_with_many(sequent.right, f, [instantiated])
            return Sequent(sequent.left, new_right), counter + 1
    return None, counter

def apply_exists_left(sequent: Sequent, terms: set, counter: int):
    """∃L: replace ∃x.A on the left with A[x/c] for a fresh constant c."""
    for f in sequent.left:
        if isinstance(f, Exists):
            new_const = fresh_constant(counter)
            terms.add(new_const)
            instantiated = substitute(f.body, f.var, new_const)
            new_left = replace_one_with_many(sequent.left, f, [instantiated])
            return Sequent(new_left, sequent.right), counter + 1
    return None, counter


# ── Non-deterministic quantifier rules (improved) ────────────────────────────
# These differ from the baseline in two ways:
#   - Used-terms tracking prevents re-instantiation with the same term
#   - Goal-directed term selection prefers terms already in the sequent

def apply_forall_left(
    sequent: Sequent, terms: set, used_terms: dict, counter: int
) -> Tuple[Optional[Sequent], int]:
    """
    ∀L: add A[x/t] to the left for some term t, keeping ∀x.A.
    Selects the most relevant unused term via goal-directed selection.
    Falls back to a fresh constant if all existing terms are exhausted.
    """
    # Collect constants from the right side to guide term selection
    target_terms = set()
    for f in sequent.right:
        target_terms |= collect_constants(f)

    best_choice = None

    for f in sequent.left:
        if isinstance(f, ForAll):
            already_used = used_terms.get(f, set())
            available = terms - already_used

            t = choose_term_for_instantiation(sequent, available, terms)
            fresh_needed = False

            if t is None:
                # All existing terms exhausted — generate a fresh constant
                t = fresh_constant(counter)
                fresh_needed = True

            instantiated = substitute(f.body, f.var, t)
            # Score this instantiation by whether it mentions a goal term
            score = 1 if formula_mentions_target(instantiated, target_terms) else 0

            if best_choice is None or score > best_choice[0]:
                best_choice = (score, f, t, instantiated, fresh_needed)

    if best_choice is None:
        return None, counter

    _, f, t, instantiated, fresh_needed = best_choice

    if fresh_needed:
        counter += 1
        terms.add(t)

    used_terms.setdefault(f, set()).add(t)
    new_left = tuple(list(sequent.left) + [instantiated])
    return Sequent(new_left, sequent.right), counter


def apply_exists_right(
    sequent: Sequent, terms: set, used_terms: dict, counter: int
) -> Tuple[Optional[Sequent], int]:
    
    # Collect constants from the left side to guide term selection
    target_terms = set()
    for f in sequent.left:
        target_terms |= collect_constants(f)

    best_choice = None

    for f in sequent.right:
        if isinstance(f, Exists):
            already_used = used_terms.get(f, set())
            available = terms - already_used

            t = choose_term_for_instantiation(sequent, available, terms)
            fresh_needed = False

            if t is None:
                t = fresh_constant(counter)
                fresh_needed = True

            instantiated = substitute(f.body, f.var, t)
            score = 1 if formula_mentions_target(instantiated, target_terms) else 0

            if best_choice is None or score > best_choice[0]:
                best_choice = (score, f, t, instantiated, fresh_needed)

    if best_choice is None:
        return None, counter

    _, f, t, instantiated, fresh_needed = best_choice

    if fresh_needed:
        counter += 1
        terms.add(t)

    used_terms.setdefault(f, set()).add(t)
    new_right = tuple(list(sequent.right) + [instantiated])
    return Sequent(sequent.left, new_right), counter


# ── Rule dispatch ─────────────────────────────────────────────────────────────

def apply_non_branching_rule(sequent, terms, counter, used_terms):
    """
    Apply the highest-priority applicable non-branching rule.
    Returns (rule_name, sequent, counter) or (None, None, counter).

    Priority:
      1. Propositional non-branching rules
      2. Deterministic quantifier rules (∀R, ∃L)
      3. Skip if a branching rule applies (delay ∀L/∃R)
      4. Non-deterministic quantifier rules (∀L, ∃R)
    """
    # Priority 1: propositional non-branching rules
    for name, rule in [
        ("∧L", apply_and_left),
        ("∨R", apply_or_right),
        ("→R", apply_implies_right),
        ("¬L", apply_not_left),
        ("¬R", apply_not_right),
    ]:
        result = rule(sequent)
        if result is not None:
            return name, result, counter

    # Priority 2: deterministic quantifier rules
    res, counter = apply_forall_right(sequent, terms, counter)
    if res is not None:
        return "∀R", res, counter

    res, counter = apply_exists_left(sequent, terms, counter)
    if res is not None:
        return "∃L", res, counter

    # Priority 3: defer to branching rules if one is available
    branch_rule_name, split = apply_branching_rule(sequent)
    if split is not None:
        return None, None, counter

    # Priority 4: non-deterministic instantiation as last resort
    res, counter = apply_forall_left(sequent, terms, used_terms, counter)
    if res is not None:
        return "∀L", res, counter

    res, counter = apply_exists_right(sequent, terms, used_terms, counter)
    if res is not None:
        return "∃R", res, counter

    return None, None, counter

def apply_branching_rule(sequent: Sequent):
    """
    Apply the highest-priority applicable branching rule.
    Returns (rule_name, (s1, s2)) or (None, None).
    """
    for name, rule in [
        ("∧R", apply_and_right),
        ("∨L", apply_or_left),
        ("→L", apply_implies_left),
    ]:
        result = rule(sequent)
        if result is not None:
            return name, result
    return None, None


# ── Main proof search loop ────────────────────────────────────────────────────

def prove(formula: Formula, max_steps: int = 1000, debug: bool = False) -> bool:
    """
    Attempt to prove a formula using improved backward proof search.

    Improvements over the baseline:
      - Branch-local state (terms, counter, used_terms, seen)
      - Goal-directed term selection
      - Duplicate sequent detection
      - Branch selection heuristic (smallest branch first)

    Returns True if all branches close, False otherwise.

    Args:
        formula: The formula to prove.
        max_steps: Maximum rule applications before giving up.
        debug: If True, print each step to stdout.
    """
    initial = Sequent((), (formula,))
    initial_terms = collect_constants(formula)

    # Each branch gets its own independent copy of state
    branches = [
        Branch(
            sequents=[initial],
            terms=set(initial_terms),
            counter=0,
            used_terms={}
        )
    ]

    steps = 0

    while branches and steps < max_steps:
        # Branch selection heuristic — process smallest open branch first
        open_branch = min(
            (b for b in branches if not b.closed and not b.failed),
            key=branch_score,
            default=None
        )

        if open_branch is None:
            break

        current = open_branch.top()

        if debug:
            print(f"Step {steps + 1}: {short_sequent(current)}")

        # Priority 1: trivial closure
        if is_trivial(current):
            if debug:
                print("  Closed by trivial rule")
            open_branch.closed = True
            steps += 1
            continue

        # Priority 2-4: non-branching rules
        rule_name, next_seq, new_counter = apply_non_branching_rule(
            current,
            open_branch.terms,
            open_branch.counter,
            open_branch.used_terms
        )

        if next_seq is not None:
            if not open_branch.add(next_seq):
                # Duplicate detected — this branch is cycling, mark as failed
                if debug:
                    print(f"  {rule_name} produced duplicate sequent, branch failed")
                open_branch.failed = True
            else:
                open_branch.counter = new_counter
                if debug:
                    print(f"  Applied {rule_name} -> {short_sequent(next_seq)}")
            steps += 1
            continue

        # Priority 3: branching rules
        branch_rule_name, split = apply_branching_rule(current)
        if split is not None:
            s1, s2 = split

            if debug:
                print(f"  Applied {branch_rule_name} -> {short_sequent(s1)}  |  {short_sequent(s2)}")

            # Snapshot parent state before modifying current branch
            parent_path = open_branch.sequents[:]
            parent_seen = set(open_branch.seen)
            parent_terms = set(open_branch.terms)
            parent_counter = open_branch.counter
            parent_used_terms = {k: set(v) for k, v in open_branch.used_terms.items()}

            # Left child continues on the current branch
            added_left = open_branch.add(s1)
            if not added_left:
                if debug:
                    print("  Left branch duplicate, marking failed")
                open_branch.failed = True
            else:
                # Restore parent state so left branch is independent
                open_branch.terms = set(parent_terms)
                open_branch.counter = parent_counter
                open_branch.used_terms = {k: set(v) for k, v in parent_used_terms.items()}

            # Right child becomes a new independent branch
            if s2 not in parent_seen:
                new_branch = Branch(
                    sequents=parent_path + [s2],
                    terms=set(parent_terms),
                    counter=parent_counter,
                    used_terms={k: set(v) for k, v in parent_used_terms.items()},
                    seen=parent_seen | {s2},
                )
                branches.append(new_branch)
            elif debug:
                print("  Right branch duplicate, skipped")

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
    
