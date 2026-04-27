from dataclasses import dataclass, field
from multiprocessing.util import debug
from shlex import split
from typing import List, Optional, Tuple, Union
from formulas import *

@dataclass
class Branch:
    sequents: List[Sequent]
    closed: bool = False
    failed: bool = False
    terms: set = field(default_factory=set)
    counter: int = 0
    used_terms: dict = field(default_factory=dict)
    seen: set = field(default_factory=set)

    def __post_init__(self):
        if not self.seen:
            self.seen = set(self.sequents)

    def top(self) -> Sequent:
        return self.sequents[-1]

    def add(self, sequent: Sequent) -> bool:
        sequent = normalize_sequent(sequent)
        if sequent in self.seen:
            return False
        self.sequents.append(sequent)
        self.seen.add(sequent)
        return True

def fresh_constant(counter: int) -> Constant:
    return Constant(f"c{counter}")



def short_side(formulas, limit=5):
    items = [str(f) for f in formulas[:limit]]
    if len(formulas) > limit:
        items.append(f"... (+{len(formulas) - limit} more)")
    return ", ".join(items) if items else "·"

def short_sequent(sequent, limit=5):
    left = short_side(list(sequent.left), limit)
    right = short_side(list(sequent.right), limit)
    return f"{left} ⊢ {right}"

def collect_terms_from_sequent(sequent: Sequent) -> set:
    """Collect all constants currently appearing anywhere in the sequent."""
    terms = set()
    for formula in sequent.left:
        terms |= collect_constants(formula)
    for formula in sequent.right:
        terms |= collect_constants(formula)
    return terms


def choose_term_for_instantiation(
    sequent: Sequent,
    available_terms: set,
    global_terms: set
):
    """
    Prefer terms already visible in the current sequent.
    If none are available, fall back to any known global term.
    """
    sequent_terms = collect_terms_from_sequent(sequent)

    for term in sequent_terms:
        if term in available_terms:
            return term

    for term in global_terms:
        if term in available_terms:
            return term

    return None

def branch_score(branch: Branch) -> int:
    s = branch.top()
    return len(s.left) + len(s.right)

def formula_mentions_target(formula: Formula, target_terms: set) -> bool:
    return bool(collect_constants(formula) & target_terms)

def normalize_sequent(sequent: Sequent) -> Sequent:
    left = tuple(sorted(set(sequent.left), key=str))
    right = tuple(sorted(set(sequent.right), key=str))
    return Sequent(left, right)


def apply_and_left(sequent: Sequent) -> Optional[Sequent]:
    for f in sequent.left:
        if isinstance(f, And):
            new_left = replace_one_with_many(sequent.left, f, [f.left, f.right])
            return Sequent(new_left, sequent.right)
    return None


def apply_or_right(sequent: Sequent) -> Optional[Sequent]:
    for f in sequent.right:
        if isinstance(f, Or):
            new_right = replace_one_with_many(sequent.right, f, [f.left, f.right])
            return Sequent(sequent.left, new_right)
    return None


def apply_implies_right(sequent: Sequent) -> Optional[Sequent]:
    for f in sequent.right:
        if isinstance(f, Implies):
            new_right = remove_one(sequent.right, f)
            new_right = tuple(list(new_right) + [f.right])
            new_left = tuple(list(sequent.left) + [f.left])
            return Sequent(new_left, new_right)
    return None


def apply_not_left(sequent: Sequent) -> Optional[Sequent]:
    for f in sequent.left:
        if isinstance(f, Not):
            new_left = remove_one(sequent.left, f)
            new_right = tuple(list(sequent.right) + [f.formula])
            return Sequent(new_left, new_right)
    return None


def apply_not_right(sequent: Sequent) -> Optional[Sequent]:
    for f in sequent.right:
        if isinstance(f, Not):
            new_right = remove_one(sequent.right, f)
            new_left = tuple(list(sequent.left) + [f.formula])
            return Sequent(new_left, new_right)
    return None


def apply_and_right(sequent: Sequent) -> Optional[Tuple[Sequent, Sequent]]:
    for f in sequent.right:
        if isinstance(f, And):
            base_right = list(remove_one(sequent.right, f))
            s1 = Sequent(sequent.left, tuple(base_right + [f.left]))
            s2 = Sequent(sequent.left, tuple(base_right + [f.right]))
            return s1, s2
    return None


def apply_or_left(sequent: Sequent) -> Optional[Tuple[Sequent, Sequent]]:
    for f in sequent.left:
        if isinstance(f, Or):
            base_left = list(remove_one(sequent.left, f))
            s1 = Sequent(tuple(base_left + [f.left]), sequent.right)
            s2 = Sequent(tuple(base_left + [f.right]), sequent.right)
            return s1, s2
    return None


def apply_implies_left(sequent: Sequent) -> Optional[Tuple[Sequent, Sequent]]:
    for f in sequent.left:
        if isinstance(f, Implies):
            base_left = list(remove_one(sequent.left, f))
            s1 = Sequent(tuple(base_left), tuple(list(sequent.right) + [f.left]))
            s2 = Sequent(tuple(base_left + [f.right]), sequent.right)
            return s1, s2
    return None

def apply_forall_left(
    sequent: Sequent,
    terms: set,
    used_terms: dict,
    counter: int
) -> Tuple[Optional[Sequent], int]:
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
    new_left = tuple(list(sequent.left) + [instantiated])
    return Sequent(new_left, sequent.right), counter


def apply_exists_right(
    sequent: Sequent,
    terms: set,
    used_terms: dict,
    counter: int
) -> Tuple[Optional[Sequent], int]:
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


def apply_forall_right(sequent: Sequent, terms: set, counter: int):
    for f in sequent.right:
        if isinstance(f, ForAll):
            new_const = fresh_constant(counter)
            terms.add(new_const)
            instantiated = substitute(f.body, f.var, new_const)
            new_right = replace_one_with_many(sequent.right, f, [instantiated])
            return Sequent(sequent.left, new_right), counter + 1
    return None, counter


def apply_exists_left(sequent: Sequent, terms: set, counter: int):
    for f in sequent.left:
        if isinstance(f, Exists):
            new_const = fresh_constant(counter)
            terms.add(new_const)
            instantiated = substitute(f.body, f.var, new_const)
            new_left = replace_one_with_many(sequent.left, f, [instantiated])
            return Sequent(new_left, sequent.right), counter + 1
    return None, counter

def apply_non_branching_rule(sequent, terms, counter, used_terms):
    # 1. Propositional non-branching rules first
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

    # 2. Deterministic quantifier rules next
    res, counter = apply_forall_right(sequent, terms, counter)
    if res is not None:
        return "∀R", res, counter

    res, counter = apply_exists_left(sequent, terms, counter)
    if res is not None:
        return "∃L", res, counter

    # 3. If a branching rule is available, do NOT instantiate ∀L / ∃R yet
    branch_rule_name, split = apply_branching_rule(sequent)
    if split is not None:
        return None, None, counter

    # 4. Only now try non-deterministic instantiation rules
    res, counter = apply_forall_left(sequent, terms, used_terms, counter)
    if res is not None:
        return "∀L", res, counter

    res, counter = apply_exists_right(sequent, terms, used_terms, counter)
    if res is not None:
        return "∃R", res, counter

    return None, None, counter

def apply_branching_rule(sequent: Sequent):
    for name, rule in [
        ("∧R", apply_and_right),
        ("∨L", apply_or_left),
        ("→L", apply_implies_left),
    ]:
        result = rule(sequent)
        if result is not None:
            return name, result
    return None, None


def prove(formula: Formula, max_steps: int = 1000, debug: bool = False) -> bool:
    initial = Sequent((), (formula,))
    initial_terms = collect_constants(formula)

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

        # Trivial closing rules
        if is_trivial(current):
            if debug:
                print("  Closed by trivial rule")
            open_branch.closed = True
            steps += 1
            continue

        # Non-branching rules
        rule_name, next_seq, new_counter = apply_non_branching_rule(
            current,
            open_branch.terms,
            open_branch.counter,
            open_branch.used_terms
        )

        if next_seq is not None:
            if not open_branch.add(next_seq):
                if debug:
                    print(f"  {rule_name} produced duplicate sequent, branch failed")
                open_branch.failed = True
            else:
                open_branch.counter = new_counter
                if debug:
                    print(f"  Applied {rule_name} -> {short_sequent(next_seq)}")
            steps += 1
            continue

        # Branching rules
        branch_rule_name, split = apply_branching_rule(current)
        if split is not None:
            s1, s2 = split

            if debug:
                print(f"  Applied {branch_rule_name} -> {short_sequent(s1)}   |   {short_sequent(s2)}")

            parent_path = open_branch.sequents[:]
            parent_seen = set(open_branch.seen)
            parent_terms = set(open_branch.terms)
            parent_counter = open_branch.counter
            parent_used_terms = {k: set(v) for k, v in open_branch.used_terms.items()}

            # Left child stays on current branch
            added_left = open_branch.add(s1)

            if not added_left:
                if debug:
                    print("  Left branch duplicate, marking current branch failed")
                open_branch.failed = True
            else:
                open_branch.terms = set(parent_terms)
                open_branch.counter = parent_counter
                open_branch.used_terms = {k: set(v) for k, v in parent_used_terms.items()}

            # Right child becomes a new branch
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
    