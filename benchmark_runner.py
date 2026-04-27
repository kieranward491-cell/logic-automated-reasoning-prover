import time
from parser import parse_file
from assignment1_baseline import prove as baseline_prove
from assignment1_improved import prove as improved_prove
from formulas import *

def run_benchmark(file_path: str, prover, prover_name: str, max_steps: int = 1000):
    formulas = parse_file(file_path)

    solved = 0
    total = len(formulas)
    results = []

    start_time = time.perf_counter()

    for i, formula in enumerate(formulas, start=1):
        formula_start = time.perf_counter()
        try:
            result = prover(formula, max_steps=max_steps, debug=False)
        except TypeError:
            # fallback in case baseline prover doesn't support debug argument
            result = prover(formula, max_steps=max_steps)
        formula_end = time.perf_counter()

        elapsed = formula_end - formula_start
        if result:
            solved += 1

        results.append((i, result, elapsed, formula))

    end_time = time.perf_counter()
    total_time = end_time - start_time

    print(f"\n=== {prover_name} on {file_path} ===")
    print(f"Solved: {solved}/{total}")
    print(f"Total time: {total_time:.6f} seconds")
    print(f"Average time per formula: {total_time / total:.6f} seconds")

    print("\nPer-formula results:")
    for i, result, elapsed, formula in results:
        status = "PASS" if result else "FAIL"
        print(f"[{status}] #{i:02d} ({elapsed:.6f}s)  {formula}")

    return {
        "file": file_path,
        "prover": prover_name,
        "solved": solved,
        "total": total,
        "total_time": total_time,
        "avg_time": total_time / total if total > 0 else 0,
        "details": results,
    }


if __name__ == "__main__":

    # Quick sanity check
    from assignment1_baseline import (Atom, Implies)
    test = Implies(Atom("P"), Atom("P"))
    print("Baseline sanity:", baseline_prove(test))
    print("Improved sanity:", improved_prove(test))

    benchmark_files = [
        "benchmark_easy.txt",
        "benchmark_medium.txt",
        "benchmark_hard.txt",
    ]

    summary = []

    for file_path in benchmark_files:
        summary.append(run_benchmark(file_path, baseline_prove, "Baseline"))
        summary.append(run_benchmark(file_path, improved_prove, "Improved"))

    print("\n=== Summary Table ===")
    print(f"{'Dataset':<22} {'Prover':<10} {'Solved':<10} {'Total Time (s)':<15} {'Avg Time (s)':<15}")
    print("-" * 75)

    for row in summary:
        print(
            f"{row['file']:<22} "
            f"{row['prover']:<10} "
            f"{row['solved']}/{row['total']:<8} "
            f"{row['total_time']:<15.6f} "
            f"{row['avg_time']:<15.6f}"
        )