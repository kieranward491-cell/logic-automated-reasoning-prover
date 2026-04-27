# First-Order Logic Prover

This project implements a backward proof search system for first-order logic using a sequent calculus approach.

## Features

- Baseline prover (naive backward search)
- Improved prover with:
  - branch-local state
  - goal-directed term selection
  - duplicate detection
  - branch prioritisation
- Custom parser for first-order logic formulas
- Benchmarking system with easy, medium, and hard datasets

## How to Run

Run the benchmark:

```bash
python benchmark_runner.py
