# Project 13 - Learning-Based Register Allocation

Train a model that decides **which live ranges lose their register** when
register pressure exceeds K, and compare it against classic heuristics and the
provable optimum.

## Run
```bash
pip install -r requirements.txt
python run_experiment.py        # ~1-2 min, writes everything to results/
```

## Pipeline
| Step | File | What it does |
|---|---|---|
| 1 | `program_gen.py` | Generates 300 synthetic functions (nested loops, calls, long/medium/short live ranges). Spill cost = sum of 10^loop_depth over defs/uses. |
| 2 | `allocators.py` | Interference graph, Chaitin-Briggs colouring with pluggable spill score, linear scan, LLVM-style weight, and an **exact ILP optimum** (interval graphs: LP is integral). |
| 3 | `features.py` | 19 features per live range: length, uses, loop depth, degree, crosses-call, local pressure, cost relative to neighbours, K, ... |
| 4 | `run_experiment.py` | Labels = "kept in optimal allocation". Trains Gradient Boosting on 200 functions, tests on 100 *unseen* functions with K = 4, 8, 16. |

The ML model plugs into the allocator through `spill_score = 1 - P(keep)`; it
is consulted only when the simplify phase gets stuck (no node of degree < K).

## Results (100 unseen test functions, from `results/summary.csv`)
Average weighted spill cost (lower is better):

| K | Linear scan | Chaitin deg/cost | LLVM-style | **ML (GBM)** | Optimal |
|---|---|---|---|---|---|
| 4 | 28899 | 27581 | 27100 | **27118** | 26381 |
| 8 | 20674 | 19252 | 18595 | **18595** | 17836 |
| 16 | 14068 | 10784 | 11264 | **10925** | 10527 |

Takeaways (honest version):
* The ML allocator clearly beats linear scan and is **on par with or slightly better than Chaitin's heuristic in total cost at K=4 and K=8**, and matches the LLVM-style weight; at K=16 Chaitin is still marginally better.
* It does **not** consistently beat the best heuristic per function (strictly better on ~25% of cases). All colouring-based methods are within ~3-9% of optimal, so there is little headroom.
* Feature importance (`results/feature_importance.png`) shows neighbour-relative cost and local pressure matter most.

## Limitations (put these in your report)
* Programs are **synthetic**; no real compiler front end (no clang in the build sandbox).
* No spill-code insertion / re-allocation, no register classes, no live-range splitting, no coalescing, no calling conventions beyond a `crosses_call` feature.
* Labels come from a global optimum while decisions are made greedily, so the model learns an approximation.

## Future work
* Replace the generator with real intervals from LLVM (`llc -print-after=liveintervals`, or `llvmlite`) and emit the same JSON schema (see `program_gen.py`).
* Integrate with LLVM's MLGO register-allocation priority advisor.
* Train with reinforcement learning or imitation of the greedy trajectory instead of per-node labels.
* Add register classes, live-range splitting and runtime measurements (`perf stat`).

## Outputs in `results/`
`summary.csv`, `eval_raw.csv`, `dataset_train.csv`, `model.joblib`, `programs.json`,
`spill_cost_comparison.png`, `spill_count_vs_K.png`, `feature_importance.png`
