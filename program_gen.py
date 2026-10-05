"""Synthetic 'compiled function' generator.

A function is a linear sequence of N instruction slots with nested loops.
Each virtual register becomes a live interval [start, end) with a list of
use positions. Spill cost = sum over defs/uses of 10**loop_depth(position),
which mirrors the block-frequency weighting real compilers use.

The JSON schema produced here is the same one a real LLVM front end would
emit (see README), so real data can be dropped in instead.
"""
import json, random
from dataclasses import dataclass, asdict


@dataclass
class Interval:
    vreg: int
    start: int
    end: int            # exclusive
    uses: list          # positions of uses (def is at `start`)
    cost: float = 0.0   # frequency-weighted spill cost
    max_depth: int = 0
    crosses_call: bool = False


@dataclass
class Function:
    name: str
    n: int
    loops: list         # (start, end, depth)
    calls: list
    intervals: list


def depth_at(loops, p):
    return sum(1 for s, e, _ in loops if s <= p < e)


def gen_function(rng, name):
    n = rng.randint(80, 400)
    # nested loops
    loops = []
    for _ in range(rng.randint(1, 4)):
        s = rng.randint(0, n - 20)
        e = min(n, s + rng.randint(15, n // 2))
        loops.append((s, e, 0))
    calls = sorted(rng.sample(range(n), rng.randint(0, max(1, n // 25))))
    style = rng.choice(["compute", "memory", "mixed"])  # shifts live-range mix
    ivs = []
    n_v = int(n * rng.uniform(0.6, 1.2))
    for v in range(n_v):
        kind = rng.random()
        long_p = {"compute": 0.12, "memory": 0.25, "mixed": 0.18}[style]
        if kind < long_p and loops:       # loop-invariant / loop-spanning value
            ls, le, _ = rng.choice(loops)
            start = max(0, ls - rng.randint(0, 15))
            end = min(n, le + rng.randint(0, 15))
        elif kind < long_p + 0.25:        # medium
            start = rng.randint(0, n - 2)
            end = min(n, start + rng.randint(10, 50))
        else:                             # short temporary
            start = rng.randint(0, n - 2)
            end = min(n, start + rng.randint(2, 9))
        if end - start < 2:
            end = min(n, start + 2)
            start = end - 2
        k = max(1, int(rng.expovariate(1 / 2.5)))
        uses = sorted(rng.randint(start + 1, end - 1) for _ in range(k))
        # uses cluster in loops with higher probability for hot values
        cost = float(10 ** depth_at(loops, start))
        cost += sum(10 ** depth_at(loops, u) for u in uses)
        md = max([depth_at(loops, p) for p in [start] + uses])
        cc = any(start < c < end for c in calls)
        ivs.append(Interval(v, start, end, uses, cost, md, cc))
    return Function(name, n, loops, calls, ivs)


def generate(num, seed=0):
    rng = random.Random(seed)
    return [gen_function(rng, f"fn{i:03d}") for i in range(num)]


def save(funcs, path):
    with open(path, "w") as f:
        json.dump([asdict(x) for x in funcs], f)


def load(path):
    with open(path) as f:
        raw = json.load(f)
    return [Function(r["name"], r["n"], [tuple(l) for l in r["loops"]], r["calls"],
                     [Interval(**i) for i in r["intervals"]]) for r in raw]
