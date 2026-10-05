"""Register allocators operating on live intervals.

Every allocator returns the set of spilled vreg ids. Spill code is not
rewritten (no spill temporaries), so results measure *which ranges lose
their register*, which is exactly the decision the ML model replaces.
"""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def overlaps(a, b):
    return a.start < b.end and b.start < a.end


def build_interference(intervals):
    n = len(intervals)
    order = sorted(range(n), key=lambda i: intervals[i].start)
    adj = [set() for _ in range(n)]
    active = []
    for i in order:
        iv = intervals[i]
        active = [j for j in active if intervals[j].end > iv.start]
        for j in active:
            adj[i].add(j); adj[j].add(i)
        active.append(i)
    return adj


def pressure_profile(intervals, n):
    p = np.zeros(n + 1, dtype=int)
    for iv in intervals:
        p[iv.start:iv.end] += 1
    return p


# ---------------------------------------------------------------- optimal
def optimal_spills(intervals, n, K):
    """Exact min weighted-spill allocation.
    Interval graphs are K-colourable iff max overlap <= K, and the constraint
    matrix is totally unimodular, so the LP relaxation is integral."""
    m = len(intervals)
    pts = sorted({iv.start for iv in intervals})
    A = np.zeros((len(pts), m))
    for r, p in enumerate(pts):
        for c, iv in enumerate(intervals):
            if iv.start <= p < iv.end:
                A[r, c] = 1
    cost = np.array([iv.cost for iv in intervals])
    res = milp(-cost, constraints=LinearConstraint(A, -np.inf, K),
               integrality=np.ones(m), bounds=Bounds(0, 1))
    keep = np.round(res.x).astype(int)
    return {intervals[i].vreg for i in range(m) if keep[i] == 0}, keep


# ------------------------------------------------- graph colouring (Briggs)
def color_allocate(intervals, K, spill_score):
    """Chaitin-Briggs optimistic colouring. `spill_score(i)` -> higher means
    'spill this first' when no node has degree < K."""
    adj = build_interference(intervals)
    live = set(range(len(intervals)))
    deg = {i: len(adj[i]) for i in live}
    stack = []
    while live:
        low = [i for i in live if deg[i] < K]
        if low:
            node = low[0]
        else:
            node = max(live, key=spill_score)       # potential spill
        stack.append(node)
        live.remove(node)
        for j in adj[node]:
            if j in live:
                deg[j] -= 1
    colour, spilled = {}, set()
    while stack:
        node = stack.pop()
        used = {colour[j] for j in adj[node] if j in colour}
        free = [c for c in range(K) if c not in used]
        if free:
            colour[node] = free[0]
        else:
            spilled.add(intervals[node].vreg)
    return spilled


def heuristic_chaitin(intervals, K):
    adj = build_interference(intervals)
    deg = [len(a) for a in adj]
    return color_allocate(intervals, K, lambda i: deg[i] / intervals[i].cost)


def heuristic_llvm_style(intervals, K):
    """LLVM's spill weight ~ (frequency-weighted uses) / interval size."""
    return color_allocate(
        intervals, K,
        lambda i: (intervals[i].end - intervals[i].start) / intervals[i].cost)


def linear_scan(intervals, K):
    """Poletto & Sarkar linear scan: spill the active interval ending last."""
    order = sorted(range(len(intervals)), key=lambda i: intervals[i].start)
    active, spilled = [], set()
    for i in order:
        iv = intervals[i]
        active = [j for j in active if intervals[j].end > iv.start]
        if len(active) < K:
            active.append(i)
        else:
            far = max(active + [i], key=lambda j: intervals[j].end)
            spilled.add(intervals[far].vreg)
            if far != i:
                active.remove(far); active.append(i)
    return spilled


def ml_allocator(intervals, K, n, model, featurize):
    X = featurize(intervals, n, K)
    p_keep = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
    return color_allocate(intervals, K, lambda i: 1.0 - p_keep[i])


def weighted_cost(intervals, spilled):
    return sum(iv.cost for iv in intervals if iv.vreg in spilled)
