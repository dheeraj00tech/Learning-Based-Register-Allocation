import numpy as np
from allocators import build_interference, pressure_profile

FEATURES = ["length", "n_uses", "log_cost", "max_depth", "degree", "degree_over_K",
            "crosses_call", "max_pressure", "pressure_minus_K", "cost_rel_fn",
            "cost_per_len", "cost_rank", "rel_start", "K",
            "cost_per_degree", "cost_vs_nbr_mean", "frac_nbrs_costlier", "frac_nbrs_costlier_overK",
            "avg_pressure"]


def featurize(intervals, n, K):
    adj = build_interference(intervals)
    prof = pressure_profile(intervals, n)
    costs = np.array([iv.cost for iv in intervals])
    cmax = costs.max() if len(costs) else 1.0
    rank = costs.argsort().argsort() / max(1, len(costs) - 1)
    rows = []
    for i, iv in enumerate(intervals):
        L = iv.end - iv.start
        mp = prof[iv.start:iv.end].max()
        nb = costs[list(adj[i])] if adj[i] else np.array([iv.cost])
        frac_c = float((nb > iv.cost).mean())
        # neighbours alive at this interval's peak-pressure point
        pk = iv.start + int(prof[iv.start:iv.end].argmax())
        at_pk = [j for j in adj[i] if intervals[j].start <= pk < intervals[j].end]
        rk_pk = float(np.mean([costs[j] > iv.cost for j in at_pk])) if at_pk else 0.0
        rows.append([L, len(iv.uses), np.log10(iv.cost + 1), iv.max_depth, len(adj[i]),
                     len(adj[i]) / K, int(iv.crosses_call), mp, mp - K, iv.cost / cmax,
                     iv.cost / L, rank[i], iv.start / n, K,
                     iv.cost / max(1, len(adj[i])), iv.cost / (nb.mean() + 1e-9), frac_c, rk_pk,
                     prof[iv.start:iv.end].mean()])
    return np.array(rows, dtype=float)
