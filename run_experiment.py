"""End-to-end: generate programs -> optimal labels -> train -> evaluate -> plots."""
import numpy as np, pandas as pd, joblib, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from program_gen import generate, save
from allocators import *
from features import featurize, FEATURES

KS = [4, 8, 16]
NUM, N_TRAIN = 300, 200
funcs = generate(NUM, seed=42)
save(funcs, "results/programs.json")
train, test = funcs[:N_TRAIN], funcs[N_TRAIN:]       # split BY PROGRAM
print(f"{len(train)} train / {len(test)} test functions")

def build_dataset(fs):
    rows = []
    for f in fs:
        for K in KS:
            _, keep = optimal_spills(f.intervals, f.n, K)
            X = featurize(f.intervals, f.n, K)
            for x, y, iv in zip(X, keep, f.intervals):
                rows.append([f.name, iv.vreg, *x, int(y)])
    return pd.DataFrame(rows, columns=["func", "vreg", *FEATURES, "keep"])

print("Solving optimal allocations (labels)...")
df_train = build_dataset(train)
df_train.to_csv("results/dataset_train.csv", index=False)
print(df_train.shape, "kept fraction: %.2f" % df_train.keep.mean())

model = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=0)
model.fit(df_train[FEATURES].values, df_train.keep.values)
joblib.dump(model, "results/model.joblib")

# ---- evaluation on unseen programs
methods = {
    "Linear scan": lambda f, K: linear_scan(f.intervals, K),
    "Chaitin (deg/cost)": lambda f, K: heuristic_chaitin(f.intervals, K),
    "LLVM-style weight": lambda f, K: heuristic_llvm_style(f.intervals, K),
    "ML (GBM)": lambda f, K: ml_allocator(f.intervals, K, f.n, model, featurize),
    "Optimal (ILP)": lambda f, K: optimal_spills(f.intervals, f.n, K)[0],
}
rec = []
for f in test:
    for K in KS:
        if pressure_profile(f.intervals, f.n).max() <= K:
            continue                                  # nothing to spill
        for name, fn in methods.items():
            sp = fn(f, K)
            rec.append(dict(func=f.name, K=K, method=name, spills=len(sp),
                            cost=weighted_cost(f.intervals, sp)))
res = pd.DataFrame(rec)
res.to_csv("results/eval_raw.csv", index=False)

opt = res[res.method == "Optimal (ILP)"].set_index(["func", "K"]).cost
res["ratio"] = [r.cost / opt[(r.func, r.K)] if opt[(r.func, r.K)] > 0 else 1.0 for r in res.itertuples()]
summary = res.groupby(["K", "method"]).agg(avg_spills=("spills", "mean"),
          avg_weighted_cost=("cost", "mean"), avg_cost_ratio_vs_opt=("ratio", "mean"),
          n_funcs=("func", "nunique")).round(2)
summary.to_csv("results/summary.csv")
print(summary.to_string())

# win-rate ML vs best heuristic
piv = res.pivot_table(index=["func", "K"], columns="method", values="cost")
heur = piv[["Chaitin (deg/cost)", "LLVM-style weight", "Linear scan"]].min(axis=1)
print("\nML <= best heuristic on %.1f%% of cases; strictly better on %.1f%%" %
      (100 * (piv["ML (GBM)"] <= heur).mean(), 100 * (piv["ML (GBM)"] < heur).mean()))
print("ML <= Chaitin on %.1f%%, ML <= LLVM-style on %.1f%%" %
      (100 * (piv["ML (GBM)"] <= piv["Chaitin (deg/cost)"]).mean(),
       100 * (piv["ML (GBM)"] <= piv["LLVM-style weight"]).mean()))

# ---- plots
order = list(methods)
fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
w = 0.16
for ax, col, title in [(axs[0], "avg_weighted_cost", "Avg weighted spill cost (lower is better)"),
                       (axs[1], "avg_cost_ratio_vs_opt", "Cost ratio vs optimal (1.0 = optimal)")]:
    for i, m in enumerate(order):
        vals = [summary.loc[(K, m), col] for K in KS]
        ax.bar(np.arange(len(KS)) + i * w, vals, w, label=m)
    ax.set_xticks(np.arange(len(KS)) + 2 * w); ax.set_xticklabels([f"K={k}" for k in KS])
    ax.set_title(title); ax.set_yscale("log" if col == "avg_weighted_cost" else "linear")
axs[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig("results/spill_cost_comparison.png", dpi=150)

plt.figure(figsize=(7, 4.5))
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
imp.plot.barh(); plt.title("Feature importance (GBM)"); plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=150)

plt.figure(figsize=(7, 4))
for m in order:
    s = res[res.method == m].groupby("K").spills.mean()
    plt.plot(s.index, s.values, marker="o", label=m)
plt.xlabel("K (registers)"); plt.ylabel("avg # spilled ranges"); plt.legend(fontsize=8)
plt.title("Spill count vs register count"); plt.tight_layout()
plt.savefig("results/spill_count_vs_K.png", dpi=150)
print("done")
