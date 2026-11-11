# ---- Modelling and evaluation (Notebook 2) ----
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_LGB = False


def make_model(kind, seed, small_data=False):
    if kind == "logit":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0))
    if HAVE_LGB:
        p = dict(n_estimators=150, learning_rate=0.05, num_leaves=7, min_child_samples=20,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8) if small_data else \
            dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0)
        return lgb.LGBMClassifier(random_state=seed, verbose=-1, n_jobs=-1, **p)
    return HistGradientBoostingClassifier(max_iter=150 if small_data else 400, learning_rate=0.05,
                                          max_leaf_nodes=7 if small_data else 31, random_state=seed)


def oof_predictions(X, y, kind, seeds, n_splits=5, small_data=False):
    """Out-of-fold predicted default probabilities, one column per seed."""
    P = np.zeros((len(y), len(seeds)))
    for j, s in enumerate(seeds):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=s)
        for tr, te in skf.split(X, y):
            m = make_model(kind, s, small_data)
            m.fit(X.iloc[tr], y[tr])
            P[te, j] = m.predict_proba(X.iloc[te])[:, 1]
    return P


def ks_stat(y, p):
    return float(stats.ks_2samp(p[y == 1], p[y == 0]).statistic)


def approval_metrics(y, p, approve_rate=0.7):
    """Approve the lowest-risk share; report default rate among approved."""
    cut = np.quantile(p, approve_rate)
    ap = p <= cut
    return {"approve_rate": approve_rate, "default_rate_approved": float(y[ap].mean()),
            "default_rate_declined": float(y[~ap].mean()), "threshold": float(cut)}


def calib_bins(y, p, n_bins=10):
    q = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    d = pd.DataFrame({"y": y, "p": p, "bin": q}).groupby("bin").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    return d.reset_index().to_dict(orient="list")


def bootstrap_auc_ci(y, p, n_boot=500, seed=0):
    rng = np.random.default_rng(seed); n = len(y); vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        vals.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# Fast DeLong test (Sun & Xu 2014) for two correlated AUCs on the same cases
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def delong_test(y, p1, p2):
    y = np.asarray(y); order = np.argsort(-y, kind="mergesort")
    y = y[order]; preds = np.vstack([p1, p2])[:, order]
    m = int(y.sum()); n = len(y) - m
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.array([_midrank(r) for r in pos]); ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    diff = aucs[0] - aucs[1]
    se = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    z = diff / se if se > 0 else 0.0
    return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "diff": float(diff),
            "z": float(z), "p_value": float(2 * stats.norm.sf(abs(z)))}


def evaluate(y, P, mask=None, n_boot=500):
    """P: n x seeds out-of-fold probabilities. Seed spread plus bootstrap CI on the seed-averaged score."""
    y = np.asarray(y)
    if mask is not None:
        y, P = y[mask], P[mask]
    per_seed = [roc_auc_score(y, P[:, j]) for j in range(P.shape[1])]
    p = P.mean(axis=1)
    return {"n": int(len(y)), "default_rate": float(y.mean()),
            "auc_mean_over_seeds": float(np.mean(per_seed)), "auc_sd_over_seeds": float(np.std(per_seed)),
            "auc": float(roc_auc_score(y, p)), "auc_ci95": bootstrap_auc_ci(y, p, n_boot),
            "pr_auc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "ks": ks_stat(y, p), "approval_at_70pct": approval_metrics(y, p, 0.7),
            "calibration": calib_bins(y, p)}


def run_suite(df, y, feature_sets, seeds, subgroups, small_data=False, n_boot=500, log=print):
    """Train every learner on every feature set; evaluate overall and per subgroup; DeLong vs the first set."""
    y = np.asarray(y).astype(int)
    subgroups = {g: (None if m is None else np.asarray(m, bool)) for g, m in subgroups.items()}
    results, oof = {}, {}
    for kind in ["logit", "gbm"]:
        for name, cols in feature_sets.items():
            log(f"  {kind:5s} | {name:12s} | {len(cols)} features")
            P = oof_predictions(df[cols], y, kind, seeds, small_data=small_data)
            oof[(kind, name)] = P
            results[f"{kind}|{name}"] = {g: evaluate(y, P, m, n_boot) for g, m in subgroups.items()}
    base = list(feature_sets)[0]
    tests = {}
    for kind in ["logit", "gbm"]:
        for name in list(feature_sets)[1:]:
            for g, m in subgroups.items():
                mm = np.ones(len(y), bool) if m is None else m
                tests[f"{kind}|{name} vs {base}|{g}"] = delong_test(
                    y[mm], oof[(kind, name)][mm].mean(1), oof[(kind, base)][mm].mean(1))
    return results, tests, oof


def headline_table(results):
    rows = []
    for key, groups in results.items():
        kind, fs = key.split("|")
        for g, r in groups.items():
            rows.append({"learner": kind, "features": fs, "group": g, "n": r["n"],
                         "AUC": round(r["auc"], 3), "AUC 95% CI": f"{r['auc_ci95'][0]:.3f}-{r['auc_ci95'][1]:.3f}",
                         "AUC sd (seeds)": round(r["auc_sd_over_seeds"], 4), "PR-AUC": round(r["pr_auc"], 3),
                         "Brier": round(r["brier"], 4), "KS": round(r["ks"], 3),
                         "Default rate if 70% approved": round(r["approval_at_70pct"]["default_rate_approved"], 4)})
    return pd.DataFrame(rows)
