# ---- Robustness extensions (Notebook 5): tuning, overfitting, class imbalance, external scores, fairness mitigation ----
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, ParameterSampler, ParameterGrid, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

HC_FIXED = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0)
BK_FIXED = dict(n_estimators=150, learning_rate=0.05, num_leaves=7, min_child_samples=20,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8)

# Search spaces. Home Credit: random search with early stopping picks the number of trees.
HC_SPACE = {"num_leaves": [7, 15, 31, 63, 127], "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
            "min_child_samples": [20, 50, 100, 200, 400], "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0], "reg_lambda": [0.0, 0.1, 1.0, 5.0, 10.0]}
# Berka: full grid, number of trees searched directly (early stopping on ~110 validation loans is too noisy).
BK_GRID = {"num_leaves": [3, 7, 15], "n_estimators": [50, 150, 300], "min_child_samples": [10, 20, 40]}
LOGIT_C = [0.001, 0.01, 0.1, 1.0, 10.0]


def lgbm(params, seed, **extra):
    p = {"subsample_freq": 1, **params, **extra}
    return lgb.LGBMClassifier(random_state=seed, verbose=-1, n_jobs=-1, **p)


def logit(C=1.0):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(max_iter=2000, C=C))


def _sub(X, y, max_rows, seed):
    if max_rows is None or len(y) <= max_rows:
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=max_rows, stratify=y, random_state=seed)
    return Xs, np.asarray(ys)


# ---------- metrics ----------
def ece(y, p, n_bins=10):
    """Expected calibration error over deciles of predicted risk."""
    q = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    d = pd.DataFrame({"y": y, "p": p, "b": q}).groupby("b").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((d["n"] * (d["p"] - d["y"]).abs()).sum() / d["n"].sum())


def auc_ci(y, p, n_boot=200, seed=0):
    rng = np.random.default_rng(seed); n = len(y); v = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def decision_metrics(y, p, approve_rate=0.7):
    cut = np.quantile(p, approve_rate); ap = p <= cut; dec = ~ap
    return {"default_rate_approved": float(y[ap].mean()),
            "share_of_defaulters_declined": float(dec[y == 1].mean()),       # recall on defaults
            "default_rate_among_declined": float(y[dec].mean()),              # precision on defaults
            "approval_rate_good_borrowers": float(ap[y == 0].mean())}


def summarise(y, p, mask=None, n_boot=200):
    y = np.asarray(y); p = np.asarray(p)
    if mask is not None:
        y, p = y[mask], p[mask]
    return {"n": int(len(y)), "auc": float(roc_auc_score(y, p)), "auc_ci95": auc_ci(y, p, n_boot),
            "pr_auc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
            "ece": ece(y, p), **decision_metrics(y, p)}


# ---------- A. hyperparameter tuning inside nested cross-validation ----------
def tune_lgbm(X, y, seed, space=None, n_iter=12, inner_folds=3, max_rows=50000, early_stop=True, base=None):
    """Random (or full-grid) search on the training data only. Returns best params and the search trace."""
    Xs, ys = _sub(X, y, max_rows, seed)
    if early_stop:
        cands = list(ParameterSampler(space, n_iter, random_state=seed))
        fixed = {k: v for k, v in (base or HC_FIXED).items() if k in space}
        cands.insert(0, fixed)  # the hand-set configuration always competes
    else:
        cands = list(ParameterGrid(space))
    skf = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
    splits = list(skf.split(Xs, ys)); trace = []
    for c in cands:
        aucs, iters = [], []
        for tr, va in splits:
            if early_stop:
                m = lgbm({**(base or HC_FIXED), **c, "n_estimators": 2000}, seed, metric="auc")
                m.fit(Xs.iloc[tr], ys[tr], eval_set=[(Xs.iloc[va], ys[va])], eval_metric="auc",
                      callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)])
                iters.append(m.best_iteration_ or 2000)
            else:
                m = lgbm({**(base or BK_FIXED), **c}, seed).fit(Xs.iloc[tr], ys[tr])
            aucs.append(roc_auc_score(ys[va], m.predict_proba(Xs.iloc[va])[:, 1]))
        rec = {**c, "inner_auc": float(np.mean(aucs))}
        if early_stop:
            rec["n_estimators"] = int(max(50, np.mean(iters) * 1.1))
        trace.append(rec)
    best = max(trace, key=lambda r: r["inner_auc"])
    params = {**(base or (HC_FIXED if early_stop else BK_FIXED)), **{k: v for k, v in best.items() if k != "inner_auc"}}
    return params, trace


def tune_logit(X, y, seed, inner_folds=3, max_rows=50000):
    Xs, ys = _sub(X, y, max_rows, seed)
    skf = StratifiedKFold(inner_folds, shuffle=True, random_state=seed); res = {}
    for C in LOGIT_C:
        a = [roc_auc_score(ys[va], logit(C).fit(Xs.iloc[tr], ys[tr]).predict_proba(Xs.iloc[va])[:, 1])
             for tr, va in skf.split(Xs, ys)]
        res[C] = float(np.mean(a))
    return max(res, key=res.get), res


def nested_cv(X, y, kind, outer_seed, fixed_params, tune_kw, outer_folds=5, log=print):
    """Outer folds measure performance; all tuning happens inside each outer training fold."""
    y = np.asarray(y); P_t = np.zeros(len(y)); P_f = np.zeros(len(y)); chosen = []
    skf = StratifiedKFold(outer_folds, shuffle=True, random_state=outer_seed)
    for k, (tr, te) in enumerate(skf.split(X, y)):
        s = outer_seed * 100 + k
        if kind == "gbm":
            best, _ = tune_lgbm(X.iloc[tr], y[tr], s, **tune_kw)
            m_t, m_f = lgbm(best, s), lgbm(fixed_params, s)
            chosen.append({k2: best[k2] for k2 in ("num_leaves", "learning_rate", "n_estimators", "min_child_samples",
                                                   "subsample", "colsample_bytree", "reg_lambda") if k2 in best})
        else:
            C, _ = tune_logit(X.iloc[tr], y[tr], s, max_rows=tune_kw.get("max_rows", 50000))
            m_t, m_f = logit(C), logit(1.0); chosen.append({"C": C})
        P_t[te] = m_t.fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        P_f[te] = m_f.fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        log(f"    fold {k + 1}/{outer_folds}: tuned AUC {roc_auc_score(y[te], P_t[te]):.3f}, "
            f"fixed AUC {roc_auc_score(y[te], P_f[te]):.3f}, chosen {chosen[-1]}")
    return P_t, P_f, chosen


# ---------- B. overfitting and out-of-sample stability (Berka) ----------
def cv_auc(X, y, make, seed, folds=5):
    y = np.asarray(y); P = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        P[te] = make(seed).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    return roc_auc_score(y, P), P


def repeat_spread(X, y, make, seeds):
    a = [cv_auc(X, y, make, s)[0] for s in seeds]
    return {"auc_mean": float(np.mean(a)), "auc_sd": float(np.std(a)), "auc_min": float(np.min(a)),
            "auc_max": float(np.max(a)), "n_repeats": len(a)}


def train_test_gap(X, y, make, seeds, folds=5):
    """Mean AUC on the training folds vs on the held-out folds."""
    y = np.asarray(y); tr_a, te_a = [], []
    for s in seeds:
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=s).split(X, y):
            m = make(s).fit(X.iloc[tr], y[tr])
            tr_a.append(roc_auc_score(y[tr], m.predict_proba(X.iloc[tr])[:, 1]))
            te_a.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1]))
    return {"train_auc": float(np.mean(tr_a)), "test_auc": float(np.mean(te_a)), "gap": float(np.mean(tr_a) - np.mean(te_a))}


def permutation_test(X, y, make, n_perm=200, seed=0):
    """Ojala and Garriga (2010): AUC with shuffled labels gives the chance distribution."""
    y = np.asarray(y); obs = cv_auc(X, y, make, seed)[0]; rng = np.random.default_rng(seed); null = []
    for _ in range(n_perm):
        null.append(cv_auc(X, rng.permutation(y), make, seed)[0])
    null = np.array(null)
    return {"observed_auc": float(obs), "null_mean": float(null.mean()), "null_95th": float(np.percentile(null, 95)),
            "null_max": float(null.max()), "p_value": float((1 + (null >= obs).sum()) / (n_perm + 1)), "n_perm": n_perm}


def learning_curve(X, y, make, fractions=(0.2, 0.4, 0.6, 0.8, 1.0), seeds=range(10), folds=5):
    y = np.asarray(y); out = {}
    for f in fractions:
        a = []
        for s in seeds:
            rng = np.random.default_rng(s)
            for tr, te in StratifiedKFold(folds, shuffle=True, random_state=s).split(X, y):
                if f < 1.0:
                    tr, _ = train_test_split(tr, train_size=f, stratify=y[tr], random_state=s)
                if y[tr].min() == y[tr].max():
                    continue
                a.append(roc_auc_score(y[te], make(s).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]))
        out[str(f)] = {"n_train_approx": int(len(y) * (folds - 1) / folds * f), "auc_mean": float(np.mean(a)),
                       "auc_sd": float(np.std(a))}
    return out


def out_of_time(df, cols, y_col, date_col, make, train_share=0.7, seeds=range(10), n_boot=1000):
    """Train on the earliest loans, test on the latest ones."""
    d = df.sort_values(date_col).reset_index(drop=True)
    cut = int(len(d) * train_share); tr, te = d.iloc[:cut], d.iloc[cut:]
    P = np.mean([make(s).fit(tr[cols], tr[y_col].values).predict_proba(te[cols])[:, 1] for s in seeds], axis=0)
    yt = te[y_col].values
    return {"train_period": [str(tr[date_col].min())[:10], str(tr[date_col].max())[:10]],
            "test_period": [str(te[date_col].min())[:10], str(te[date_col].max())[:10]],
            "n_train": int(len(tr)), "n_test": int(len(te)), "defaults_train": int(tr[y_col].sum()),
            "defaults_test": int(yt.sum()), "auc": float(roc_auc_score(yt, P)), "auc_ci95": auc_ci(yt, P, n_boot)}


# ---------- C. class imbalance ----------
def focal_objective(gamma=2.0):
    """Binary focal loss (Lin et al. 2017) for LightGBM; gradients by central differences on the raw margin."""
    def loss(z, y):
        p = 1 / (1 + np.exp(-z)); pt = np.where(y == 1, p, 1 - p)
        return -((1 - pt) ** gamma) * np.log(np.clip(pt, 1e-12, 1))

    def obj(y_true, z):
        h = 1e-3
        l0, lp, lm = loss(z, y_true), loss(z + h, y_true), loss(z - h, y_true)
        grad = (lp - lm) / (2 * h); hess = np.maximum((lp - 2 * l0 + lm) / h ** 2, 1e-6)
        return grad, hess
    return obj


def smote(X, y, ratio=1.0, k=5, seed=0):
    """Chawla et al. (2002): synthetic minority points on lines between a minority point and one of its k neighbours.
    X must have no missing values. ratio = minority / majority after resampling."""
    rng = np.random.default_rng(seed); X = np.asarray(X, float); y = np.asarray(y)
    mino = X[y == 1]; n_new = int(ratio * (y == 0).sum()) - len(mino)
    if n_new <= 0:
        return X, y
    nn = NearestNeighbors(n_neighbors=k + 1).fit(mino); _, idx = nn.kneighbors(mino)
    base = rng.integers(0, len(mino), n_new); nb = idx[base, rng.integers(1, k + 1, n_new)]
    gap = rng.random((n_new, 1)); new = mino[base] + gap * (mino[nb] - mino[base])
    return np.vstack([X, new]), np.concatenate([y, np.ones(n_new, int)])


def fit_predict_imbalance(method, Xtr, ytr, Xte, params, seed):
    ytr = np.asarray(ytr)
    if method == "none":
        return lgbm(params, seed).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    if method == "class_weight":
        spw = (ytr == 0).sum() / (ytr == 1).sum()
        return lgbm(params, seed, scale_pos_weight=spw).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    if method == "focal_loss":
        m = lgbm(params, seed, objective=focal_objective(2.0)).fit(Xtr, ytr)
        z = m.predict(Xte, raw_score=True)
        return 1 / (1 + np.exp(-z))
    if method == "undersample":
        rng = np.random.default_rng(seed); pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
        n = min(len(pos), len(neg))  # keep every default, sample the same number of repaid loans
        keep = np.concatenate([pos, rng.choice(neg, n, replace=False)])
        return lgbm(params, seed).fit(Xtr.iloc[keep], ytr[keep]).predict_proba(Xte)[:, 1]
    if method == "smote":
        imp = SimpleImputer(strategy="median").fit(Xtr)
        Xs, ys = smote(imp.transform(Xtr), ytr, ratio=1.0, seed=seed)
        cols = list(Xtr.columns)
        m = lgbm(params, seed).fit(pd.DataFrame(Xs, columns=cols), ys)
        return m.predict_proba(pd.DataFrame(imp.transform(Xte), columns=cols))[:, 1]
    raise ValueError(method)


def imbalance_cv(X, y, methods, params, seed=11, folds=5, log=print):
    y = np.asarray(y); out = {}
    splits = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y))
    for meth in methods:
        t0 = time.time(); P = np.zeros(len(y))
        for tr, te in splits:
            P[te] = fit_predict_imbalance(meth, X.iloc[tr], y[tr], X.iloc[te], params, seed)
        out[meth] = P
        log(f"  {meth:13s} AUC {roc_auc_score(y, P):.4f}  ({time.time() - t0:.0f}s)")
    return out


# ---------- E. fairness mitigation ----------
def reweighing_weights(y, g):
    """Kamiran and Calders (2012): weight each (group, outcome) cell so outcome and group are independent."""
    d = pd.DataFrame({"y": np.asarray(y), "g": np.asarray(g)}); n = len(d)
    pg = d["g"].map(d["g"].value_counts() / n); py = d["y"].map(d["y"].value_counts() / n)
    pgy = d.groupby(["g", "y"])["y"].transform("size") / n
    return (pg * py / pgy).values


def age_proxies(X, age, threshold=0.2):
    rho = {c: float(stats.spearmanr(X[c], age, nan_policy="omit").statistic) for c in X.columns}
    rho = dict(sorted(rho.items(), key=lambda kv: -abs(kv[1])))
    chosen = [c for c, r in rho.items() if abs(r) >= threshold]
    return (chosen or list(rho)[:3]), rho  # if nothing passes the threshold, drop the three closest to age


def group_thresholds(p_cal, y_cal, g_cal, mode, approve_rate=0.7):
    """Score cut-offs per group, learned on calibration (out-of-fold training) scores.
    mode 'equal_opportunity': same approval rate among good borrowers in every group, overall approval = approve_rate.
    mode 'equal_approval':    same approval rate in every group."""
    p_cal = np.asarray(p_cal); y_cal = np.asarray(y_cal); g_cal = np.asarray(g_cal); groups = pd.unique(g_cal)
    if mode == "equal_approval":
        return {g: float(np.quantile(p_cal[g_cal == g], approve_rate)) for g in groups}

    def thr_for(t):
        return {g: float(np.quantile(p_cal[(g_cal == g) & (y_cal == 0)], t)) for g in groups}

    def overall(t):
        th = thr_for(t); return np.mean(p_cal <= pd.Series(g_cal).map(th).values)
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if overall(mid) < approve_rate else (lo, mid)
    return thr_for((lo + hi) / 2)


def approvals_from_thresholds(p, g, thr):
    t = pd.Series(np.asarray(g)).map(thr).fillna(np.inf).values
    return np.asarray(p) <= t


def fairness_table(y, approve, g, young="<30", old="50+"):
    y = np.asarray(y); approve = np.asarray(approve, bool); g = np.asarray(g); rows = {}
    for k in pd.unique(g):
        m = g == k
        rows[str(k)] = {"n": int(m.sum()), "approval_rate": float(approve[m].mean()),
                        "approval_rate_good": float(approve[m & (y == 0)].mean()),
                        "default_rate_approved": float(y[m & approve].mean()) if (m & approve).any() else None}
    return {"by_group": rows, "overall_approval_rate": float(approve.mean()),
            "default_rate_approved": float(y[approve].mean()),
            "approval_ratio_young_vs_old": rows[young]["approval_rate"] / rows[old]["approval_rate"],
            "good_borrower_gap_young_minus_old": rows[young]["approval_rate_good"] - rows[old]["approval_rate_good"]}


def oof_scores(X, y, make, seed, folds=5, weights=None):
    y = np.asarray(y); P = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        kw = {} if weights is None else {"sample_weight": weights[tr]}
        P[te] = make(seed).fit(X.iloc[tr], y[tr], **kw).predict_proba(X.iloc[te])[:, 1]
    return P
