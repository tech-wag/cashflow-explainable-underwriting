# ---- Explainability, fairness and robustness (Notebook 3) ----
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REASON_TEXT = {
    "bur_ext_source_1": "External credit score 1 is low", "bur_ext_source_2": "External credit score 2 is low",
    "bur_ext_source_3": "External credit score 3 is low", "bur_n_loans": "Number of credit bureau accounts",
    "bur_n_active": "Number of active credit accounts", "bur_oldest_years": "Length of credit history",
    "bur_newest_years": "Time since most recent credit account", "bur_max_days_overdue": "Days overdue on bureau accounts",
    "bur_total_credit": "Total credit extended", "bur_total_debt": "Total outstanding debt",
    "bur_total_overdue": "Amount currently overdue", "bur_n_prolonged": "Number of extended credit agreements",
    "bur_debt_to_credit": "Debt is high relative to credit", "bur_late_month_share": "Share of months with late payments",
    "alt_income": "Income level", "alt_credit_to_income": "Loan amount is high relative to income",
    "alt_annuity_to_income": "Repayment is high relative to income", "alt_credit_to_goods": "Loan amount relative to purchase price",
    "alt_employment_years": "Length of employment", "alt_not_employed": "Not currently employed",
    "alt_income_type": "Type of income", "alt_prev_n": "Number of previous applications",
    "alt_prev_approved_share": "Share of previous applications approved", "alt_prev_refused_share": "Share of previous applications refused",
    "alt_inst_n": "Number of past instalments", "alt_inst_late_share": "Share of past instalments paid late",
    "alt_inst_mean_days_late": "Average days late on past instalments", "alt_inst_max_days_late": "Longest delay on a past instalment",
    "alt_inst_short_share": "Share of past instalments underpaid", "alt_inst_paid_ratio": "Share of instalment amounts repaid",
    "alt_pos_dpd_share": "Past due on point-of-sale loans", "alt_pos_max_dpd": "Longest delay on point-of-sale loans",
    "alt_cc_util_mean": "Card balance is high relative to limit", "alt_cc_dpd_share": "Past due on card payments",
}


def lgb_contributions(model, X):
    """Exact TreeSHAP contributions from LightGBM (last column is the base value)."""
    return model.booster_.predict(X, pred_contrib=True)[:, :-1]


def top_reasons(contrib, cols, k=4):
    """Features pushing risk up the most, per applicant."""
    idx = np.argsort(-contrib, axis=1)[:, :k]
    return [[cols[j] for j in row if contrib[i, j] > 0] for i, row in enumerate(idx)]


def reason_stability(make_model_fn, X, y, cols, seeds, approve_rate=0.7, k=4, n_explain=5000):
    """Train one model per seed on the same split; compare adverse-action reasons for the same declined applicants."""
    Xtr, Xte, ytr, yte = train_test_split(X[cols], y, test_size=0.2, stratify=y, random_state=0)
    models = [make_model_fn(s).fit(Xtr, ytr) for s in seeds]
    p0 = models[0].predict_proba(Xte)[:, 1]
    declined = np.where(p0 > np.quantile(p0, approve_rate))[0][:n_explain]
    Xd = Xte.iloc[declined]
    reasons = [top_reasons(lgb_contributions(m, Xd), cols, k) for m in models]
    top1, jacc = [], []
    for a in range(len(seeds)):
        for b in range(a + 1, len(seeds)):
            for ra, rb in zip(reasons[a], reasons[b]):
                if ra and rb:
                    top1.append(ra[0] == rb[0])
                    sa, sb = set(ra), set(rb)
                    jacc.append(len(sa & sb) / len(sa | sb))
    return {"n_declined_explained": int(len(declined)), "top1_agreement": float(np.mean(top1)),
            "top4_jaccard_mean": float(np.mean(jacc))}, models[0], Xtr, Xte, yte, p0


def reason_faithfulness(model, Xtr, Xte, p_te, cols, approve_rate=0.7, n_explain=5000, seed=0):
    """Move each declined applicant's top reason to the typical approved value; compare with a random other feature."""
    rng = np.random.default_rng(seed)
    cut = np.quantile(p_te, approve_rate)
    dec = np.where(p_te > cut)[0][:n_explain]
    Xd = Xte.iloc[dec].copy()
    contrib = lgb_contributions(model, Xd)
    ref = Xtr.median(numeric_only=True)
    base = model.predict_proba(Xd)[:, 1]
    top = np.argmax(contrib, axis=1)
    Xt, Xr = Xd.copy(), Xd.copy()
    for i in range(len(Xd)):
        j = top[i]; Xt.iat[i, j] = ref[cols[j]]
        others = [c for c in range(len(cols)) if c != j]
        r = rng.choice(others); Xr.iat[i, r] = ref[cols[r]]
    pt, pr = model.predict_proba(Xt)[:, 1], model.predict_proba(Xr)[:, 1]
    return {"mean_pd_declined": float(base.mean()),
            "mean_drop_top_reason": float((base - pt).mean()), "mean_drop_random_feature": float((base - pr).mean()),
            "share_flipped_to_approve_top_reason": float((pt <= cut).mean()),
            "share_flipped_to_approve_random_feature": float((pr <= cut).mean())}


def global_importance(model, X, cols, n=20000, seed=0):
    Xs = X[cols].sample(min(n, len(X)), random_state=seed)
    c = np.abs(lgb_contributions(model, Xs)).mean(axis=0)
    return (pd.Series(c, index=cols).sort_values(ascending=False) / c.sum()).round(4)


def fairness_report(y, p, groups, approve_rate=0.7, reference=None):
    """Approval rates by group at one overall cutoff; four-fifths ratio and equal-opportunity gap for good borrowers."""
    y = np.asarray(y); p = np.asarray(p); g = pd.Series(np.asarray(groups))
    cut = np.quantile(p, approve_rate); ap = p <= cut
    rows = {}
    for k in sorted(g.dropna().unique()):
        m = (g == k).values
        rows[str(k)] = {"n": int(m.sum()), "approval_rate": float(ap[m].mean()),
                        "approval_rate_good_borrowers": float(ap[m & (y == 0)].mean()),
                        "approval_rate_defaulters": float(ap[m & (y == 1)].mean()),
                        "default_rate_among_approved": float(y[m & ap].mean())}
    ref = reference or max(rows, key=lambda k: rows[k]["approval_rate"])
    for k in rows:
        rows[k]["approval_ratio_vs_" + ref] = rows[k]["approval_rate"] / rows[ref]["approval_rate"]
        rows[k]["good_borrower_gap_vs_" + ref] = rows[k]["approval_rate_good_borrowers"] - rows[ref]["approval_rate_good_borrowers"]
    return rows
