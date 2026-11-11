# ---- Worked example and Berka trajectories (Notebook 4) ----
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .features import _to_date


def sig2(x):
    """Round to 2 significant figures so no exact record is reproduced."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    x = float(x)
    if x == 0:
        return 0.0
    return float(f"{x:.2g}")


def worked_example(hc, y, bur_cols, comb_cols, make_model_fn, reason_text, approve_rate=0.7, seed=11, k=6):
    X = hc[comb_cols]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    m_comb = make_model_fn(seed).fit(Xtr, ytr)
    m_bur = make_model_fn(seed).fit(Xtr[bur_cols], ytr)
    p_c = m_comb.predict_proba(Xte)[:, 1]
    p_b = m_bur.predict_proba(Xte[bur_cols])[:, 1]
    cut_c, cut_b = np.quantile(p_c, approve_rate), np.quantile(p_b, approve_rate)
    thin_te = hc.loc[Xte.index, 'thin_file'].values == 1
    out = {'approval_cutoff_combined': float(cut_c), 'approval_cutoff_bureau': float(cut_b)}

    def describe(pos, label):
        row = Xte.iloc[[pos]]
        contrib = m_comb.booster_.predict(row, pred_contrib=True)[0]
        base, contrib = float(contrib[-1]), contrib[:-1]
        order = np.argsort(-np.abs(contrib))[:k]
        drivers = [{'feature': comb_cols[j], 'plain_language': reason_text.get(comb_cols[j], comb_cols[j]),
                    'value_rounded': sig2(row.iloc[0, j]), 'contribution_log_odds': round(float(contrib[j]), 3),
                    'direction': 'raises risk' if contrib[j] > 0 else 'lowers risk'} for j in order]
        reasons = [reason_text.get(comb_cols[j], comb_cols[j]) for j in np.argsort(-contrib)[:4] if contrib[j] > 0]
        d = {'label': label, 'pd_combined': round(float(p_c[pos]), 3), 'pd_bureau_only': round(float(p_b[pos]), 3),
             'decision_combined': 'approve' if p_c[pos] <= cut_c else 'decline',
             'decision_bureau_only': 'approve' if p_b[pos] <= cut_b else 'decline',
             'actual_outcome': 'defaulted' if yte[pos] == 1 else 'repaid',
             'base_log_odds': round(base, 3), 'top_drivers': drivers, 'adverse_action_reasons': reasons}
        top = int(np.argmax(contrib))
        if contrib[top] > 0:
            cf = row.copy(); cf.iloc[0, top] = Xtr[comb_cols[top]].median()
            pcf = float(m_comb.predict_proba(cf)[:, 1][0])
            d['counterfactual'] = {'changed_feature': reason_text.get(comb_cols[top], comb_cols[top]),
                                   'set_to_typical_value': sig2(Xtr[comb_cols[top]].median()),
                                   'new_pd': round(pcf, 3), 'new_decision': 'approve' if pcf <= cut_c else 'decline'}
        return d

    # Typical declined thin-file applicant: the one closest to the median risk of that group
    dec = np.where(thin_te & (p_c > cut_c))[0]
    pos_dec = dec[np.argmin(np.abs(p_c[dec] - np.median(p_c[dec])))]
    out['declined_thin_file'] = describe(pos_dec, 'Typical declined thin-file applicant')
    # A thin-file applicant the bureau-only model declines but the combined model approves, and who repaid
    rescued = np.where(thin_te & (p_b > cut_b) & (p_c <= cut_c) & (yte == 0))[0]
    out['n_thin_file_declined_by_bureau_approved_by_combined'] = int((thin_te & (p_b > cut_b) & (p_c <= cut_c)).sum())
    out['n_of_those_who_repaid'] = int(len(rescued))
    out['n_thin_file_test'] = int(thin_te.sum())
    # Reverse swap: approved on bureau data, declined with alternative data
    reverse = thin_te & (p_b <= cut_b) & (p_c > cut_c)
    out['n_thin_file_approved_by_bureau_declined_by_combined'] = int(reverse.sum())
    out['n_of_those_who_defaulted'] = int((reverse & (yte == 1)).sum())
    if len(rescued):
        pos_r = rescued[np.argmin(np.abs(p_c[rescued] - np.median(p_c[rescued])))]
        out['rescued_thin_file'] = describe(pos_r, 'Thin-file applicant declined on bureau data, approved with alternative data, repaid')
    return out


def berka_trajectories(tables, months=12):
    """Median and quartiles of month-end balance and monthly inflow, by outcome, for the months before each loan."""
    loan = tables['loan'].copy(); tr = tables['trans'].copy()
    for t in (loan, tr):
        t.columns = [c.lower() for c in t.columns]
    loan['date'] = _to_date(loan['date']); tr['date'] = _to_date(tr['date'])
    loan['default'] = loan['status'].astype(str).str.upper().isin(['B', 'D']).astype(int)
    tr = tr.merge(loan[['account_id', 'date', 'default']].rename(columns={'date': 'loan_date'}), on='account_id')
    tr = tr[tr['date'] < tr['loan_date']]
    tr['m'] = ((tr['loan_date'].dt.year - tr['date'].dt.year) * 12 + (tr['loan_date'].dt.month - tr['date'].dt.month))
    tr = tr[(tr['m'] >= 1) & (tr['m'] <= months)]
    tr['inflow'] = np.where(tr['type'].astype(str).str.upper().isin(['PRIJEM', 'CREDIT']), tr['amount'].abs(), 0.0)
    tr = tr.sort_values('date')
    per = tr.groupby(['account_id', 'm', 'default']).agg(balance=('balance', 'last'), inflow=('inflow', 'sum')).reset_index()
    res = {}
    for d, g in per.groupby('default'):
        q = g.groupby('m')[['balance', 'inflow']].quantile([0.25, 0.5, 0.75]).unstack()
        res['defaulted' if d == 1 else 'repaid'] = {
            'months_before_loan': [int(x) for x in q.index],
            'balance_q25': q[('balance', 0.25)].round(0).tolist(), 'balance_median': q[('balance', 0.5)].round(0).tolist(),
            'balance_q75': q[('balance', 0.75)].round(0).tolist(), 'inflow_median': q[('inflow', 0.5)].round(0).tolist(),
            'n_accounts': int(g['account_id'].nunique())}
    return res
