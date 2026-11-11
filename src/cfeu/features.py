# ---- Home Credit feature building ----
import os, json
import numpy as np
import pandas as pd


def _safe_div(a, b):
    a = np.asarray(a, dtype="float64"); b = np.asarray(b, dtype="float64")
    out = np.full_like(a, np.nan)
    m = (b != 0) & ~np.isnan(b)
    out[m] = a[m] / b[m]
    return out


def build_home_credit(hc_dir):
    """Return one row per applicant with TARGET, feature groups and audit-only fields."""
    app = pd.read_csv(os.path.join(hc_dir, "application_train.csv"), usecols=[
        "SK_ID_CURR", "TARGET", "CODE_GENDER", "DAYS_BIRTH", "DAYS_EMPLOYED",
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE",
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"])
    df = app[["SK_ID_CURR", "TARGET"]].copy()

    # Audit-only fields: never used as model inputs
    df["audit_gender"] = app["CODE_GENDER"].where(app["CODE_GENDER"].isin(["M", "F"]))
    age = -app["DAYS_BIRTH"] / 365.25
    df["audit_age"] = age.round(1)
    df["audit_age_band"] = pd.cut(age, [0, 30, 50, 200], labels=["<30", "30-49", "50+"], right=False).astype(str)
    df["contract_type"] = app["NAME_CONTRACT_TYPE"]

    # Bureau group: external scores + credit bureau records
    for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        df["bur_" + c.lower()] = app[c]

    bur = pd.read_csv(os.path.join(hc_dir, "bureau.csv"), usecols=[
        "SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE",
        "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE", "CNT_CREDIT_PROLONG"])
    bur["is_active"] = (bur["CREDIT_ACTIVE"] == "Active").astype(int)
    g = bur.groupby("SK_ID_CURR")
    b = pd.DataFrame({
        "bur_n_loans": g.size(),
        "bur_n_active": g["is_active"].sum(),
        "bur_oldest_years": -g["DAYS_CREDIT"].min() / 365.25,
        "bur_newest_years": -g["DAYS_CREDIT"].max() / 365.25,
        "bur_max_days_overdue": g["CREDIT_DAY_OVERDUE"].max(),
        "bur_total_credit": g["AMT_CREDIT_SUM"].sum(),
        "bur_total_debt": g["AMT_CREDIT_SUM_DEBT"].sum(),
        "bur_total_overdue": g["AMT_CREDIT_SUM_OVERDUE"].sum(),
        "bur_n_prolonged": g["CNT_CREDIT_PROLONG"].sum(),
    })
    b["bur_debt_to_credit"] = _safe_div(b["bur_total_debt"], b["bur_total_credit"])

    bb_path = os.path.join(hc_dir, "bureau_balance.csv")
    if os.path.exists(bb_path):
        bb = pd.read_csv(bb_path, usecols=["SK_ID_BUREAU", "STATUS"])
        bb["late"] = bb["STATUS"].astype(str).isin(list("12345")).astype(int)
        bb = bb.groupby("SK_ID_BUREAU")["late"].agg(["sum", "count"]).reset_index()
        bb = bb.merge(bur[["SK_ID_BUREAU", "SK_ID_CURR"]], on="SK_ID_BUREAU")
        bb = bb.groupby("SK_ID_CURR")[["sum", "count"]].sum()
        b["bur_late_month_share"] = _safe_div(bb["sum"].reindex(b.index), bb["count"].reindex(b.index))
    df = df.merge(b, left_on="SK_ID_CURR", right_index=True, how="left")
    df["thin_file"] = df["bur_n_loans"].isna().astype(int)
    df["bur_n_loans"] = df["bur_n_loans"].fillna(0)
    df["bur_n_active"] = df["bur_n_active"].fillna(0)

    # Alternative group: affordability from the application + repayment behaviour outside the bureau
    df["alt_income"] = app["AMT_INCOME_TOTAL"]
    df["alt_credit_to_income"] = _safe_div(app["AMT_CREDIT"], app["AMT_INCOME_TOTAL"])
    df["alt_annuity_to_income"] = _safe_div(app["AMT_ANNUITY"], app["AMT_INCOME_TOTAL"])
    df["alt_credit_to_goods"] = _safe_div(app["AMT_CREDIT"], app["AMT_GOODS_PRICE"])
    emp = app["DAYS_EMPLOYED"].where(app["DAYS_EMPLOYED"] != 365243)  # 365243 is a placeholder for not employed
    df["alt_employment_years"] = -emp / 365.25
    df["alt_not_employed"] = (app["DAYS_EMPLOYED"] == 365243).astype(int)
    df["alt_income_type"] = app["NAME_INCOME_TYPE"].astype("category").cat.codes

    prev = pd.read_csv(os.path.join(hc_dir, "previous_application.csv"),
                       usecols=["SK_ID_CURR", "NAME_CONTRACT_STATUS", "AMT_APPLICATION", "AMT_CREDIT"])
    prev["approved"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    prev["refused"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    gp = prev.groupby("SK_ID_CURR")
    p = pd.DataFrame({"alt_prev_n": gp.size(), "alt_prev_approved_share": gp["approved"].mean(),
                      "alt_prev_refused_share": gp["refused"].mean()})

    ins = pd.read_csv(os.path.join(hc_dir, "installments_payments.csv"), usecols=[
        "SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"],
        dtype={"SK_ID_CURR": "int32", "DAYS_INSTALMENT": "float32", "DAYS_ENTRY_PAYMENT": "float32",
               "AMT_INSTALMENT": "float32", "AMT_PAYMENT": "float32"})
    ins["days_late"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
    ins["is_late"] = (ins["days_late"] > 0).astype("int8")
    ins["is_short"] = (ins["AMT_PAYMENT"] < ins["AMT_INSTALMENT"] * 0.99).astype("int8")
    gi = ins.groupby("SK_ID_CURR")
    i = pd.DataFrame({"alt_inst_n": gi.size(), "alt_inst_late_share": gi["is_late"].mean(),
                      "alt_inst_mean_days_late": gi["days_late"].mean(), "alt_inst_max_days_late": gi["days_late"].max(),
                      "alt_inst_short_share": gi["is_short"].mean(),
                      "alt_inst_paid_ratio": gi["AMT_PAYMENT"].sum() / gi["AMT_INSTALMENT"].sum().replace(0, np.nan)})
    del ins

    pos = pd.read_csv(os.path.join(hc_dir, "POS_CASH_balance.csv"), usecols=["SK_ID_CURR", "SK_DPD_DEF"])
    pos["dpd"] = (pos["SK_DPD_DEF"] > 0).astype("int8")
    gq = pos.groupby("SK_ID_CURR")
    q = pd.DataFrame({"alt_pos_dpd_share": gq["dpd"].mean(), "alt_pos_max_dpd": gq["SK_DPD_DEF"].max()})
    del pos

    cc = pd.read_csv(os.path.join(hc_dir, "credit_card_balance.csv"),
                     usecols=["SK_ID_CURR", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD_DEF"])
    cc["util"] = _safe_div(cc["AMT_BALANCE"], cc["AMT_CREDIT_LIMIT_ACTUAL"])
    cc["dpd"] = (cc["SK_DPD_DEF"] > 0).astype("int8")
    gc = cc.groupby("SK_ID_CURR")
    c = pd.DataFrame({"alt_cc_util_mean": gc["util"].mean(), "alt_cc_dpd_share": gc["dpd"].mean()})
    del cc

    for t in (p, i, q, c):
        df = df.merge(t, left_on="SK_ID_CURR", right_index=True, how="left")
    df["alt_prev_n"] = df["alt_prev_n"].fillna(0)
    df["alt_inst_n"] = df["alt_inst_n"].fillna(0)
    return df


def home_credit_summary(df):
    bur_cols = [c for c in df.columns if c.startswith("bur_")]
    alt_cols = [c for c in df.columns if c.startswith("alt_")]
    out = {"n_applicants": int(len(df)), "default_rate": round(float(df["TARGET"].mean()), 4),
           "n_bureau_features": len(bur_cols), "n_alternative_features": len(alt_cols), "groups": {}}
    for name, sub in [("thin_file", df[df.thin_file == 1]), ("with_bureau", df[df.thin_file == 0])]:
        out["groups"][name] = {
            "n": int(len(sub)), "share": round(len(sub) / len(df), 4),
            "default_rate": round(float(sub["TARGET"].mean()), 4),
            "ext_source_missing_share": {c: round(float(sub[c].isna().mean()), 4)
                                         for c in ["bur_ext_source_1", "bur_ext_source_2", "bur_ext_source_3"]},
            "has_previous_loans_share": round(float((sub["alt_prev_n"] > 0).mean()), 4),
            "has_instalment_history_share": round(float((sub["alt_inst_n"] > 0).mean()), 4),
        }
    out["by_gender"] = df.groupby("audit_gender")["TARGET"].agg(["size", "mean"]).round(4).to_dict()
    out["by_age_band"] = df.groupby("audit_age_band")["TARGET"].agg(["size", "mean"]).round(4).to_dict()
    out["missing_share_top10"] = (df[bur_cols + alt_cols].isna().mean().sort_values(ascending=False)
                                  .head(10).round(4).to_dict())
    return out


# ---- Berka (PKDD'99) cash-flow features ----
CREDIT_TYPES = {"PRIJEM", "CREDIT"}
WITHDRAWAL_TYPES = {"VYDAJ", "VYBER", "WITHDRAWAL", "WITHDRAWAL IN CASH"}
SANCTION_SYMBOLS = {"SANKC. UROK", "SANCTION INTEREST IF NEGATIVE BALANCE"}


def _to_date(s):
    """Accept DATE values or YYMMDD integers from the original files."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    st = s.astype(str).str.replace(r"\.0$", "", regex=True)
    if st.str.fullmatch(r"\d{6}").all():
        return pd.to_datetime("19" + st, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(st, errors="coerce")


def _gender_birth_from_number(bn):
    """Birth number YYMMDD; women have 50 added to the month."""
    st = bn.astype(str).str.zfill(6)
    mm = st.str[2:4].astype(int)
    gender = np.where(mm > 50, "F", "M")
    mm = np.where(mm > 50, mm - 50, mm)
    birth = pd.to_datetime("19" + st.str[:2] + pd.Series(mm, index=st.index).astype(str).str.zfill(2) + st.str[4:6],
                           format="%Y%m%d", errors="coerce")
    return pd.Series(gender, index=bn.index), birth


def build_berka(tables, window_days=365, min_history_days=90, gap_days=0):
    """tables: dict of DataFrames with keys loan, account, trans, disp, client. One row per loan."""
    loan = tables["loan"].copy(); acc = tables["account"].copy(); tr = tables["trans"].copy()
    disp = tables["disp"].copy(); cl = tables["client"].copy()
    for t in (loan, acc, tr, disp, cl):
        t.columns = [c.lower() for c in t.columns]
    loan["date"] = _to_date(loan["date"]); acc["date"] = _to_date(acc["date"]); tr["date"] = _to_date(tr["date"])
    tr["type"] = tr["type"].astype(str).str.upper().str.strip()
    tr["k_symbol"] = tr["k_symbol"].astype(str).str.upper().str.strip()

    loan["default"] = loan["status"].astype(str).str.upper().isin(["B", "D"]).astype(int)
    loan = loan.merge(acc[["account_id", "date"]].rename(columns={"date": "acct_open"}), on="account_id", how="left")

    owner = disp[disp["type"].astype(str).str.upper() == "OWNER"][["account_id", "client_id"]]
    if "birth_number" in cl.columns:
        cl["audit_gender"], cl["birth"] = _gender_birth_from_number(cl["birth_number"])
    else:  # pre-processed versions carry gender and birth_date directly
        cl["audit_gender"] = cl["gender"].astype(str).str.upper().str[0]
        cl["birth"] = pd.to_datetime(cl["birth_date"], errors="coerce")
    owner = owner.merge(cl[["client_id", "audit_gender", "birth"]], on="client_id", how="left").drop_duplicates("account_id")
    loan = loan.merge(owner[["account_id", "audit_gender", "birth"]], on="account_id", how="left")
    loan["audit_age"] = ((loan["date"] - loan["birth"]).dt.days / 365.25).round(1)

    tr = tr.merge(loan[["account_id", "date"]].rename(columns={"date": "loan_date"}), on="account_id")
    end = tr["loan_date"] - pd.Timedelta(days=gap_days)  # gap_days > 0 leaves a buffer before the loan date
    tr = tr[(tr["date"] < end) & (tr["date"] >= end - pd.Timedelta(days=window_days))]
    tr["is_credit"] = tr["type"].isin(CREDIT_TYPES)
    tr["inflow"] = np.where(tr["is_credit"], tr["amount"].abs(), 0.0)
    tr["outflow"] = np.where(~tr["is_credit"], tr["amount"].abs(), 0.0)
    tr["cash_out"] = np.where(tr["operation"].astype(str).str.upper().str.contains("VYBER|CASH"), tr["amount"].abs(), 0.0)
    tr["sanction"] = tr["k_symbol"].isin(SANCTION_SYMBOLS).astype(int)
    tr["month"] = tr["date"].dt.to_period("M")

    m = tr.groupby(["account_id", "month"]).agg(inflow=("inflow", "sum"), outflow=("outflow", "sum"),
                                                min_bal=("balance", "min")).reset_index()
    gm = m.groupby("account_id")
    feats = pd.DataFrame({
        "cf_months_observed": gm.size(),
        "cf_inflow_monthly_mean": gm["inflow"].mean(),
        "cf_inflow_cv": gm["inflow"].std() / gm["inflow"].mean().replace(0, np.nan),
        "cf_expense_to_income": gm["outflow"].sum() / gm["inflow"].sum().replace(0, np.nan),
        "cf_months_net_negative": (m.assign(neg=m["outflow"] > m["inflow"]).groupby("account_id")["neg"].mean()),
        "cf_min_balance": gm["min_bal"].min(),
    })
    gt = tr.groupby("account_id")
    feats["cf_balance_mean"] = gt["balance"].mean()
    feats["cf_balance_volatility"] = gt["balance"].std() / gt["balance"].mean().abs().replace(0, np.nan)
    feats["cf_low_balance_share"] = gt["balance"].apply(lambda b: float((b < 5000).mean()))
    feats["cf_negative_balance_share"] = gt["balance"].apply(lambda b: float((b < 0).mean()))
    feats["cf_sanction_count"] = gt["sanction"].sum()
    feats["cf_cash_withdrawal_share"] = gt["cash_out"].sum() / gt["outflow"].sum().replace(0, np.nan)
    feats["cf_txn_per_month"] = gt.size() / feats["cf_months_observed"]
    first = gt["date"].min()

    out = loan.merge(feats, left_on="account_id", right_index=True, how="left")
    out = out.merge(first.rename("first_txn"), left_on="account_id", right_index=True, how="left")
    out["cf_history_days"] = (out["date"] - out["first_txn"]).dt.days
    out["cf_account_tenure_years"] = (out["date"] - out["acct_open"]).dt.days / 365.25
    out["terms_amount"] = out["amount"]; out["terms_duration"] = out["duration"]; out["terms_payment"] = out["payments"]
    out["cf_payment_to_income"] = out["payments"] / out["cf_inflow_monthly_mean"].replace(0, np.nan)
    out["enough_history"] = (out["cf_history_days"] >= min_history_days).astype(int)
    keep = (["loan_id", "account_id", "date", "status", "default", "enough_history", "audit_gender", "audit_age"]
            + [c for c in out.columns if c.startswith("cf_") or c.startswith("terms_")])
    return out[keep].rename(columns={"date": "loan_date"})


def berka_summary(df):
    cf = [c for c in df.columns if c.startswith("cf_")]
    return {"n_loans": int(len(df)), "n_defaults": int(df["default"].sum()),
            "default_rate": round(float(df["default"].mean()), 4),
            "status_counts": df["status"].astype(str).value_counts().to_dict(),
            "loans_with_90d_history": int(df["enough_history"].sum()),
            "n_cashflow_features": len(cf),
            "by_gender": df.groupby("audit_gender")["default"].agg(["size", "mean"]).round(4).to_dict(),
            "cashflow_feature_means_by_outcome": df.groupby("default")[cf].mean().round(3).to_dict()}
