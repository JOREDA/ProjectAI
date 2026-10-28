"""
First version of the team-level attrition risk model.
Target: will the team lose >= 10% of its people in the next 3 months?
Unit: team x month, teams with >= 5 people only (k-anonymity rule).
Split by time (no shuffling) so the test set is genuinely 'the future'.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
p = pd.read_csv(os.path.join(HERE, "data", "team_monthly_panel.csv"))
p = p[(p.headcount_start >= 5) & p.target_high_risk_next_3m.notna()].copy()
p = p.sort_values(["team_id", "month"])
g = p.groupby("team_id")

# rolling 3-month comment signals (single months are sparse)
roll = lambda col: g[col].transform(lambda s: s.rolling(3, min_periods=1).sum())
p["complaint_score_3m"] = 10 * roll("severity_sum") / p.headcount_start
p["comments_per_head_3m"] = roll("n_comments") / p.headcount_start
for th in ["workload", "pay", "career_growth", "manager_behaviour", "work_location"]:
    p[f"share_{th}_3m"] = roll(f"n_theme_{th}") / roll("n_comments").clip(lower=1)
p["escalations_3m"] = roll("n_escalations")
p["exit_rate_last_3m"] = p.exits_last_3m / p.headcount_start
p["promo_rate_12m"] = p.promotions_last_12m / p.headcount_start
p["transfer_req_rate"] = p.internal_transfer_requests / p.headcount_start

FEATURES = {
    "avg_overtime_hrs_week": "Overtime hours / week",
    "weekend_logins_per_emp": "Weekend logins",
    "unplanned_leave_days_per_emp": "Unplanned leave",
    "pulse_score_1to5": "Team pulse score",
    "manager_pulse_score_1to5": "Manager rating",
    "pay_vs_market_pct": "Pay vs market",
    "last_hike_pct": "Last hike %",
    "promo_rate_12m": "Promotion rate (12m)",
    "manager_changed_last_6m": "Manager changed (6m)",
    "transfer_req_rate": "Transfer requests",
    "rto_days_required": "Office days required",
    "avg_commute_km": "Commute distance",
    "pct_under_12m_tenure": "Share of new joiners",
    "exit_rate_last_3m": "Exits last 3 months",
    "complaint_score_3m": "Complaint score (3m)",
    "comments_per_head_3m": "Comment volume (3m)",
    "share_workload_3m": "Workload complaints share",
    "share_pay_3m": "Pay complaints share",
    "share_career_growth_3m": "Growth complaints share",
    "share_manager_behaviour_3m": "Manager complaints share",
    "share_work_location_3m": "Office-return complaints share",
    "escalations_3m": "Escalations (3m)",
}
X_cols = list(FEATURES)
y_col = "target_high_risk_next_3m"


def scores(y, pred, prob=None):
    out = dict(precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred),
               f1=f1_score(y, pred), flagged=int(pred.sum()), positives=int(y.sum()), n=len(y))
    if prob is not None:
        out["auc"] = roc_auc_score(y, prob)
    return out


def fit_models(train):
    lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", C=0.5, max_iter=2000))
    lr.fit(train[X_cols], train[y_col])
    w = np.where(train[y_col] == 1, (train[y_col] == 0).sum() / (train[y_col] == 1).sum(), 1.0)
    gb = GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)
    gb.fit(train[X_cols], train[y_col], sample_weight=w)
    return lr, gb


results = {}
# ---- Main split: train Nov-24..Dec-25, test Jan-26..Jul-26
train = p[p.month <= "2025-12"]
test = p[(p.month >= "2026-01") & (p.month <= "2026-07")]
lr, gb = fit_models(train)
y = test[y_col].values
base_pred = (test.exit_rate_last_3m >= 0.10).astype(int).values
results["baseline_history_rule"] = scores(y, base_pred)
lr_prob = lr.predict_proba(test[X_cols])[:, 1]
gb_prob = gb.predict_proba(test[X_cols])[:, 1]
results["logistic_regression"] = scores(y, (lr_prob >= 0.5).astype(int), lr_prob)
results["gradient_boosting"] = scores(y, (gb_prob >= 0.5).astype(int), gb_prob)
results["split"] = dict(train_rows=len(train), test_rows=len(test), train_pos_rate=float(train[y_col].mean()),
                        test_pos_rate=float(y.mean()))

# before vs after the office-return mandate (drift check)
pre = test.month <= "2026-03"
for name, prob in [("logistic_regression", lr_prob)]:
    results["drift_lr_pre_rto"] = scores(y[pre.values], (prob[pre.values] >= 0.5).astype(int), prob[pre.values])
    results["drift_lr_post_rto"] = scores(y[~pre.values], (prob[~pre.values] >= 0.5).astype(int), prob[~pre.values])

# ---- Retraining check: same test window (May-Jul 26), old model vs model retrained on data up to Apr-26
test2 = p[(p.month >= "2026-05") & (p.month <= "2026-07")]
y2 = test2[y_col].values
old_prob = lr.predict_proba(test2[X_cols])[:, 1]
lr_new, _ = fit_models(p[p.month <= "2026-04"])
new_prob = lr_new.predict_proba(test2[X_cols])[:, 1]
results["retrain_old_model_may_jul"] = scores(y2, (old_prob >= 0.5).astype(int), old_prob)
results["retrain_new_model_may_jul"] = scores(y2, (new_prob >= 0.5).astype(int), new_prob)

# ---- Drivers: standardised logistic coefficients (model retrained to Apr-26 so office-return is learnable)
coef = pd.Series(lr_new[-1].coef_[0], index=[FEATURES[c] for c in X_cols]).sort_values(key=abs, ascending=False)
results["top_drivers"] = coef.head(10).round(3).to_dict()

# ---- Example: why is a flagged team high risk? (per-team contribution = coef x standardised value)
scaler = lr_new[0]
latest = p[p.month == "2026-07"].copy()
latest["risk"] = lr_new.predict_proba(latest[X_cols])[:, 1]
top = latest.sort_values("risk", ascending=False).head(3)
z = (top[X_cols] - scaler.mean_) / scaler.scale_
contrib = z * lr_new[-1].coef_[0]
examples = []
for i, (_, row) in enumerate(top.iterrows()):
    c = contrib.iloc[i].sort_values(ascending=False).head(3)
    examples.append(dict(team=row.team_id, dept=row.department, location=row.location,
                         risk=round(float(row.risk), 2),
                         actual_next_3m_attrition_pct=float(row.attrition_rate_next_3m_pct),
                         reasons=[FEATURES[k] for k in c.index]))
results["example_flags_jul26"] = examples

json.dump(results, open(os.path.join(HERE, "model_v1_results.json"), "w"), indent=2, default=float)
pd.set_option("display.width", 200)
summary = pd.DataFrame({k: v for k, v in results.items()
                        if isinstance(v, dict) and "precision" in v}).T[["n", "positives", "flagged", "precision", "recall", "f1", "auc"]]
print(summary.round(3).to_string())
print(json.dumps(results["split"], indent=1))
print(coef.round(3).to_string())
print(json.dumps(examples, indent=1))
