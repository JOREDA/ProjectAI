"""
risk_model.py - Attrition risk predictive model, driver attribution, and retraining analysis.
Unit: team x month, teams with headcount >= min_group_size only.
"""
import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "attrition_model.joblib")

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
X_COLS = list(FEATURES.keys())
Y_COL = "target_high_risk_next_3m"


def prepare_features(df_panel: pd.DataFrame, min_group_size: int = 5) -> pd.DataFrame:
    """
    Computes rolling 3-month features and filters teams >= min_group_size.
    Matches data/train_model_v1.py exactly.
    """
    p = df_panel.copy()
    # Ensure numeric columns
    numeric_cols = [
        "headcount_start", "severity_sum", "n_comments", "exits_last_3m",
        "promotions_last_12m", "internal_transfer_requests", "avg_overtime_hrs_week",
        "weekend_logins_per_emp", "unplanned_leave_days_per_emp", "pulse_score_1to5",
        "manager_pulse_score_1to5", "pay_vs_market_pct", "last_hike_pct",
        "manager_changed_last_6m", "rto_days_required", "avg_commute_km",
        "pct_under_12m_tenure", "n_escalations", "complaint_score_per_10_emp"
    ]
    for th in ["workload", "pay", "career_growth", "manager_behaviour", "work_location"]:
        numeric_cols.append(f"n_theme_{th}")

    for col in numeric_cols:
        if col in p.columns:
            p[col] = pd.to_numeric(p[col], errors="coerce").fillna(0)

    if Y_COL in p.columns:
        p[Y_COL] = pd.to_numeric(p[Y_COL], errors="coerce")

    # Filter to teams with headcount >= min_group_size
    p = p[p["headcount_start"] >= min_group_size].copy()
    p = p.sort_values(["team_id", "month"]).reset_index(drop=True)
    g = p.groupby("team_id")

    roll = lambda col: g[col].transform(lambda s: s.rolling(3, min_periods=1).sum())
    p["complaint_score_3m"] = 10 * roll("severity_sum") / p["headcount_start"].clip(lower=1)
    p["comments_per_head_3m"] = roll("n_comments") / p["headcount_start"].clip(lower=1)
    for th in ["workload", "pay", "career_growth", "manager_behaviour", "work_location"]:
        p[f"share_{th}_3m"] = roll(f"n_theme_{th}") / roll("n_comments").clip(lower=1)
    p["escalations_3m"] = roll("n_escalations")
    p["exit_rate_last_3m"] = p["exits_last_3m"] / p["headcount_start"].clip(lower=1)
    p["promo_rate_12m"] = p["promotions_last_12m"] / p["headcount_start"].clip(lower=1)
    p["transfer_req_rate"] = p["internal_transfer_requests"] / p["headcount_start"].clip(lower=1)

    # 3-month average complaint score for spike calculation
    p["complaint_3m_avg"] = g["complaint_score_per_10_emp"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )

    return p


def fit_model(train_df: pd.DataFrame) -> Any:
    """
    Fits LogisticRegression(class_weight='balanced', C=0.5, max_iter=2000) inside StandardScaler pipeline.
    """
    valid = train_df[train_df[Y_COL].notna()].copy()
    pipe = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", C=0.5, max_iter=2000))
    pipe.fit(valid[X_COLS], valid[Y_COL])
    return pipe


def train_or_load_model(pnl_prepared: pd.DataFrame, cutoff_month: str = "2025-12", force_retrain: bool = False) -> Any:
    """
    Trains standard model up to cutoff_month (default 2025-12) or loads from disk.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not force_retrain and os.path.exists(MODEL_PATH) and cutoff_month == "2025-12":
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass

    train_data = pnl_prepared[pnl_prepared["month"] <= cutoff_month]
    model = fit_model(train_data)
    if cutoff_month == "2025-12":
        try:
            joblib.dump(model, MODEL_PATH)
        except Exception:
            pass
    return model


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Computes precision, recall, f1, auc, flagged count, positives count.
    """
    res = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "flagged": int(np.sum(y_pred)),
        "positives": int(np.sum(y_true)),
        "n": len(y_true),
    }
    if y_prob is not None:
        try:
            res["auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            res["auc"] = None
    return res


def evaluate_baseline_and_model(
    model: Any,
    pnl_prepared: pd.DataFrame,
    test_start: str = "2026-01",
    test_end: str = "2026-07",
    threshold: float = 0.50,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[str]]:
    """
    Evaluates baseline (historical exits >= 10%) vs Logistic Regression model.
    Checks if numbers differ from expected by more than 2 percentage points.
    Expected: baseline precision 43%, recall 30%; model precision 42%, recall 59%.
    """
    test = pnl_prepared[
        (pnl_prepared["month"] >= test_start)
        & (pnl_prepared["month"] <= test_end)
        & pnl_prepared[Y_COL].notna()
    ].copy()

    y_true = test[Y_COL].values.astype(int)
    base_pred = (test["exit_rate_last_3m"] >= 0.10).astype(int).values
    baseline_metrics = calculate_metrics(y_true, base_pred)

    probs = model.predict_proba(test[X_COLS])[:, 1]
    model_pred = (probs >= threshold).astype(int)
    model_metrics = calculate_metrics(y_true, model_pred, probs)

    warning = None
    # Check 2 percentage point deviation
    diff_bp = abs(baseline_metrics["precision"] - 0.43)
    diff_br = abs(baseline_metrics["recall"] - 0.30)
    diff_mp = abs(model_metrics["precision"] - 0.42)
    diff_mr = abs(model_metrics["recall"] - 0.59)

    if any(d > 0.025 for d in [diff_bp, diff_br, diff_mp, diff_mr]):
        warning = (
            f"Metrics variance detected (>2%): Baseline Prec {baseline_metrics['precision']:.1%} (exp 43%), "
            f"Rec {baseline_metrics['recall']:.1%} (exp 30%) | "
            f"Model Prec {model_metrics['precision']:.1%} (exp 42%), Rec {model_metrics['recall']:.1%} (exp 59%)."
        )

    return baseline_metrics, model_metrics, warning


def get_top_model_drivers(model: Any, top_n: int = 10) -> pd.Series:
    """
    Returns the standardized logistic regression coefficients sorted by absolute magnitude.
    """
    coefs = model.named_steps["logisticregression"].coef_[0]
    s = pd.Series(coefs, index=[FEATURES[c] for c in X_COLS])
    return s.sort_values(key=abs, ascending=False).head(top_n)


def score_teams_for_month(
    model: Any,
    pnl_prepared: pd.DataFrame,
    month: str,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Scores teams for a specific month and attaches:
    - risk score (0-1)
    - alarm flag (model risk >= threshold OR exits_last_3m >= 10% OR complaint spike)
    - alarm reasons (top 3 drivers pushing risk up: coef * standardized_value)
    """
    month_data = pnl_prepared[pnl_prepared["month"] == month].copy()
    if month_data.empty:
        return month_data

    scaler = model.named_steps["standardscaler"]
    clf = model.named_steps["logisticregression"]
    coefs = clf.coef_[0]

    probs = model.predict_proba(month_data[X_COLS])[:, 1]
    month_data["risk_score"] = probs
    month_data["risk_pct"] = (probs * 100).round(1)

    # Standardized contributions: z * coef
    z_scores = (month_data[X_COLS] - scaler.mean_) / scaler.scale_
    contribs = z_scores * coefs

    # Historical exit rule and complaint spike (top 5% across full panel)
    spike_cutoff = pnl_prepared["complaint_3m_avg"].quantile(0.95)
    month_data["hist_exit_flag"] = month_data["exit_rate_last_3m"] >= 0.10
    month_data["spike_flag"] = month_data["complaint_3m_avg"] >= spike_cutoff
    month_data["model_risk_flag"] = month_data["risk_score"] >= threshold

    month_data["is_alarm"] = (
        month_data["model_risk_flag"]
        | month_data["hist_exit_flag"]
        | month_data["spike_flag"]
    )

    reasons_list = []
    for i in range(len(month_data)):
        row_contrib = contribs.iloc[i].sort_values(ascending=False).head(3)
        top_reasons = [FEATURES[k] for k in row_contrib.index]
        reasons_list.append(", ".join(top_reasons))

    month_data["top_drivers"] = reasons_list
    return month_data


def get_team_summary_aggregates(
    pnl_prepared: pd.DataFrame,
    comments_df: pd.DataFrame,
    team_id: str,
    month: str,
    model: Any,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Builds structured aggregates for Call D (Team summariser):
    - risk score
    - 3 model reasons
    - top 3 complaint themes last 3 months with counts
    - Policy vs Person split
    - complaint-score trend
    - up to 5 anonymised final_text comments
    """
    month_teams = score_teams_for_month(model, pnl_prepared, month, threshold)
    team_row = month_teams[month_teams["team_id"] == team_id]

    if team_row.empty:
        return {}

    row = team_row.iloc[0]
    risk_pct = round(float(row["risk_score"]) * 100)
    drivers = str(row["top_drivers"]).split(", ")

    # Trailing 3-month window
    m = pd.Period(month, "M")
    m_window = [str(m - 2), str(m - 1), str(m)]

    team_comments = comments_df[
        (comments_df["team_id"] == team_id)
        & (comments_df["month"].isin(m_window))
        & (~comments_df["gold_domain"].isin(["Not feedback"]))
    ].copy()

    theme_counts = team_comments["gold_theme"].value_counts()
    top_themes = [f"{t} ({c})" for t, c in theme_counts.head(3).items()]

    tgt_counts = team_comments["gold_issue_target"].value_counts()
    n_policy = int(tgt_counts.get("Policy", 0))
    n_person = int(tgt_counts.get("Person", 0))
    n_both = int(tgt_counts.get("Both", 0))
    total_tgt = max(1, n_policy + n_person + n_both)
    person_pct = round(100 * (n_person + n_both) / total_tgt)
    policy_pct = round(100 * n_policy / total_tgt)

    # Complaint score trend
    history = pnl_prepared[(pnl_prepared["team_id"] == team_id) & (pnl_prepared["month"] <= month)].sort_values("month")
    if len(history) >= 2:
        prev_score = float(history.iloc[-2].get("complaint_score_per_10_emp", 0))
        curr_score = float(row.get("complaint_score_per_10_emp", 0))
        trend = "rising" if curr_score > prev_score else ("falling" if curr_score < prev_score else "stable")
    else:
        trend = "stable"

    # Up to 5 anonymised final_text comments (NEVER raw text or IDs)
    sample_comments = team_comments["gold_final_text"].dropna().head(5).tolist()

    return {
        "team_id": team_id,
        "department": row.get("department", "Unknown"),
        "location": row.get("location", "Unknown"),
        "month": month,
        "risk_pct": risk_pct,
        "is_alarm": bool(row["is_alarm"]),
        "drivers": drivers,
        "top_themes": top_themes,
        "person_pct": person_pct,
        "policy_pct": policy_pct,
        "trend": trend,
        "comment_count_3m": len(team_comments),
        "sample_final_texts": sample_comments,
    }


def run_retraining_comparison(
    pnl_prepared: pd.DataFrame,
    cutoff_month: str = "2026-04",
    test_start: str = "2026-05",
    test_end: str = "2026-07",
) -> Dict[str, Any]:
    """
    Evaluates old model (trained up to 2025-12) vs new model (trained up to cutoff_month)
    on the test window immediately following cutoff.
    Visible improvement after April 2026 return-to-office mandate.
    """
    test = pnl_prepared[
        (pnl_prepared["month"] >= test_start)
        & (pnl_prepared["month"] <= test_end)
        & pnl_prepared[Y_COL].notna()
    ].copy()

    y_test = test[Y_COL].values.astype(int)

    old_model = fit_model(pnl_prepared[pnl_prepared["month"] <= "2025-12"])
    new_model = fit_model(pnl_prepared[pnl_prepared["month"] <= cutoff_month])

    old_prob = old_model.predict_proba(test[X_COLS])[:, 1]
    new_prob = new_model.predict_proba(test[X_COLS])[:, 1]

    old_metrics = calculate_metrics(y_test, (old_prob >= 0.5).astype(int), old_prob)
    new_metrics = calculate_metrics(y_test, (new_prob >= 0.5).astype(int), new_prob)

    return {
        "cutoff_month": cutoff_month,
        "test_window": f"{test_start} to {test_end}",
        "old_model": old_metrics,
        "new_model": new_metrics,
        "recall_improvement": round(new_metrics["recall"] - old_metrics["recall"], 3),
    }
