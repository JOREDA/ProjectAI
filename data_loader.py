"""
data_loader.py - Data ingestion, startup verification, rule file parsing, and privacy rollup.
"""
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import dotenv

try:
    import streamlit as st
except ImportError:
    st = None

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REQUIRED_FILES = [
    "data/data/feedback_comments.csv",
    "data/data/team_monthly_panel.csv",
    "data/data/teams.csv",
    "data/data/employees_hris.csv",
    "data/data/listening_sessions.csv",
    "data/data/manager_yearend_review.csv",
    "data/data/hr_review_disputes.csv",
    "data/data/policy_events.csv",
    "data/data/identity_vault_SEALED.csv",
    "data/llm/llm_output_schema.json",
    "data/llm/redaction_lists.json",
    "data/llm/test_set_input.jsonl",
    "data/llm/test_set_gold.jsonl",
    "anonymisation-rules.md",
    "classification-rules.md",
    "hr-policy.md",
]


def check_startup_requirements() -> Tuple[bool, List[str]]:
    """
    Checks that all required files and deployment credentials are present.
    Returns (is_valid, list_of_missing_items).
    """
    missing = []
    for rel_path in REQUIRED_FILES:
        full_path = os.path.join(BASE_DIR, rel_path)
        if not os.path.exists(full_path):
            missing.append(rel_path)

    api_key = get_groq_api_key()
    if not api_key or not api_key.strip():
        missing.append("Groq API Key (not found in st.secrets, .env, or credentials.json)")

    ombudsperson_passcode = get_ombudsperson_passcode()
    if not ombudsperson_passcode or not ombudsperson_passcode.strip():
        missing.append("Ombudsperson passcode (not found in st.secrets, .env, or credentials.json)")

    return (len(missing) == 0, missing)


def get_groq_api_key() -> Optional[str]:
    """
    Retrieves Groq API key in exact priority order:
    1. st.secrets["groq_api_key"]
    2. GROQ_API_KEY environment variable (via .env)
    3. credentials.json -> {"groq_api_key": "gsk_..."}
    Never logs or prints the key.
    """
    # 1. Streamlit secrets
    if st is not None:
        try:
            if "groq_api_key" in st.secrets:
                key = str(st.secrets["groq_api_key"]).strip()
                if key and not key.startswith("${"):
                    return key
        except Exception:
            pass

    # 2. Environment variable (.env)
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        dotenv.load_dotenv(env_path, override=True)
    env_key = os.environ.get("GROQ_API_KEY", "").strip()
    if env_key and not env_key.startswith("${"):
        return env_key

    # 3. credentials.json
    cred_path = os.path.join(BASE_DIR, "credentials.json")
    if os.path.exists(cred_path):
        try:
            with open(cred_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                cred_key = str(data.get("groq_api_key", "")).strip()
                # Check if it was an unresolved variable
                if cred_key and not cred_key.startswith("${"):
                    return cred_key
                elif cred_key == "${GROQ_API_KEY}":
                    if env_key:
                        return env_key
        except Exception:
            pass

    return None


def get_ombudsperson_passcode() -> Optional[str]:
    """
    Retrieves ombudsperson passcode from secrets, .env, or credentials.json.
    """
    if st is not None:
        try:
            if "ombudsperson_passcode" in st.secrets:
                return str(st.secrets["ombudsperson_passcode"]).strip()
        except Exception:
            pass

    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        dotenv.load_dotenv(env_path, override=True)
    env_code = os.environ.get("OMBUDSPERSON_PASSCODE", "").strip()
    if env_code:
        return env_code

    cred_path = os.path.join(BASE_DIR, "credentials.json")
    if os.path.exists(cred_path):
        try:
            with open(cred_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("ombudsperson_passcode")
        except Exception:
            pass

    return None


def read_rules_files() -> Dict[str, str]:
    """
    Reads markdown rule files fresh from disk on every invocation.
    Never cached across runs.
    """
    rule_files = {
        "anonymisation": os.path.join(BASE_DIR, "anonymisation-rules.md"),
        "classification": os.path.join(BASE_DIR, "classification-rules.md"),
        "hr_policy": os.path.join(BASE_DIR, "hr-policy.md"),
    }
    contents = {}
    for key, path in rule_files.items():
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                contents[key] = f.read()
        else:
            contents[key] = ""
    return contents


def compute_rules_hash() -> str:
    """
    Computes a combined sha256 hash of all 3 rule files fresh from disk.
    If any rule file changes, this hash changes, triggering re-processing.
    """
    contents = read_rules_files()
    combined = (
        contents.get("anonymisation", "")
        + "|||"
        + contents.get("classification", "")
        + "|||"
        + contents.get("hr_policy", "")
    )
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def parse_hr_policy_parameters() -> Dict[str, Any]:
    """
    Extracts dynamic policy values from hr-policy.md (e.g. min_group_size, weights, thresholds).
    Falls back to safe standard defaults if not explicitly found.
    """
    rules = read_rules_files()
    text = rules.get("hr_policy", "")

    params = {
        "min_group_size": 5,
        "alarm_threshold": 0.50,
        "listening_session_target_days": 14,
        "weights": {
            "attrition": 0.30,
            "complaint_score": 0.25,
            "trend": 0.15,
            "person_share": 0.15,
            "actions_gap": 0.15,
        },
        "dispute_window_days": 7,
    }

    # Extract min_group_size
    m_size = re.search(r"min_group_size`?\s*:\s*`?(\d+)`?", text, re.IGNORECASE)
    if not m_size:
        m_size = re.search(r"Minimum Group Size.*?(\d+)", text, re.IGNORECASE)
    if m_size:
        try:
            params["min_group_size"] = int(m_size.group(1))
        except ValueError:
            pass

    # Extract alarm threshold
    m_alarm = re.search(r"risk\s+threshold[^\d]*(\d+\.?\d*)", text, re.IGNORECASE)
    if m_alarm:
        try:
            params["alarm_threshold"] = float(m_alarm.group(1))
        except ValueError:
            pass

    return params


def load_dataset() -> Dict[str, Any]:
    """
    Loads all required CSV and JSON data.
    Ensures string types for tokens, IDs, and receipt codes.
    Never loads answer_key_hidden_drivers.csv.
    """
    str_cols = {
        "identity_token": str,
        "receipt_code": str,
        "comment_id": str,
        "employee_id": str,
        "employee_id_hidden": str,
        "team_id": str,
        "manager_id": str,
        "session_id": str,
        "dispute_id": str,
    }

    def _read_csv(rel_path: str) -> pd.DataFrame:
        full_path = os.path.join(BASE_DIR, rel_path)
        return pd.read_csv(
            full_path,
            keep_default_na=False,
            na_values=[""],
            dtype=str_cols,
            low_memory=False,
        )

    feedback_comments = _read_csv("data/data/feedback_comments.csv")
    team_monthly_panel = _read_csv("data/data/team_monthly_panel.csv")
    teams = _read_csv("data/data/teams.csv")
    employees_hris = _read_csv("data/data/employees_hris.csv")
    listening_sessions = _read_csv("data/data/listening_sessions.csv")
    manager_yearend_review = _read_csv("data/data/manager_yearend_review.csv")
    hr_review_disputes = _read_csv("data/data/hr_review_disputes.csv")
    policy_events = _read_csv("data/data/policy_events.csv")
    identity_vault = _read_csv("data/data/identity_vault_SEALED.csv")

    receipt_codes = feedback_comments.set_index("comment_id")["receipt_code"]
    hr_review_disputes["receipt_code"] = (
        hr_review_disputes["comment_id"].map(receipt_codes).fillna("")
    )
    hr_review_disputes["dispute_month"] = hr_review_disputes["review_window"].str.extract(
        r"(\d{4}-\d{2})", expand=False
    ).fillna("")
    hr_review_disputes["reason"] = hr_review_disputes["dispute_reason"]
    hr_review_disputes["employee_note"] = "Not recorded"
    hr_review_disputes["days_to_resolve"] = pd.to_numeric(
        hr_review_disputes["resolved_within_days"], errors="coerce"
    )

    sealed_critical = (
        feedback_comments["anonymity_mode"]
        .astype(str)
        .str.contains("sealed", case=False, na=False)
        & pd.to_numeric(feedback_comments["gold_severity"], errors="coerce").eq(4)
        & feedback_comments["gold_escalate"].astype(str).str.lower().isin(["true", "1"])
    )
    eligible_tokens = feedback_comments.loc[sealed_critical, "identity_token"].unique()
    identity_vault["token_status"] = identity_vault["identity_token"].isin(
        eligible_tokens
    ).map({True: "Eligible", False: "Restricted"})

    with open(os.path.join(BASE_DIR, "data/llm/llm_output_schema.json"), "r", encoding="utf-8") as f:
        schema = json.load(f)

    with open(os.path.join(BASE_DIR, "data/llm/redaction_lists.json"), "r", encoding="utf-8") as f:
        redaction_lists = json.load(f)

    return {
        "feedback_comments": feedback_comments,
        "team_monthly_panel": team_monthly_panel,
        "teams": teams,
        "employees_hris": employees_hris,
        "listening_sessions": listening_sessions,
        "manager_yearend_review": manager_yearend_review,
        "hr_review_disputes": hr_review_disputes,
        "policy_events": policy_events,
        "identity_vault": identity_vault,
        "schema": schema,
        "redaction_lists": redaction_lists,
    }


def enforce_minimum_group_size(
    df: pd.DataFrame,
    min_size: int = 5,
    team_col: str = "team_id",
    headcount_col: str = "headcount_start",
    dept_col: str = "department",
) -> pd.DataFrame:
    """
    Data privacy rule:
    Never show team-level comment counts, scores, themes or risk for a team with
    fewer people than min_group_size (default 5).
    Roll those teams up into 'Other teams (department)'.
    Enforced by this single function across all HR views.
    """
    if df.empty or team_col not in df.columns:
        return df

    out = df.copy()

    # Determine which rows fall below min_size
    if headcount_col in out.columns:
        # Numeric conversion if string
        counts = pd.to_numeric(out[headcount_col], errors="coerce").fillna(0)
        is_small = counts < min_size
    elif "team_size_at_submission" in out.columns:
        counts = pd.to_numeric(out["team_size_at_submission"], errors="coerce").fillna(0)
        is_small = counts < min_size
    elif "below_k5_reporting_threshold" in out.columns:
        is_small = out["below_k5_reporting_threshold"].astype(str).str.lower().isin(["true", "1"])
    else:
        # If no explicit headcount, do not modify
        return out

    if is_small.any():
        if dept_col in out.columns:
            dept_vals = out[dept_col].astype(str)
            rollup_labels = "Other teams (" + dept_vals + ")"
        else:
            rollup_labels = pd.Series(["Other teams"] * len(out), index=out.index)
        out.loc[is_small, team_col] = rollup_labels[is_small]
        if "team_name" in out.columns:
            out.loc[is_small, "team_name"] = rollup_labels[is_small]

    return out



def ensure_app_directories():
    """
    Creates cache/ and models/ directories if they don't exist.
    """
    for folder in ["cache", "models"]:
        path = os.path.join(BASE_DIR, folder)
        os.makedirs(path, exist_ok=True)
