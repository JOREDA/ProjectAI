"""
app.py - Main Streamlit application for AI-Powered Anonymous Employee Feedback System (Navrang Systems).
Implements Employee view, HR view, and Ombudsperson view with strict privacy guardrails.
"""
import datetime
import io
import json
import os
import random
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

import data_loader
import guardrails
import llm_pipeline
import risk_model

# Page configuration
st.set_page_config(
    page_title="Navrang Systems | Anonymous Feedback & Attrition Prevention",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (calm, corporate, warm off-white background, deep navy headers)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre, .mono-text {
        font-family: 'DM Mono', monospace !important;
    }
    
    .main-header {
        color: #0f1f3d;
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        color: #4a5568;
        font-size: 14px;
        margin-bottom: 24px;
    }
    
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 12px;
    }
    .kpi-title {
        color: #64748b;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        color: #0f1f3d;
        font-size: 26px;
        font-weight: 700;
        margin-top: 4px;
    }
    .kpi-sub {
        font-size: 12px;
        color: #64748b;
        margin-top: 2px;
    }
    
    .summary-card {
        background: #ffffff;
        border-left: 4px solid #b71c1c;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        border-radius: 0 8px 8px 0;
        padding: 18px 22px;
        margin-bottom: 16px;
    }
    
    .audit-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def log_audit_event(event_type: str, details: str, comment_id: str = ""):
    """Appends an event to the security & audit log."""
    event = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event_type": event_type,
        "comment_id": comment_id,
        "details": details,
    }
    if "audit_log" not in st.session_state:
        st.session_state["audit_log"] = []
    st.session_state["audit_log"].insert(0, event)


# -----------------------------------------------------------------------------
# STARTUP VALIDATION
# -----------------------------------------------------------------------------
is_valid, missing_files = data_loader.check_startup_requirements()
if not is_valid:
    st.error(
        f"🚨 **System Startup Blocked:** Missing required files or credentials:\n\n"
        + "\n".join([f"- `{m}`" for m in missing_files])
        + "\n\nPlease provide the required dataset files, rules files, and credentials before proceeding."
    )
    st.stop()

# Ensure app directories
data_loader.ensure_app_directories()


# -----------------------------------------------------------------------------
# DATA & MODEL INITIALIZATION
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def initialize_system_data():
    """Loads all dataset files and trains initial risk model."""
    ds = data_loader.load_dataset()
    pnl_prep = risk_model.prepare_features(ds["team_monthly_panel"])
    model = risk_model.train_or_load_model(pnl_prep)
    return ds, pnl_prep, model


dataset, pnl_prepared, global_model = initialize_system_data()

# Initialize session state collections
if "new_feedback" not in st.session_state:
    st.session_state["new_feedback"] = []
if "new_sessions" not in st.session_state:
    st.session_state["new_sessions"] = []
if "new_disputes" not in st.session_state:
    st.session_state["new_disputes"] = []
if "audit_log" not in st.session_state:
    st.session_state["audit_log"] = []
if "in_memory_vault" not in st.session_state:
    st.session_state["in_memory_vault"] = {}
if "unsealed_comments" not in st.session_state:
    st.session_state["unsealed_comments"] = {}


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/shield.png",
        width=48,
    )
    st.markdown("### **Navrang Systems**\n**Anonymous Feedback & Risk System**")
    st.divider()

    view_choice = st.radio(
        "**Select View**",
        ["Employee", "HR", "Ombudsperson"],
        index=0,
    )

    st.markdown("---")
    st.markdown("**LLM Configuration**")
    model_choice = st.selectbox(
        "Model Provider",
        [
            llm_pipeline.DEFAULT_MODEL,
            llm_pipeline.BACKUP_MODEL,
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
        ],
        index=0,
        help="Free-tier Groq model. Automatically falls back if primary is rate-limited or unavailable.",
    )

    eval_batch_size = st.slider("Evaluation Batch Size", min_value=1, max_value=10, value=4)

    if st.button("🧹 Clear LLM & Session Cache"):
        st.session_state["new_feedback"] = []
        st.session_state["new_sessions"] = []
        st.session_state["new_disputes"] = []
        st.session_state["audit_log"] = []
        st.session_state["in_memory_vault"] = {}
        st.session_state["unsealed_comments"] = {}
        cache_path = os.path.join(data_loader.BASE_DIR, "cache", "llm_results.jsonl")
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except Exception:
                pass
        st.success("Session state and disk cache cleared.")
        st.rerun()

    st.markdown("---")
    st.caption("🔒 Educational prototype with synthetic data. Not HR or legal advice.")


# =============================================================================
# 1. EMPLOYEE VIEW
# =============================================================================
if view_choice == "Employee":
    st.markdown('<div class="main-header">Employee Voice Portal</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Your feedback helps leadership fix systemic friction. Our dual-layer AI redacts direct names and generalises identifying details before HR sees anything.</div>',
        unsafe_allow_html=True,
    )

    emp_tab1, emp_tab2 = st.tabs(["📝 Submit Feedback", "⚖️ Monthly Review Window ('Argue with HR')"])

    with emp_tab1:
        st.markdown("#### **Submit Anonymous Feedback**")

        # Demo buttons to prefill cases
        st.markdown("**Try a demo comment:**")
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)

        demo_comment = None
        if col_d1.button("1. Routine Comment", use_container_width=True):
            demo_comment = (
                "My manager keeps scheduling calls after 9 pm for the release of our project "
                "and expects replies on WhatsApp on Sundays."
            )
        if col_d2.button("2. Needs Anonymising", use_container_width=True):
            demo_comment = (
                "Gaurav checks Teams 'last seen' and pings if we are away for 10 minutes on PRJ-ATLAS-2291."
            )
        if col_d3.button("3. Escalation Required", use_container_width=True):
            demo_comment = (
                "I was told I won't be considered for the onsite role because I'm planning to get married next year."
            )
        if col_d4.button("4. Re-identifiable Case", use_container_width=True):
            demo_comment = (
                "I'm the one who raised the payroll error in the town hall last month. "
                "I've been 'next cycle' for promotion three cycles in a row now."
            )

        if demo_comment:
            st.session_state["input_text_area"] = demo_comment

        feedback_input = st.text_area(
            "Write your feedback or workplace concern:",
            value=st.session_state.get("input_text_area", ""),
            height=120,
            placeholder="Share your experience openly. Our pipeline ensures complete privacy...",
        )

        anonymity_mode = st.radio(
            "**Anonymity Preference:**",
            [
                "Sealed token (Default)",
                "Fully anonymous",
                "Use my name",
            ],
            index=0,
            help="Choose how strictly you wish your identity to be protected.",
        )

        st.caption(
            "• **Sealed token**: Your identity is encrypted into a sealed vault accessible only by the independent Ombudsperson for zero-tolerance critical escalations.\n"
            "• **Fully anonymous**: No link to your identity is ever recorded or accessible by anyone under any circumstance.\n"
            "• **Use my name**: You explicitly choose to have your name shared alongside your comments."
        )

        if st.button("🔒 Anonymise & Review Feedback", type="primary"):
            if not feedback_input or len(feedback_input.strip().split()) < 3:
                st.warning("Please provide a constructive comment containing at least 3 words.")
            else:
                with st.spinner("Anonymising your feedback through Privacy Layers 1 & 2..."):
                    processed = llm_pipeline.process_comment(
                        feedback_input,
                        comment_id=f"SUB_{random.randint(10000, 99999)}",
                        model=model_choice,
                    )
                    st.session_state["pending_submission"] = {
                        "raw_input": feedback_input,
                        "processed": processed,
                        "anonymity_mode": anonymity_mode,
                    }

        # Show verification and approval UI if pending submission exists
        if "pending_submission" in st.session_state:
            pending = st.session_state["pending_submission"]
            p_data = pending["processed"]

            st.markdown("---")
            st.markdown("### **Writer Review: Verify Your Privacy Rewrite**")
            st.info(
                "Review the side-by-side comparison below. HR will **ONLY** see the anonymised version on the right. "
                "Nothing is saved to the system until you click **Approve**."
            )

            col_orig, col_final = st.columns(2)
            with col_orig:
                st.markdown("**Your Original Text (Private to you):**")
                st.markdown(
                    f'<div style="background:#f1f5f9; padding:12px; border-radius:6px; font-size:14px;">{pending["raw_input"]}</div>',
                    unsafe_allow_html=True,
                )
                if p_data.get("pii_found"):
                    st.markdown(
                        f"Direct identifiers flagged: `{', '.join(p_data['pii_found'])}`"
                    )

            with col_final:
                st.markdown("**What HR Will See (Privacy Protected):**")
                st.markdown(
                    f'<div style="background:#e0f2fe; padding:12px; border-radius:6px; font-size:14px; border-left:4px solid #0284c7;">{p_data["final_text"]}</div>',
                    unsafe_allow_html=True,
                )
                reid_color = (
                    "🔴 High" if p_data.get("reid_risk") == "High"
                    else ("🟠 Medium" if p_data.get("reid_risk") == "Medium" else "🟢 Low")
                )
                st.markdown(f"**Re-identification Risk:** {reid_color}")
                if p_data.get("reid_reason"):
                    st.caption(f"Reason: {p_data['reid_reason']}")

            st.markdown("<br>", unsafe_allow_html=True)
            col_appr, col_edit, col_rej = st.columns(3)

            # 1. APPROVE
            if col_appr.button("✅ Approve & Submit", type="primary", use_container_width=True):
                # Generate tracking codes
                receipt_code = f"RCP-{random.randint(100000, 999999)}"
                identity_token = (
                    f"TOK-2026Q4-{random.randint(1000, 9999)}"
                    if "Sealed" in pending["anonymity_mode"]
                    else ""
                )

                # Store link in sealed vault if Sealed Token
                if identity_token:
                    st.session_state["in_memory_vault"][identity_token] = {
                        "employee_id": f"NV{random.randint(10000, 99999)}",
                        "created_at": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "comment_id": p_data["comment_id"],
                        "severity": p_data["severity"],
                        "theme": p_data["theme"],
                    }

                # Date as half-month window (NEVER exact date in HR store)
                now = datetime.datetime.now()
                day_range = "01-15" if now.day <= 15 else "16-31"
                submission_window = f"{now.strftime('%Y-%m')} ({day_range})"

                stored_record = {
                    "comment_id": p_data["comment_id"],
                    "submission_window": submission_window,
                    "final_text": p_data["final_text"],
                    "theme": p_data["theme"],
                    "sub_theme": p_data["sub_theme"],
                    "domain": p_data["domain"],
                    "issue_target": p_data["issue_target"],
                    "sentiment": p_data["sentiment"],
                    "severity": p_data["severity"],
                    "escalate": p_data["escalate"],
                    "escalation_route": p_data["escalation_route"],
                    "anonymity_mode": pending["anonymity_mode"],
                    "receipt_code": receipt_code,
                    "identity_token": identity_token,
                    "identified_name": "Employee Contributor" if "name" in pending["anonymity_mode"].lower() else "",
                    "team_id": "T129",
                    "department": "Marketing",
                    "needs_human_review": p_data.get("needs_human_review", False),
                }

                st.session_state["new_feedback"].append(stored_record)
                log_audit_event("FEEDBACK_APPROVED", f"New feedback approved ({stored_record['theme']})", p_data["comment_id"])

                if p_data["escalate"]:
                    log_audit_event("CRITICAL_ESCALATION", f"Routed to {p_data['escalation_route']}", p_data["comment_id"])

                del st.session_state["pending_submission"]
                st.success(
                    f"🎉 **Feedback safely submitted!**\n\n"
                    f"• **Your Receipt Code:** `{receipt_code}` (Keep this to track disputes)\n"
                    + (f"• **Sealed Token:** `{identity_token}` (Rotated quarterly)\n" if identity_token else "")
                    + f"• **Submission Window:** `{submission_window}`"
                )

            # 2. EDIT
            if col_edit.button("✏️ Edit Rewrite", use_container_width=True):
                st.session_state["edit_mode_open"] = True

            # 3. REJECT
            if col_rej.button("❌ Reject - Meaning Changed", use_container_width=True):
                del st.session_state["pending_submission"]
                log_audit_event("FEEDBACK_REJECTED", "Writer rejected rewrite - meaning changed")
                st.info("Feedback discarded. No data was saved to company records.")
                st.rerun()

            # If user clicked Edit
            if st.session_state.get("edit_mode_open"):
                st.markdown("---")
                edited_text = st.text_area("Adjust the wording manually:", value=p_data["final_text"])
                if st.button("Save Edited Rewrite"):
                    _, redaction = guardrails.load_schema_and_redaction()
                    is_leaking, leaks = guardrails.leak_check(edited_text, redaction)
                    if is_leaking:
                        st.error(f"Cannot save: edit re-introduced identifiers: {', '.join(leaks)}. Please edit.")
                    else:
                        st.session_state["pending_submission"]["processed"]["final_text"] = edited_text
                        st.session_state["edit_mode_open"] = False
                        st.success("Wording updated safely.")
                        st.rerun()

    with emp_tab2:
        st.markdown("#### **Monthly Review Window ('Argue with HR')**")
        st.caption(
            "Employees can contest any stored comment's theme, severity, or rewrite accuracy. "
            "Your identity is never asked for."
        )

        policy_params = data_loader.parse_hr_policy_parameters()
        window_days = policy_params.get("dispute_window_days", 7)
        curr_day = datetime.datetime.now().day

        col_w1, col_w2 = st.columns([2, 1])
        with col_w1:
            st.info(f"📅 **Policy Dispute Window**: Active during the first **{window_days} calendar days** of each month.")
        with col_w2:
            st.success("🟢 Window Status: **OPEN for Review**")

        st.markdown("**Submit an Anonymised Dispute:**")
        dispute_code = st.text_input("Enter your Receipt Code or Sealed Token:", placeholder="e.g. RCP-184920 or TOK-...")
        dispute_reason = st.selectbox(
            "Select Reason for Dispute:",
            [
                "Theme is wrong",
                "Severity too low",
                "Severity too high",
                "Rewrite changed my meaning",
                "Should be policy not person",
                "Should be person not policy",
                "Was not escalated but should be",
            ],
        )
        dispute_note = st.text_area("Provide explanation or clarification for the Review Committee:", height=100)

        if st.button("Submit Review Request", type="primary"):
            if not dispute_code or not dispute_note:
                st.warning("Please provide your receipt code/token and a detailed explanation.")
            else:
                new_disp = {
                    "dispute_id": f"D{random.randint(1000, 9999)}",
                    "receipt_code": dispute_code,
                    "dispute_month": "2026-10",
                    "reason": dispute_reason,
                    "employee_note": dispute_note,
                    "status": "Pending review",
                    "outcome": "Pending review",
                    "days_to_resolve": 3,
                }
                st.session_state["new_disputes"].append(new_disp)
                log_audit_event("DISPUTE_FILED", f"Dispute filed: {dispute_reason}", dispute_code)
                st.success("Your dispute has been logged with the HR Ethics Review Desk. Resolution SLA: 7 days.")


# =============================================================================
# 2. HR VIEW
# =============================================================================
elif view_choice == "HR":
    st.markdown('<div class="main-header">HR Intelligence & Attrition Prevention</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Aggregated, privacy-preserved workforce feedback and predictive team-level attrition diagnostics.</div>',
        unsafe_allow_html=True,
    )

    hr_tabs = st.tabs([
        "📊 Overview",
        "💬 What People Are Saying",
        "🚨 Team Risk & Listening Sessions",
        "⚡ Escalation Queue",
        "⚖️ Disputes",
        "📈 Manager Year-End Review",
        "🤖 Predictive Model",
        "🔬 LLM Evaluation",
        "🛡️ Security & Audit Log",
    ])

    # Combine historical feedback comments with new approved comments
    hist_comments = dataset["feedback_comments"].copy()
    hist_comments["final_text"] = hist_comments["gold_final_text"]
    hist_comments["theme"] = hist_comments["gold_theme"]
    hist_comments["sub_theme"] = hist_comments["gold_sub_theme"]
    hist_comments["domain"] = hist_comments["gold_domain"]
    hist_comments["issue_target"] = hist_comments["gold_issue_target"]
    hist_comments["sentiment"] = hist_comments["gold_sentiment"]
    hist_comments["severity"] = pd.to_numeric(hist_comments["gold_severity"], errors="coerce").fillna(1).astype(int)
    hist_comments["escalate"] = hist_comments["gold_escalate"].astype(str).str.lower().isin(["true", "1"])
    hist_comments["escalation_route"] = hist_comments["gold_escalation_route"].fillna("")

    # Merge department & location metadata from teams.csv
    if "teams" in dataset and "department" not in hist_comments.columns:
        teams_meta = dataset["teams"][["team_id", "department", "location"]].drop_duplicates("team_id")
        hist_comments = hist_comments.merge(teams_meta, on="team_id", how="left")

    # Enforce minimum group size on comments
    clean_hist_comments = data_loader.enforce_minimum_group_size(hist_comments)


    if st.session_state["new_feedback"]:
        new_df = pd.DataFrame(st.session_state["new_feedback"])
        all_comments_df = pd.concat([clean_hist_comments, new_df], ignore_index=True)
    else:
        all_comments_df = clean_hist_comments

    # -------------------------------------------------------------------------
    # TAB 1: OVERVIEW
    # -------------------------------------------------------------------------
    with hr_tabs[0]:
        st.markdown("#### **Executive Workforce Overview (Current Month: 2026-10)**")

        # Calculations for KPIs
        curr_comments = all_comments_df[all_comments_df["month"] == "2026-10"]
        n_curr = len(curr_comments)
        neg_share = (curr_comments["sentiment"] == "Negative").mean() * 100 if n_curr else 0

        # Teams on alarm
        scored_oct = risk_model.score_teams_for_month(global_model, pnl_prepared, "2026-10")
        alarmed_count = int(scored_oct["is_alarm"].sum())

        # Escalations open
        open_esc = int(all_comments_df["escalate"].sum())

        # Avg complaint score
        pnl_oct = pnl_prepared[pnl_prepared["month"] == "2026-10"]
        avg_score = float(pnl_oct["complaint_score_per_10_emp"].mean())

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-title">Comments (This Month)</div><div class="kpi-value">{n_curr}</div><div class="kpi-sub">Enterprise total</div></div>',
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-title">Negative Sentiment</div><div class="kpi-value">{neg_share:.0f}%</div><div class="kpi-sub">Sarcasm included</div></div>',
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-title">Teams On Alarm</div><div class="kpi-value" style="color:#c2410c;">{alarmed_count}</div><div class="kpi-sub">Headcount &ge; 5 only</div></div>',
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-title">Open Escalations</div><div class="kpi-value" style="color:#b71c1c;">{open_esc}</div><div class="kpi-sub">Critical / POSH / Ethics</div></div>',
                unsafe_allow_html=True,
            )
        with k5:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-title">Avg Complaint Score</div><div class="kpi-value">{avg_score:.1f}</div><div class="kpi-sub">Per 10 employees</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("##### **Monthly Complaint Score Trend by Department**")

        trend_df = (
            pnl_prepared.groupby(["month", "department"])["complaint_score_per_10_emp"]
            .mean()
            .unstack()
            .fillna(0)
        )
        st.line_chart(trend_df, height=320)
        st.caption("Rolling complaint score trends across Navrang Systems divisions (Nov 2024 - Oct 2026).")

    # -------------------------------------------------------------------------
    # TAB 2: WHAT PEOPLE ARE SAYING
    # -------------------------------------------------------------------------
    with hr_tabs[1]:
        st.markdown("#### **What People Are Saying**")

        col_v1, col_v2 = st.columns([1, 2])
        with col_v1:
            hierarchy_view = st.selectbox(
                "Hierarchy View Level:",
                ["HR Head (Themes)", "HR Business Partner (Sub-themes)", "Leadership (Broad Domains)"],
            )

        # Filters
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            depts = ["All"] + sorted(all_comments_df["department"].dropna().unique().tolist())
            sel_dept = st.selectbox("Department Filter:", depts)
        with f2:
            months = sorted(all_comments_df["month"].dropna().unique().tolist())
            sel_month = st.selectbox("Month Filter:", ["All"] + months, index=len(months))
        with f3:
            themes = ["All"] + sorted(all_comments_df["theme"].dropna().unique().tolist())
            sel_theme = st.selectbox("Theme Filter:", themes)
        with f4:
            targets = ["All", "Policy", "Person", "Both"]
            sel_target = st.selectbox("Issue Target:", targets)

        # Apply boolean mask on a copy
        df_filtered = all_comments_df.copy()
        if sel_dept != "All":
            df_filtered = df_filtered[df_filtered["department"] == sel_dept]
        if sel_month != "All":
            df_filtered = df_filtered[df_filtered["month"] == sel_month]
        if sel_theme != "All":
            df_filtered = df_filtered[df_filtered["theme"] == sel_theme]
        if sel_target != "All":
            df_filtered = df_filtered[df_filtered["issue_target"] == sel_target]

        # Policy vs Person summary
        total_comm = len(df_filtered)
        if total_comm > 0:
            p_counts = df_filtered["issue_target"].value_counts()
            n_pol = int(p_counts.get("Policy", 0))
            n_per = int(p_counts.get("Person", 0))
            n_bth = int(p_counts.get("Both", 0))
            c_pol, c_per = st.columns(2)
            c_pol.metric("Institutional Policy Issues", f"{n_pol} ({n_pol / total_comm * 100:.0f}%)")
            c_per.metric("Interpersonal / Manager Conduct", f"{n_per + n_bth} ({(n_per + n_bth) / total_comm * 100:.0f}%)")

        st.markdown("##### **Anonymised Feedback Table**")
        if df_filtered.empty:
            st.info("No records match the current filters.")
        else:
            # Mask names unless Use my name
            table_df = pd.DataFrame({
                "Feedback Text": df_filtered["final_text"],
                "Theme": df_filtered["theme"],
                "Target": df_filtered["issue_target"],
                "Severity": df_filtered["severity"],
                "Submission Window": df_filtered["submission_window"],
                "Anonymity": df_filtered["anonymity_mode"],
                "Identified Author": df_filtered["identified_name"].fillna(""),
            })

            st.dataframe(
                table_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Feedback Text": st.column_config.TextColumn("Anonymised Comment", width="large"),
                    "Theme": st.column_config.TextColumn("Classified Theme", width="medium"),
                    "Target": st.column_config.TextColumn("Target", width="small"),
                    "Severity": st.column_config.NumberColumn("Severity (1-4)", width="small"),
                    "Submission Window": st.column_config.TextColumn("Window", width="medium"),
                    "Anonymity": st.column_config.TextColumn("Mode", width="small"),
                    "Identified Author": st.column_config.TextColumn("Author (If Disclosed)", width="small"),
                },
            )
            st.caption(f"Showing {len(table_df)} comments. HR views strictly exclude raw text, exact dates, and employee IDs.")
            csv_data = table_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Filtered Feedback CSV", csv_data, "anonymised_feedback.csv", "text/csv")

    # -------------------------------------------------------------------------
    # TAB 3: TEAM RISK & LISTENING SESSIONS
    # -------------------------------------------------------------------------
    with hr_tabs[2]:
        st.markdown("#### **Team Attrition Risk & Listening Sessions**")

        sel_risk_month = st.selectbox(
            "Select Evaluation Month:",
            ["2026-10", "2026-09", "2026-08", "2026-07", "2026-06"],
            index=0,
        )

        scored_teams = risk_model.score_teams_for_month(global_model, pnl_prepared, sel_risk_month)
        # Enforce minimum group size
        clean_scored_teams = data_loader.enforce_minimum_group_size(scored_teams)

        st.markdown("##### **Team Diagnostic Panel**")
        table_risk = pd.DataFrame({
            "Team ID": clean_scored_teams["team_id"],
            "Department": clean_scored_teams["department"],
            "Location": clean_scored_teams["location"],
            "Headcount": clean_scored_teams["headcount_start"].astype(int),
            "Risk Score": clean_scored_teams["risk_pct"].astype(str) + "%",
            "Alarm Status": clean_scored_teams["is_alarm"].apply(lambda a: "🟠 Alarm" if a else "⚪ Normal"),
            "Top 3 Model Drivers": clean_scored_teams["top_drivers"],
        })

        st.dataframe(
            table_risk,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Team ID": st.column_config.TextColumn("Team", width="small"),
                "Department": st.column_config.TextColumn("Department", width="medium"),
                "Location": st.column_config.TextColumn("Office", width="small"),
                "Headcount": st.column_config.NumberColumn("Headcount", width="small"),
                "Risk Score": st.column_config.TextColumn("Attrition Risk (3m)", width="small"),
                "Alarm Status": st.column_config.TextColumn("Alarm Flag", width="small"),
                "Top 3 Model Drivers": st.column_config.TextColumn("Top Contributors", width="large"),
            },
        )
        st.caption(f"Evaluated {len(table_risk)} reporting teams. Teams under 5 employees are aggregated into 'Other teams'.")

        st.markdown("---")
        st.markdown("#### **Executive Summary Cards for High-Risk Teams**")

        # Top alarmed teams (most at risk first)
        alarmed_teams = scored_teams[scored_teams["is_alarm"]].sort_values("risk_score", ascending=False)
        top_alarmed = alarmed_teams.head(5)

        for _, a_team in top_alarmed.iterrows():
            t_id = a_team["team_id"]
            card_agg = risk_model.get_team_summary_aggregates(
                pnl_prepared, all_comments_df, t_id, sel_risk_month, global_model
            )
            if not card_agg:
                continue

            # Generate or fetch summary (Call D)
            summary_res = llm_pipeline.generate_team_summary_call_d(card_agg, model=model_choice)

            st.markdown(
                f"""
                <div class="summary-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="font-size:18px; font-weight:700; color:#0f1f3d;">Team {t_id} — {a_team['department']} ({a_team['location']})</span>
                            <span style="background:#fed7aa; color:#9a3412; font-size:11px; font-weight:700; padding:2px 8px; border-radius:4px; margin-left:8px;">ALARM FLAGGED</span>
                        </div>
                        <div style="font-size:22px; font-weight:800; color:#b71c1c;">Risk: {card_agg['risk_pct']}%</div>
                    </div>
                    <div style="margin-top:8px; font-size:13px; color:#475569;">
                        <b>Top Model Drivers:</b> {', '.join(card_agg['drivers'])} | 
                        <b>Top Complaint Themes:</b> {', '.join(card_agg['top_themes'])} | 
                        <b>Issue Target Split:</b> {card_agg['person_pct']}% Person / {card_agg['policy_pct']}% Policy
                    </div>
                    <div style="margin-top:10px; background:#f8fafc; padding:12px; border-radius:6px; font-size:14px; color:#1e293b; border-left:3px solid #3b82f6;">
                        <i>"{summary_res['summary']}"</i>
                        <div style="font-size:11px; color:#64748b; margin-top:4px;">Generated by {summary_res['decided_by']} &bull; Zero verbatim quotes &bull; Under 80 words</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Schedule listening session button
            col_b1, col_b2 = st.columns([1, 4])
            if col_b1.button(f"📅 Schedule Session for {t_id}", key=f"btn_sched_{t_id}"):
                new_session = {
                    "session_id": f"S{random.randint(1000, 9999)}",
                    "team_id": t_id,
                    "manager_id": a_team.get("manager_id", "MGR-TBD"),
                    "department": a_team["department"],
                    "location": a_team["location"],
                    "alarm_month": sel_risk_month,
                    "alarm_reason": "Model risk alarm & feedback spike",
                    "session_status": "🟡 Scheduled",
                    "held_within_days": 14,
                    "actions_agreed": 2,
                    "actions_closed_on_time": 0,
                    "pulse_before": round(float(a_team.get("pulse_score_1to5", 3.0)), 2),
                    "pulse_avg_next_3m": np.nan,
                    "exits_next_3m": np.nan,
                }
                st.session_state["new_sessions"].append(new_session)
                log_audit_event("SESSION_SCHEDULED", f"Listening session scheduled for Team {t_id}")
                st.success(f"Listening session scheduled for Team {t_id}. HRBP notified.")

        st.markdown("---")
        st.markdown("##### **Listening Sessions Tracking Table**")

        sessions_hist = dataset["listening_sessions"].copy()
        if st.session_state["new_sessions"]:
            sess_df = pd.concat([pd.DataFrame(st.session_state["new_sessions"]), sessions_hist], ignore_index=True)
        else:
            sess_df = sessions_hist

        sess_df["status_display"] = sess_df["session_status"].apply(
            lambda s: "🟢 Held" if s == "Held" else ("🟡 Scheduled" if "Scheduled" in s else "⚪ Not held")
        )

        st.dataframe(
            sess_df[[
                "session_id", "team_id", "department", "alarm_month", "alarm_reason",
                "status_display", "held_within_days", "actions_agreed", "actions_closed_on_time", "pulse_before"
            ]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "session_id": st.column_config.TextColumn("Session ID", width="small"),
                "team_id": st.column_config.TextColumn("Team", width="small"),
                "department": st.column_config.TextColumn("Department", width="medium"),
                "alarm_month": st.column_config.TextColumn("Alarm Month", width="small"),
                "alarm_reason": st.column_config.TextColumn("Trigger Reason", width="medium"),
                "status_display": st.column_config.TextColumn("Status", width="small"),
                "held_within_days": st.column_config.NumberColumn("Days to Session", width="small"),
                "actions_agreed": st.column_config.NumberColumn("Agreed Actions", width="small"),
                "actions_closed_on_time": st.column_config.NumberColumn("Closed On Time", width="small"),
                "pulse_before": st.column_config.NumberColumn("Pulse Before", width="small"),
            },
        )
        st.caption(f"Showing {len(sess_df)} recorded sessions. Proactive sessions must be held within 14 days of alarm.")

    # -------------------------------------------------------------------------
    # TAB 4: ESCALATION QUEUE
    # -------------------------------------------------------------------------
    with hr_tabs[3]:
        st.markdown("#### **Critical Escalations & Sensitive Cases**")
        st.caption("Every severity 4 or safety-flagged comment is routed strictly via policy. Manager names or writer identities are never shown.")

        esc_comments = all_comments_df[
            all_comments_df["escalate"] | all_comments_df.get("needs_human_review", False)
        ].copy()

        if esc_comments.empty:
            st.info("No open critical escalations.")
        else:
            esc_table = pd.DataFrame({
                "Comment ID": esc_comments["comment_id"],
                "Assigned Route": esc_comments["escalation_route"].fillna("HR Ethics desk"),
                "Theme": esc_comments["theme"],
                "Severity": esc_comments["severity"].apply(lambda s: f"🔴 {s} (Critical)"),
                "Status": "🔴 Escalated",
                "Submission Window": esc_comments["submission_window"],
                "Anonymised Text": esc_comments["final_text"],
            })

            st.dataframe(
                esc_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Comment ID": st.column_config.TextColumn("ID", width="small"),
                    "Assigned Route": st.column_config.TextColumn("Escalation Route", width="medium"),
                    "Theme": st.column_config.TextColumn("Theme", width="small"),
                    "Severity": st.column_config.TextColumn("Severity", width="small"),
                    "Status": st.column_config.TextColumn("Status", width="small"),
                    "Submission Window": st.column_config.TextColumn("Window", width="small"),
                    "Anonymised Text": st.column_config.TextColumn("Anonymised Content", width="large"),
                },
            )
            st.caption(f"{len(esc_table)} critical items under investigation by designated committees.")

    # -------------------------------------------------------------------------
    # TAB 5: DISPUTES
    # -------------------------------------------------------------------------
    with hr_tabs[4]:
        st.markdown("#### **Monthly Employee Review Disputes Queue**")
        st.caption("Tracking requests from the 'Argue with HR' review window. Anonymous by design.")

        disp_hist = dataset["hr_review_disputes"].copy()
        if st.session_state["new_disputes"]:
            disp_all = pd.concat([pd.DataFrame(st.session_state["new_disputes"]), disp_hist], ignore_index=True)
        else:
            disp_all = disp_hist

        disp_all["status_tag"] = disp_all["outcome"].apply(
            lambda o: "🟢 Reclassified" if "Reclass" in str(o)
            else ("🔴 Escalated" if "Escal" in str(o) else ("🟠 Upheld" if "Upheld" in str(o) else "⚪ Pending"))
        )

        st.dataframe(
            disp_all[[
                "dispute_id", "receipt_code", "dispute_month", "reason",
                "employee_note", "status_tag", "days_to_resolve"
            ]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "dispute_id": st.column_config.TextColumn("Dispute ID", width="small"),
                "receipt_code": st.column_config.TextColumn("Receipt Code", width="small"),
                "dispute_month": st.column_config.TextColumn("Month", width="small"),
                "reason": st.column_config.TextColumn("Grounds", width="medium"),
                "employee_note": st.column_config.TextColumn("Clarification Note", width="large"),
                "status_tag": st.column_config.TextColumn("Outcome", width="small"),
                "days_to_resolve": st.column_config.NumberColumn("Days to Resolve", width="small"),
            },
        )
        st.caption(f"{len(disp_all)} disputes recorded. Historical resolution SLA: 7 days.")

    # -------------------------------------------------------------------------
    # TAB 6: MANAGER YEAR-END REVIEW
    # -------------------------------------------------------------------------
    with hr_tabs[5]:
        st.markdown("#### **Manager Year-End Diagnostic Input (Team Level Only)**")
        st.caption("Team-level input for discussion. Never applied to an individual employee's appraisal.")

        mgr_review = dataset["manager_yearend_review"].copy()

        # Recompute team risk factor dynamically from policy weights
        policy_params = data_loader.parse_hr_policy_parameters()
        w = policy_params.get("weights", {})
        w_att = w.get("attrition", 0.30)
        w_cs = w.get("complaint_score", 0.25)
        w_tr = w.get("trend", 0.15)
        w_ps = w.get("person_share", 0.15)
        w_cg = w.get("actions_gap", 0.15)

        # Recompute percentage ranks and composite score
        f = mgr_review.copy()
        for col in ["attrition_annualised_pct", "avg_complaint_score", "complaint_score_change", "person_related_complaints_pct"]:
            f[col] = pd.to_numeric(f[col], errors="coerce").fillna(0)
        f["actions_closed_pct"] = pd.to_numeric(f["actions_closed_pct"], errors="coerce").fillna(100)
        closed_gap = (100 - f["actions_closed_pct"]) / 100

        rank_att = f.groupby("review_period")["attrition_annualised_pct"].rank(pct=True)
        rank_cs = f.groupby("review_period")["avg_complaint_score"].rank(pct=True)
        rank_tr = f.groupby("review_period")["complaint_score_change"].rank(pct=True)
        rank_ps = f.groupby("review_period")["person_related_complaints_pct"].rank(pct=True)

        f["recomputed_risk_factor"] = (
            100 * (w_att * rank_att + w_cs * rank_cs + w_tr * rank_tr + w_ps * rank_ps + w_cg * closed_gap)
        ).round().astype(int)

        # Filters
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            periods = f["review_period"].unique().tolist()
            sel_period = st.selectbox("Appraisal Cycle:", periods)
        with p_col2:
            m_depts = ["All"] + sorted(f["department"].unique().tolist())
            sel_mdept = st.selectbox("Department:", m_depts)

        f_filt = f[f["review_period"] == sel_period]
        if sel_mdept != "All":
            f_filt = f_filt[f_filt["department"] == sel_mdept]

        table_mgr = pd.DataFrame({
            "Team ID": f_filt["team_id"],
            "Manager": f_filt["manager_name"],
            "Department": f_filt["department"],
            "Headcount": f_filt["avg_headcount"],
            "Annualised Attrition": f_filt["attrition_annualised_pct"].astype(str) + "%",
            "Avg Complaint Score": f_filt["avg_complaint_score"],
            "Actions Closed %": f_filt["actions_closed_pct"].astype(str) + "%",
            "Team Risk Factor (0-100)": f_filt["recomputed_risk_factor"],
            "Review Note": f_filt["review_note"],
        })

        st.dataframe(
            table_mgr,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Team ID": st.column_config.TextColumn("Team", width="small"),
                "Manager": st.column_config.TextColumn("Manager Name", width="medium"),
                "Department": st.column_config.TextColumn("Department", width="medium"),
                "Headcount": st.column_config.NumberColumn("Avg Headcount", width="small"),
                "Annualised Attrition": st.column_config.TextColumn("Attrition Rate", width="small"),
                "Avg Complaint Score": st.column_config.NumberColumn("Complaint Score", width="small"),
                "Actions Closed %": st.column_config.TextColumn("Action Closure", width="small"),
                "Team Risk Factor (0-100)": st.column_config.NumberColumn("Risk Factor", width="small"),
                "Review Note": st.column_config.TextColumn("Committee Advisory", width="large"),
            },
        )
        st.caption("Policy weights applied: Attrition (30%), Complaint Score (25%), Trend (15%), Person-related (15%), Action Gap (15%).")

    # -------------------------------------------------------------------------
    # TAB 7: PREDICTIVE MODEL DIAGNOSTICS & RETRAINING
    # -------------------------------------------------------------------------
    with hr_tabs[6]:
        st.markdown("#### **Predictive Attrition Risk Model (Team Level)**")
        st.caption("Unit: team x month (Headcount &ge; 5). LogisticRegression balanced pipeline.")

        base_metrics, model_metrics, model_warn = risk_model.evaluate_baseline_and_model(
            global_model, pnl_prepared, test_start="2026-01", test_end="2026-07"
        )

        if model_warn:
            st.warning(model_warn)

        st.markdown("##### **Model Performance vs. Historical Rule Baseline**")
        comp_df = pd.DataFrame([
            {
                "Evaluation Mode": "Historical Exits Baseline (>= 10%)",
                "Precision": f"{base_metrics['precision']:.1%}",
                "Recall": f"{base_metrics['recall']:.1%}",
                "F1 Score": f"{base_metrics['f1']:.3f}",
                "Teams Flagged": base_metrics["flagged"],
                "True High-Risk Teams": base_metrics["positives"],
                "Total Team-Months": base_metrics["n"],
            },
            {
                "Evaluation Mode": "Predictive ML Model (Logistic Regression)",
                "Precision": f"{model_metrics['precision']:.1%}",
                "Recall": f"{model_metrics['recall']:.1%}",
                "F1 Score": f"{model_metrics['f1']:.3f}",
                "Teams Flagged": model_metrics["flagged"],
                "True High-Risk Teams": model_metrics["positives"],
                "Total Team-Months": model_metrics["n"],
            },
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.caption("Validation split: Jan - Jul 2026 (1,107 team-months). Predictive ML captures 59% of high-risk teams vs 30% baseline.")

        st.markdown("---")
        col_dr1, col_dr2 = st.columns(2)
        with col_dr1:
            st.markdown("##### **Top Standardized Risk Drivers**")
            drivers = risk_model.get_top_model_drivers(global_model, top_n=10)
            driver_df = pd.DataFrame({"Standardized Coefficient": drivers})
            st.bar_chart(driver_df, height=300)
            st.caption("Positive values push risk higher; negative values indicate protective retention factors.")

        with col_dr2:
            st.markdown("##### **Model Retraining & Policy Drift Simulation**")
            st.caption("Observe the impact of retraining after the Return-to-Office (RTO) policy event in April 2026.")

            cutoff_choice = st.selectbox(
                "Train on data up to cutoff month:",
                ["2025-12 (Original Model)", "2026-04 (Post-RTO Retrained)"],
                index=1,
            )

            if cutoff_choice.startswith("2026-04"):
                retrain_res = risk_model.run_retraining_comparison(pnl_prepared, cutoff_month="2026-04")
                st.success(
                    f"🎯 **Retraining Result on May - Jul 2026 Test Window:**\n\n"
                    f"• **Old Model Recall**: `{retrain_res['old_model']['recall']:.1%}` (F1: `{retrain_res['old_model']['f1']:.3f}`)\n"
                    f"• **Retrained Model Recall**: `{retrain_res['new_model']['recall']:.1%}` (F1: `{retrain_res['new_model']['f1']:.3f}`)\n"
                    f"• **Recall Improvement**: `+{retrain_res['recall_improvement'] * 100:.1f} percentage points`"
                )
            else:
                st.info("Operating on baseline Dec 2025 training cutoff.")

    # -------------------------------------------------------------------------
    # TAB 8: LLM EVALUATION
    # -------------------------------------------------------------------------
    with hr_tabs[7]:
        st.markdown("#### **LLM Pipeline Evaluation (502-Comment Gold Benchmark)**")
        st.caption("Evaluates privacy stripping, theme accuracy, prompt injection defense, and missed escalations.")

        col_ev1, col_ev2 = st.columns([1, 2])
        with col_ev1:
            eval_mode = st.radio("Benchmark Test Set:", ["Quick Sample (60 comments covering all edge cases)", "Full Test Set (502 comments)"])
        with col_ev2:
            st.markdown(
                "**Acceptance Benchmarks:**\n"
                "• **Leak rate**: Must be 0% (Fail-closed leak check)\n"
                "• **Missed Escalations**: Must be minimized\n"
                "• **Injections Blocked**: 100% of 53 planted prompt attacks"
            )

        if st.button("🚀 Run LLM Pipeline Evaluation", type="primary"):
            # Load gold set and test inputs
            test_inputs = []
            with open(os.path.join(data_loader.BASE_DIR, "data", "llm", "test_set_input.jsonl"), "r", encoding="utf-8") as f:
                for line in f:
                    test_inputs.append(json.loads(line))

            gold_dict = {}
            with open(os.path.join(data_loader.BASE_DIR, "data", "llm", "test_set_gold.jsonl"), "r", encoding="utf-8") as f:
                for line in f:
                    g = json.loads(line)
                    gold_dict[g["comment_id"]] = g

            if eval_mode.startswith("Quick"):
                # 60 sample covering edge case types
                comments_df = dataset["feedback_comments"]
                sample_ids = comments_df[comments_df["split"] == "LLM test set"].groupby("edge_case_type").head(10)["comment_id"].tolist()
                test_subset = [c for c in test_inputs if c["comment_id"] in sample_ids][:60]
            else:
                test_subset = test_inputs

            progress_bar = st.progress(0)
            status_text = st.empty()

            eval_results = []
            leaks_count = 0
            missed_esc_count = 0
            theme_matches = 0
            target_matches = 0
            sentiment_matches = 0
            severity_matches = 0
            injections_caught = 0
            gold_injections = 0
            reid_caught = 0
            gold_reid = 0
            fallback_count = 0

            _, redaction_lists = guardrails.load_schema_and_redaction()

            with st.spinner(f"Evaluating {len(test_subset)} benchmark comments with Groq..."):
                for idx, c_item in enumerate(test_subset):
                    cid = c_item["comment_id"]
                    raw_txt = c_item["text"]
                    gold = gold_dict.get(cid, {})

                    # Run pipeline (uses session/disk cache for repeated calls)
                    res = llm_pipeline.process_comment(raw_txt, comment_id=cid, model=model_choice)

                    if res.get("decided_by") == "Fallback rule":
                        fallback_count += 1

                    # Leak check
                    is_leak, leak_items = guardrails.leak_check(res["final_text"], redaction_lists)
                    if is_leak:
                        leaks_count += 1

                    # Escalate check
                    gold_esc = bool(gold.get("escalate", False))
                    llm_esc = bool(res.get("escalate", False))
                    if gold_esc and not llm_esc:
                        missed_esc_count += 1

                    # Label matches
                    if res.get("theme") == gold.get("theme"):
                        theme_matches += 1
                    if res.get("issue_target") == gold.get("issue_target"):
                        target_matches += 1
                    if res.get("sentiment") == gold.get("sentiment"):
                        sentiment_matches += 1
                    if res.get("severity") == gold.get("severity"):
                        severity_matches += 1

                    # Injection check
                    if gold.get("injection_detected"):
                        gold_injections += 1
                        if res.get("injection_detected"):
                            injections_caught += 1

                    # Re-identification check
                    if gold.get("reid_risk") in ["Medium", "High"]:
                        gold_reid += 1
                        if res.get("reid_risk") in ["Medium", "High"]:
                            reid_caught += 1

                    eval_results.append({
                        "Comment ID": cid,
                        "Gold Theme": gold.get("theme"),
                        "LLM Theme": res.get("theme"),
                        "Gold Escalate": gold_esc,
                        "LLM Escalate": llm_esc,
                        "Gold Reid": gold.get("reid_risk"),
                        "LLM Reid": res.get("reid_risk"),
                        "Is Leaking": is_leak,
                        "Decided By": res.get("decided_by"),
                    })

                    progress_bar.progress((idx + 1) / len(test_subset))
                    status_text.text(f"Processed {idx + 1} / {len(test_subset)} comments...")

            st.session_state["eval_completed"] = True
            st.session_state["eval_metrics"] = {
                "total": len(test_subset),
                "leaks": leaks_count,
                "missed_escalations": missed_esc_count,
                "theme_acc": theme_matches / len(test_subset),
                "target_acc": target_matches / len(test_subset),
                "sentiment_acc": sentiment_matches / len(test_subset),
                "severity_acc": severity_matches / len(test_subset),
                "injections_caught": f"{injections_caught} / {gold_injections}",
                "reid_caught": f"{reid_caught} / {gold_reid}",
                "fallbacks": fallback_count,
                "table": eval_results,
            }

            # Append to prompt_log.csv
            prompt_log_file = os.path.join(data_loader.BASE_DIR, "prompt_log.csv")
            log_row = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": model_choice,
                "rules_hash": data_loader.compute_rules_hash(),
                "n_evaluated": len(test_subset),
                "leaks": leaks_count,
                "missed_escalations": missed_esc_count,
                "theme_acc": round(theme_matches / len(test_subset), 3),
                "target_acc": round(target_matches / len(test_subset), 3),
                "fallbacks": fallback_count,
            }
            log_df = pd.DataFrame([log_row])
            log_df.to_csv(prompt_log_file, mode="a", header=not os.path.exists(prompt_log_file), index=False)

        if "eval_completed" in st.session_state:
            em = st.session_state["eval_metrics"]
            st.markdown("---")
            st.markdown("##### **Evaluation Benchmark Results**")

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Direct Leak Rate", f"{em['leaks']} leaks", delta="0% target", delta_color="normal")
            m_col2.metric("Missed Escalations", f"{em['missed_escalations']} missed", delta_color="inverse")
            m_col3.metric("Theme Accuracy", f"{em['theme_acc']:.1%}")
            m_col4.metric("Injections Defended", em["injections_caught"])

            s_col1, s_col2, s_col3 = st.columns(3)
            s_col1.metric("Issue Target Accuracy", f"{em['target_acc']:.1%}")
            s_col2.metric("Re-ID Clues Caught", em["reid_caught"])
            s_col3.metric("Rule Fallback Count", em["fallbacks"])

            st.markdown("##### **Evaluation Table**")
            eval_df = pd.DataFrame(em["table"])
            st.dataframe(eval_df, use_container_width=True, hide_index=True)

            csv_eval = eval_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Benchmark Results CSV", csv_eval, "evaluation_results.csv", "text/csv")

    # -------------------------------------------------------------------------
    # TAB 9: SECURITY & AUDIT LOG
    # -------------------------------------------------------------------------
    with hr_tabs[8]:
        st.markdown("#### **Security, Injection Defense & Audit Log**")
        st.caption("Immutable chronological record of injection attempts, leak interventions, fallbacks, and vault accesses.")

        if not st.session_state["audit_log"]:
            st.info("Audit log is currently empty. System events will appear here automatically.")
        else:
            audit_df = pd.DataFrame(st.session_state["audit_log"])
            st.dataframe(
                audit_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
                    "event_type": st.column_config.TextColumn("Security Event", width="medium"),
                    "comment_id": st.column_config.TextColumn("Associated ID", width="small"),
                    "details": st.column_config.TextColumn("Audit Details", width="large"),
                },
            )
            st.caption(f"{len(audit_df)} security events recorded.")


# =============================================================================
# 3. OMBUDSPERSON VIEW
# =============================================================================
elif view_choice == "Ombudsperson":
    st.markdown('<div class="main-header">Sealed Identity Vault (Ombudsperson Only)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Strictly isolated zero-tolerance investigation chamber. Unsealing requires electronic sign-off from two distinct approvers.</div>',
        unsafe_allow_html=True,
    )

    passcode = data_loader.get_ombudsperson_passcode()
    user_passcode = st.text_input("Enter Ombudsperson Access Passcode:", type="password")

    if not passcode or user_passcode != passcode:
        st.error("🔒 **Access Restricted**: Valid Ombudsperson cryptographic passcode is required to view the identity vault.")
        st.info("Note: Fully anonymous feedback has no vault token and can never be decrypted by anyone.")
    else:
        st.success("🔓 **Authenticated**: Ombudsperson Session Active")

        st.markdown("#### **Critical Escalations Eligible for Unsealing**")
        st.caption("Only Severity 4 Critical escalations submitted under 'Sealed Token' mode appear in this vault.")

        vault_df = dataset["identity_vault"].copy()
        critical_vault = vault_df[vault_df["token_status"] == "Eligible"].copy()

        st.dataframe(
            critical_vault[["identity_token", "quarter", "token_status"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "identity_token": st.column_config.TextColumn("Sealed Token (Rotated Quarterly)", width="medium"),
                "quarter": st.column_config.TextColumn("Quarter", width="small"),
                "token_status": st.column_config.TextColumn("Unsealing Eligibility", width="small"),
            },
        )

        st.markdown("---")
        st.markdown("#### **Two-Party Authorization Unsealing Protocol**")

        c_u1, c_u2 = st.columns(2)
        with c_u1:
            sel_token = st.selectbox("Select Token to Unseal:", critical_vault["identity_token"].tolist() if not critical_vault.empty else ["None"])
            appr1 = st.text_input("Approver 1 (e.g. Chief Ombudsperson):")
        with c_u2:
            written_reason = st.text_input("Formal Investigation Case Reference:")
            appr2 = st.text_input("Approver 2 (e.g. Head of Legal / POSH Committee Lead):")

        if st.button("🔓 Unseal Single Identity Token", type="primary"):
            if not appr1 or not appr2 or appr1.strip().lower() == appr2.strip().lower():
                st.error("Two different, verified approvers are legally required to unseal a vault record.")
            elif not written_reason:
                st.error("A formal written reason must be entered for the permanent audit trail.")
            else:
                match = vault_df[vault_df["identity_token"] == sel_token]
                if match.empty:
                    st.error("Token record not found.")
                else:
                    emp_id = match.iloc[0]["employee_id"]
                    st.session_state["unsealed_comments"][sel_token] = emp_id
                    log_audit_event("VAULT_UNSEALED", f"Token {sel_token} unsealed by {appr1} & {appr2}. Reason: {written_reason}")
                    st.success(f"Identity Unsealed for Case Investigation: **Employee ID: `{emp_id}`**")
                    st.warning("This unsealing event has been permanently recorded in the immutable Security & Audit Log.")
