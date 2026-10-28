# Human Resources Policy & Privacy Governance

This document establishes the official HR policies, thresholds, privacy minimums, and governance protocols for Navrang Systems' anonymous feedback and attrition prevention systems.

---

## 1. Privacy Thresholds & Group Size
- **Minimum Group Size (`min_group_size`)**: `5` employees.
- To protect anonymity, no team-level metrics (comment volume, sentiment, complaint scores, themes, or attrition risk) may ever be displayed or attributed to a team with fewer than 5 members.
- Teams below this threshold must be aggregated into `"Other teams (<Department>)"`.

---

## 2. Team Attrition Alarm Rules
A team is flagged on **Alarm** in any month where at least one of the following triggers is met (provided team headcount >= `min_group_size`):
1. **Model Risk Alarm**: The predictive attrition model outputs a risk probability >= `0.50` (or the configured threshold).
2. **Historical Exits Rule**: The team's voluntary exits in the trailing 3 months account for >= `10%` of starting headcount (`exit_rate_last_3m >= 0.10`).
3. **Complaint Score Spike**: The team's 3-month rolling complaint score per 10 employees ranks in the **top 5%** of all historical team-months across the enterprise.

Teams flagged with an Alarm trigger a mandatory proactive listening session.

---

## 3. Listening Session Protocol
- **Target Response Window**: A listening session must be scheduled and held within **14 calendar days** of the alarm notification.
- **Participants**: Department HR Business Partner (HRBP) and team members. Direct managers are excluded to maintain open dialogue.
- **Output**: 1 to 4 concrete agreed action items tracked to closure in the HR portal.

---

## 4. Escalation Routing Table
All critical feedback items (`severity = 4`) and sensitive themes must be routed strictly according to this table. Under no circumstances may an escalation be routed to the writer's immediate manager or team lead:

| Theme / Condition | Designated Escalation Route | Resolution SLA |
| :--- | :--- | :--- |
| **Harassment** | `Internal Committee (POSH) via HR` | 7 business days |
| **Discrimination** | `HR Ethics desk` | 10 business days |
| **Ethics** | `HR Ethics desk / Compliance` | 10 business days |
| **Other Critical Severity (4)** | `HR Ethics desk` | 10 business days |

---

## 5. Year-End Manager Review Input (Team Level Only)
To incentivize healthy team leadership and proactive issue resolution, a composite Team Risk Factor (0–100) is calculated at the team level for the annual manager review cycle using these policy weights:
- **Annualised Attrition Rate**: `30%` (0.30)
- **Average Complaint Score**: `25%` (0.25)
- **Complaint Score Trend (Last 3m minus First 3m)**: `15%` (0.15)
- **Person-Related Complaint Share**: `15%` (0.15)
- **Actions Not Closed Gap**: `15%` (0.15)

*Policy Constraint*: This metric is strictly a team-level diagnostic for managerial discussion. It must never be applied directly to an individual employee's performance appraisal or compensation score. Teams below 5 members are excluded.

---

## 6. Monthly Dispute Window ("Argue with HR")
- **Review Period**: Employees may contest the classification, severity, or rewrite accuracy of their submitted feedback during the **first 7 calendar days** of each calendar month (Days 1–7).
- **Authentication**: Access requires entering the submission's unique `receipt_code` or `identity_token`. The system never requests employee identity or credentials.
- **Allowed Grounds**:
  - Theme is wrong
  - Severity too low
  - Severity too high
  - Rewrite changed my meaning
  - Should be policy not person
  - Should be person not policy
  - Was not escalated but should be

---

## 7. Sealed Identity Vault Governance (Ombudsperson Only)
- The Identity Vault maps sealed tokens to employee records and is strictly isolated from standard HR dashboards.
- **Unsealing Criteria**:
  1. The feedback must be classified as a **Critical escalation** (`severity = 4`).
  2. Requires verified electronic authorization from **two separate approvers** (Ombudsperson + Head of People / Legal).
  3. A formal written justification must be recorded in the permanent audit trail.
- **Immutable Restriction**: Fully anonymous submissions have no vault token and can **never** be unsealed under any circumstances.

---

## 8. Model Retraining Cadence
- The team attrition risk model must undergo formal retraining and evaluation on a **quarterly** schedule, or immediately following enterprise-wide policy disruptions (such as Return-to-Office mandates or organizational restructuring).

---

## 9. High-Risk Team Summaries (Call D Guidelines)
For alarmed teams, an executive summary is generated for HR listening session prep under these constraints:
- Length: **2 to 3 sentences, maximum 80 words**.
- Style: Plain, constructive, professional English.
- Grounding: Must use **only** the numerical metrics and aggregates provided (e.g., risk probability, top themes, policy vs. person split). No numbers may be invented or hallucinated.
- Anonymity: Under no circumstances may individual comments be quoted verbatim, nor may any person or role be named.
- Conclusion: Must conclude with the **single most pressing theme** to prioritize during the initial 14-day listening session.
