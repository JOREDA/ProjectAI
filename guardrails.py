"""
guardrails.py - Deterministic pre-pass, leak checking, escalation enforcement,
safety net keyword screening, and schema validation.
The LLM proposes, Python decides; the LLM can never bypass these rules.
"""
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import jsonschema

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SAFETY_NET_KEYWORDS = [
    "unsafe", "touching", "harass", "marked as passed", "inflated",
    "posh", "bribe", "sexual", "retaliation", "caste", "religion",
    "gender", "threatened", "kickback", "falsified"
]

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|previous\s+|prior\s+)?instructions",
    r"system\s*:",
    r"you\s+are\s+now",
    r"reveal\b",
    r"print\s+(the\s+)?original",
    r"</comment>",
    r"<comment",
    r"disregard\s+",
    r"bypass\s+guardrails",
    r"output\s+the\s+system\s+prompt",
    r"as\s+an\s+ai\s+language\s+model",
]

ESCALATION_ROUTES = {
    "Harassment": "Internal Committee (POSH) via HR",
    "Discrimination": "HR Ethics desk",
    "Ethics": "HR Ethics desk / Compliance",
}


def load_schema_and_redaction() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Loads output schema and redaction lists.
    """
    schema_path = os.path.join(BASE_DIR, "data", "llm", "llm_output_schema.json")
    redaction_path = os.path.join(BASE_DIR, "data", "llm", "redaction_lists.json")

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    with open(redaction_path, "r", encoding="utf-8") as f:
        redaction = json.load(f)

    return schema, redaction


def deterministic_step0_prepass(
    raw_text: str,
    redaction_lists: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Step 0 - Python pre-pass (deterministic):
    1. Strip control characters.
    2. Reject empty, test/spam (< 3 real words) or > 2,000 characters.
    3. Replace anything on redaction_lists.json with neutral words.
    4. Run injection pre-screen regex.
    """
    result = {
        "is_valid": True,
        "rejection_reason": None,
        "cleaned_text": raw_text,
        "step0_text": raw_text,
        "injection_detected": False,
        "redactions_made": [],
    }

    if not raw_text or not isinstance(raw_text, str):
        result["is_valid"] = False
        result["rejection_reason"] = "Comment is empty."
        return result

    # Strip control characters (preserve normal whitespace: spaces, tabs, newlines)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", raw_text).strip()
    result["cleaned_text"] = cleaned

    # Check length and word count
    words = [w for w in re.split(r"\s+", cleaned) if re.search(r"\w", w)]
    if len(words) < 3:
        result["is_valid"] = False
        result["rejection_reason"] = "Comment contains fewer than 3 words (spam/test)."
        return result

    if len(cleaned) > 2000:
        result["is_valid"] = False
        result["rejection_reason"] = "Comment exceeds 2,000 characters limit."
        return result

    # Injection pre-screen
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            result["injection_detected"] = True
            break

    # Redactions
    processed = cleaned

    # Employee ID pattern NV\d{5}
    id_pattern = redaction_lists.get("employee_id_pattern", r"NV\d{5}")
    if re.search(id_pattern, processed, re.IGNORECASE):
        processed = re.sub(id_pattern, "[ID removed]", processed, flags=re.IGNORECASE)
        result["redactions_made"].append("Employee ID")

    # Project codes
    for prj in redaction_lists.get("project_codes", []):
        if prj in processed:
            processed = processed.replace(prj, "our project")
            result["redactions_made"].append(f"Project code ({prj})")

    # Client names
    for client in redaction_lists.get("clients", []):
        pattern = r"\b" + re.escape(client) + r"\b"
        if re.search(pattern, processed, re.IGNORECASE):
            processed = re.sub(pattern, "the client", processed, flags=re.IGNORECASE)
            result["redactions_made"].append(f"Client ({client})")

    # Building / floors
    for bld in redaction_lists.get("buildings_floors", []):
        pattern = r"\b" + re.escape(bld) + r"\b"
        if re.search(pattern, processed, re.IGNORECASE):
            processed = re.sub(pattern, "the office", processed, flags=re.IGNORECASE)
            result["redactions_made"].append(f"Building/Floor ({bld})")

    # Office locations
    for office in redaction_lists.get("offices", []):
        pattern = r"\b" + re.escape(office) + r"\b"
        if re.search(pattern, processed, re.IGNORECASE):
            processed = re.sub(pattern, "our office", processed, flags=re.IGNORECASE)
            result["redactions_made"].append(f"Office ({office})")

    # Manager full names
    mgr_names = redaction_lists.get("manager_full_names", [])
    for mgr in mgr_names:
        pattern = r"\b" + re.escape(mgr) + r"\b"
        if re.search(pattern, processed, re.IGNORECASE):
            processed = re.sub(pattern, "my manager", processed, flags=re.IGNORECASE)
            result["redactions_made"].append(f"Manager name ({mgr})")

    # Employee full names
    emp_names = redaction_lists.get("employee_full_names", [])
    for emp in emp_names:
        pattern = r"\b" + re.escape(emp) + r"\b"
        if re.search(pattern, processed, re.IGNORECASE):
            processed = re.sub(pattern, "a colleague", processed, flags=re.IGNORECASE)
            result["redactions_made"].append(f"Employee name ({emp})")

    result["step0_text"] = processed
    return result


def leak_check(
    text: str,
    redaction_lists: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Fail-closed leak check:
    Scans final_text against redaction_lists.json and NV\\d{5} pattern.
    Returns (is_leaking, list_of_leaked_items).
    Any hit blocks the comment from reaching HR.
    """
    if not text:
        return False, []

    leaks = []

    # 1. Employee ID pattern
    id_pattern = redaction_lists.get("employee_id_pattern", r"NV\d{5}")
    if re.search(id_pattern, text, re.IGNORECASE):
        leaks.append("Employee ID (NV pattern)")

    # 2. Project codes
    for prj in redaction_lists.get("project_codes", []):
        if prj.lower() in text.lower():
            leaks.append(f"Project code: {prj}")

    # 3. Clients
    for client in redaction_lists.get("clients", []):
        if re.search(r"\b" + re.escape(client) + r"\b", text, re.IGNORECASE):
            leaks.append(f"Client: {client}")

    # 4. Buildings & floors
    for bld in redaction_lists.get("buildings_floors", []):
        if re.search(r"\b" + re.escape(bld) + r"\b", text, re.IGNORECASE):
            leaks.append(f"Building/Floor: {bld}")

    # 5. Offices / cities
    for office in redaction_lists.get("offices", []):
        if re.search(r"\b" + re.escape(office) + r"\b", text, re.IGNORECASE):
            leaks.append(f"Office: {office}")

    # 6. Manager full names
    for mgr in redaction_lists.get("manager_full_names", []):
        if re.search(r"\b" + re.escape(mgr) + r"\b", text, re.IGNORECASE):
            leaks.append(f"Manager: {mgr}")

    # 7. Employee full names
    for emp in redaction_lists.get("employee_full_names", []):
        if re.search(r"\b" + re.escape(emp) + r"\b", text, re.IGNORECASE):
            leaks.append(f"Employee: {emp}")

    return len(leaks) > 0, leaks


def enforce_code_escalation_and_safety_net(
    merged_output: Dict[str, Any],
    raw_text: str = "",
) -> Dict[str, Any]:
    """
    Deterministic policy enforcement:
    1. If theme is Harassment, Discrimination, or Ethics, OR severity == 4:
       escalate = True, severity = 4.
    2. Fixed escalation route from table in hr-policy.md. Never the writer's manager.
    3. Keyword safety net forces needs_human_review = True.
    """
    out = dict(merged_output)

    theme = out.get("theme", "")
    severity = out.get("severity", 1)

    # 1. Escalation enforcement
    is_critical_theme = theme in ["Harassment", "Discrimination", "Ethics"]
    if is_critical_theme or severity == 4:
        out["escalate"] = True
        out["severity"] = 4
        # Assign fixed route
        out["escalation_route"] = ESCALATION_ROUTES.get(theme, "HR Ethics desk")
    else:
        if not out.get("escalate", False):
            out["escalation_route"] = ""
        else:
            # If LLM proposed escalation for another reason
            if not out.get("escalation_route"):
                out["escalation_route"] = "HR Ethics desk"

    # 2. Safety net keywords
    combined_text = (raw_text + " " + out.get("final_text", "")).lower()
    needs_review = out.get("needs_human_review", False)

    for kw in SAFETY_NET_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", combined_text):
            needs_review = True
            break

    # If reid_risk is High, also force human review
    if out.get("reid_risk") == "High":
        needs_review = True

    out["needs_human_review"] = needs_review

    # Ensure decided_by is populated
    if "decided_by" not in out:
        out["decided_by"] = "LLM"

    return out


def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validates dictionary against llm_output_schema.json using jsonschema.
    """
    try:
        # Separate extra internal tracking fields before validating against strict additionalProperties: false schema
        internal_keys = ["decided_by", "needs_human_review"]
        payload = {k: v for k, v in data.items() if k not in internal_keys}
        jsonschema.validate(instance=payload, schema=schema)
        return True, None
    except jsonschema.ValidationError as e:
        return False, e.message
    except Exception as e:
        return False, str(e)


def build_fallback_response(
    comment_id: str,
    step0_text: str,
    raw_text: str,
    pre_screen_injection: bool = False,
    error_msg: str = "",
) -> Dict[str, Any]:
    """
    Fail-closed fallback response if LLM calls or validation fail after retry.
    - Layer 1 falls back to step 0 text.
    - Layer 2 falls back to reid_risk = "High" and needs_human_review = True (held, not shown to HR).
    - Classification falls back to theme from keyword heuristics.
    - decided_by = "Fallback rule".
    """
    lower = raw_text.lower()
    theme = "Workload"
    sub_theme = "General feedback"
    domain = "Work conditions"
    issue_target = "Policy"
    sentiment = "Negative"
    severity = 2
    escalate = False
    escalation_route = ""

    # Heuristic keyword check
    if pre_screen_injection or any(re.search(p, raw_text, re.IGNORECASE) for p in INJECTION_PATTERNS):
        theme = "Prompt injection attempt"
        domain = "Not feedback"
        sentiment = "Neutral"
        severity = 1
    elif any(k in lower for k in ["harass", "posh", "sexual", "touching"]):
        theme = "Harassment"
        domain = "Conduct & safety"
        severity = 4
        escalate = True
        escalation_route = ESCALATION_ROUTES["Harassment"]
    elif any(k in lower for k in ["caste", "religion", "gender", "married", "pregnancy", "bias"]):
        theme = "Discrimination"
        domain = "Conduct & safety"
        severity = 4
        escalate = True
        escalation_route = ESCALATION_ROUTES["Discrimination"]
    elif any(k in lower for k in ["bribe", "kickback", "falsified", "inflated", "fraud", "cheating"]):
        theme = "Ethics"
        domain = "Conduct & safety"
        severity = 4
        escalate = True
        escalation_route = ESCALATION_ROUTES["Ethics"]
    elif any(k in lower for k in ["pay", "salary", "hike", "bonus", "compensation"]):
        theme = "Pay"
        domain = "Rewards"
    elif any(k in lower for k in ["manager", "boss", "lead", "shouting", "pings"]):
        theme = "Manager behaviour"
        domain = "People & leadership"
        issue_target = "Person"
    elif any(k in lower for k in ["rto", "office", "wfh", "commute", "travel"]):
        theme = "Work location"
        domain = "Work conditions"

    return {
        "comment_id": comment_id,
        "layer1_text": step0_text,
        "final_text": step0_text,
        "pii_found": [],
        "reid_risk": "High",
        "reid_reason": f"Fallback rule activated: {error_msg}".strip(),
        "is_feedback": theme not in ["Prompt injection attempt", "No content"],
        "injection_detected": theme == "Prompt injection attempt",
        "theme": theme,
        "sub_theme": sub_theme,
        "domain": domain,
        "issue_target": issue_target,
        "sentiment": sentiment,
        "severity": severity,
        "escalate": escalate,
        "escalation_route": escalation_route,
        "decided_by": "Fallback rule",
        "needs_human_review": True,
    }
