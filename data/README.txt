HR Feedback System - synthetic dataset
=======================================

Fictional company "Navrang Systems": 2,000 employees, 174 teams, 6 Indian cities, Nov 2024 - Oct 2026.
Every name, comment and number is made up.

WHICH FILE FOR WHAT
- Want to look at the data?          -> HR_Feedback_Synthetic_Dataset.xlsx (start with the "Start here" tab)
- Building the risk model / app?     -> data/*.csv (same tables as the Excel)
- Running the anonymiser LLM?        -> llm/ (JSON files, see below)

WHY TWO FORMATS
Tables (CSV / Excel) are the data itself: one row per comment, team-month, session and so on.
JSON is for what goes in and out of the LLM. The LLM should always reply in the same fixed JSON
format so the app can read it, check it, and reject anything malformed (this is one of our guardrails).

data/
  feedback_comments.csv        4,090 comments: raw text + correct anonymised text (layer 1, layer 2) + correct labels
  team_monthly_panel.csv       4,176 team-months: HR signals, monthly complaint score, attrition target for the model
  teams.csv                    174 teams and their managers
  employees_hris.csv           2,663 employees (incl. leavers and new joiners)
  listening_sessions.csv       330 alarms -> session held? -> actions agreed/closed -> what happened next
  manager_yearend_review.csv   team risk factor (0-100) for the manager's year-end review, team level only
  hr_review_disputes.csv       128 disputes from the monthly "argue with HR" window
  identity_vault_SEALED.csv    token -> employee. Ombudsperson only. Never load into the app's HR view
  policy_events.csv            appraisals, leave policy change, return-to-office (Apr 2026), manager reshuffle
  answer_key_hidden_drivers.csv  hidden drivers used to generate the data. For Q&A only, never a model input

llm/
  llm_output_schema.json       the exact JSON the LLM must return for each comment
  comments_input_all.jsonl     all 4,090 comments, one per line: {"comment_id", "text"} - nothing else
  test_set_input.jsonl         the 502-comment test set (all hard cases included), same format
  test_set_gold.jsonl          the correct answer for each test comment, in the output schema format
  redaction_lists.json         names, employee ID pattern, project codes, clients, offices (layer 1 + leak check)

HOW TO TEST THE LLM
1. Send each line of test_set_input.jsonl to the LLM with your prompt.
2. Ask it to reply with JSON matching llm_output_schema.json.
3. Compare its replies with test_set_gold.jsonl, matching on comment_id:
   - leak rate: does final_text still contain anything from redaction_lists.json?
   - label accuracy: theme, issue_target, sentiment, severity
   - escalations missed: gold escalate = true but LLM said false (the most important number)
   - injections caught: injection_detected on the 53 planted attempts

DO NOT give the LLM employee_id_hidden, team_id, identity_token, the vault, or any gold_ column.

scripts
  generate_dataset.py   rebuilds data/ (fixed seed, same output every time)
  build_extras.py       listening sessions, year-end review, llm/ files
  build_workbook.py     the Excel
  train_model_v1.py     first attrition model vs the historical-rate baseline
