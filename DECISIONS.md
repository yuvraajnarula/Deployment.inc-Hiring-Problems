# Architectural Decisions

## 1. Deterministic Pre-Router vs. LLM Classifier
- **Hypothesis:** An LLM classifier could route cases more accurately than regex.
- **Constraint:** API spend ceiling ($50) and the "Zero-GPU" reproduction rule.
- **Observation:** 30% of cases contain unconditional triggers ("insolvency", "top-up") that require zero backend state inspection.
- **Decision:** Implemented a zero-cost deterministic regex router.
- **Reversal Condition:** If the case mix shifts to <10% unconditional escalations, the ROI of the router drops, and an LLM classifier becomes viable for nuanced routing.

## 2. Aggressive Context Compaction
- **Hypothesis:** Feeding raw JSON from `lookup_loan` helps the model reason about edge cases.
- **Constraint:** Context bleed was causing 72-turn loops and $0.05+ costs per case.
- **Observation:** The model only needs `status`, `balance`, and `verified` state to make policy decisions.
- **Decision:** Intercepted tool outputs and summarized them into 1-sentence strings.
- **Reversal Condition:** If quality drops below the 85.3% floor on the held-out set due to missing nuance (e.g., specific payment dates), we will revert to raw JSON for the `payment_history` tool only.

## 3. JSON-Only Tool Calling vs. Native Function Calling
- **Hypothesis:** Native function calling (OpenAI tool API) is more reliable.
- **Constraint:** The baseline `llm.py` uses a strict JSON-reply path because provider tool-calling surfaces were inconsistent across snapshots.
- **Decision:** Retained JSON-only parsing but added strict schema filtering via dynamic prompt injection.
- **Reversal Condition:** If the gateway proxy is upgraded to reliably normalize tool calls across all budget-tier models.