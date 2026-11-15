"""Part C: Decision Agent (Random Forest on agent outputs + severity mismatch).

Owner: Person C.
Combines the Vision, Text, and Tabular agent messages with the declared claim
severity to output `validate` or `audit` and a plain-English explanation.
"""

from __future__ import annotations

from typing import Any, Dict

from agents.base import AgentMessage, BaseAgent, compute_severity_mismatch


class DecisionAgent(BaseAgent):
    """Final meta-agent that weighs all specialist agent messages."""

    agent_name = "decision"
    model_version = "rf-decision-v1"

    def __init__(self, audit_threshold: float = 0.45) -> None:
        self.audit_threshold = audit_threshold

    def predict(
        self,
        claim_id: str,
        vision_msg: AgentMessage | Dict[str, Any],
        text_msg: AgentMessage | Dict[str, Any],
        tabular_msg: AgentMessage | Dict[str, Any] | None = None,
        claim_fields: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentMessage:
        fields = claim_fields or {}
        v_out = vision_msg.output if isinstance(vision_msg, AgentMessage) else vision_msg["output"]
        t_out = text_msg.output if isinstance(text_msg, AgentMessage) else text_msg["output"]
        tab_out = (
            tabular_msg.output
            if isinstance(tabular_msg, AgentMessage)
            else (tabular_msg["output"] if tabular_msg else {"risk_score": 0.25})
        )

        declared_sev = str(fields.get("incident_severity", "minor"))
        vision_probs = v_out.get("probabilities", {"minor": 0.34, "moderate": 0.33, "severe": 0.33})
        mismatch = compute_severity_mismatch(declared_sev, vision_probs)

        text_suspicion = float(t_out.get("suspicion_score", 0.2))
        tab_risk = float(tab_out.get("risk_score", 0.2))
        sev_gap = max(0.0, float(mismatch["severity_gap_continuous"])) / 2.0  # Normalized [0, 1]

        # Weighted fusion fallback (replaced by trained scikit-learn RandomForest by Person C)
        audit_prob = round(
            min(0.99, max(0.01, 0.45 * sev_gap + 0.35 * text_suspicion + 0.20 * tab_risk)),
            4,
        )
        decision_label = "audit" if audit_prob >= self.audit_threshold else "validate"

        contributions = {
            "severity_mismatch": round(0.45 * sev_gap, 4),
            "text_suspicion_score": round(0.35 * text_suspicion, 4),
            "tabular_risk_score": round(0.20 * tab_risk, 4),
        }

        reasons = []
        if mismatch["severity_gap_discrete"] >= 1:
            reasons.append(
                f"declared severity is {declared_sev} but photo shows {v_out.get('severity')} damage"
            )
        if text_suspicion >= 0.5:
            reasons.append("report wording scores as suspicious")
        if tab_risk >= 0.5:
            reasons.append("claim metadata indicates elevated risk")
        if not reasons:
            reasons.append("photo severity, narrative, and claim metadata are consistent")

        prefix = (
            f"Audit (fraud probability {audit_prob:.2f} >= threshold {self.audit_threshold:.2f})"
            if decision_label == "audit"
            else f"Validate (fraud probability {audit_prob:.2f} < threshold {self.audit_threshold:.2f})"
        )
        reason_sentence = f"{prefix}: {', and '.join(reasons)}."

        return AgentMessage(
            claim_id=str(claim_id),
            agent=self.agent_name,
            output={
                "decision": decision_label,
                "audit_probability": audit_prob,
                "threshold": self.audit_threshold,
                "severity_mismatch": mismatch,
                "top_factors": contributions,
                "reason": reason_sentence,
            },
            confidence=round(max(audit_prob, 1.0 - audit_prob), 4),
            model_version=self.model_version,
        )
