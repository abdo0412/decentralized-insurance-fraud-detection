"""Part C: Optional Tabular Risk Agent.

Owner: Person C.
Scores structured claim metadata (claim amount, policy age, witnesses, police report).
"""

from __future__ import annotations

from typing import Any, Dict

from agents.base import AgentMessage, BaseAgent


class TabularAgent(BaseAgent):
    """Computes a tabular fraud risk score from structured claim fields."""

    agent_name = "tabular"
    model_version = "rf-tabular-placeholder-v1"

    def predict(
        self,
        claim_id: str,
        claim_fields: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentMessage:
        fields = claim_fields or {}
        amount = float(fields.get("total_claim_amount", 5000.0))
        policy_months = float(fields.get("months_as_customer", 24.0))
        witnesses = int(fields.get("witnesses", 1))
        police_report = str(fields.get("police_report_available", "YES")).upper()

        risk = 0.20
        if amount > 20000:
            risk += 0.25
        if policy_months < 6:
            risk += 0.20
        if witnesses == 0:
            risk += 0.10
        if police_report in ("NO", "?"):
            risk += 0.15

        risk_score = round(min(0.95, max(0.05, risk)), 4)
        confidence = round(max(risk_score, 1.0 - risk_score), 4)

        return AgentMessage(
            claim_id=str(claim_id),
            agent=self.agent_name,
            output={"risk_score": risk_score},
            confidence=confidence,
            model_version=self.model_version,
        )
