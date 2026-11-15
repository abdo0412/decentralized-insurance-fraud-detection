"""Part B: Text Agent (Naive Bayes baseline + DistilBERT upgrade).

Owner: Person B.
This module provides a clean interface + keyword/heuristic fallback so end-to-end
integration tests and the demo app work immediately while Person B trains the models.
"""

from __future__ import annotations

from typing import Any, List

from agents.base import AgentMessage, BaseAgent

SUSPICIOUS_CUES: List[str] = [
    "urgent",
    "immediately",
    "cash",
    "no witnesses",
    "no police",
    "somewhere",
    "dark road",
    "totaled",
    "wire transfer",
    "asap",
    "cannot remember",
]


class TextAgent(BaseAgent):
    """Scores how suspicious a written claim narrative is in [0, 1]."""

    agent_name = "text"
    model_version = "nb-distilbert-placeholder-v1"

    def predict(self, claim_id: str, narrative: str = "", **kwargs: Any) -> AgentMessage:
        text_lower = str(narrative).lower()
        triggered = [cue for cue in SUSPICIOUS_CUES if cue in text_lower]

        # Simple heuristic fallback until Person B plugs in trained NB / DistilBERT weights
        base_score = 0.18 + 0.18 * len(triggered)
        suspicion_score = round(min(0.95, max(0.05, base_score)), 4)
        confidence = round(max(suspicion_score, 1.0 - suspicion_score), 4)

        return AgentMessage(
            claim_id=str(claim_id),
            agent=self.agent_name,
            output={
                "suspicion_score": suspicion_score,
                "label": "suspicious" if suspicion_score >= 0.5 else "normal",
                "top_tokens": triggered[:5],
            },
            confidence=confidence,
            model_version=self.model_version,
        )
