"""Decentralized Claim Coordinator.

Routes an incoming insurance claim (collision photo, narrative report, tabular fields)
to each specialist agent, collects their JSON envelopes, and forwards them to the
Decision Agent.
"""

from __future__ import annotations

from typing import Any, Dict

from PIL import Image

from agents.decision_agent import DecisionAgent
from agents.tabular_agent import TabularAgent
from agents.text_agent import TextAgent
from agents.vision_agent import VisionAgent


class ClaimCoordinator:
    """Orchestrates the Vision, Text, Tabular, and Decision agents."""

    def __init__(
        self,
        vision_agent: VisionAgent | None = None,
        text_agent: TextAgent | None = None,
        tabular_agent: TabularAgent | None = None,
        decision_agent: DecisionAgent | None = None,
    ) -> None:
        self.vision_agent = vision_agent or VisionAgent()
        self.text_agent = text_agent or TextAgent()
        self.tabular_agent = tabular_agent or TabularAgent()
        self.decision_agent = decision_agent or DecisionAgent()

    def audit_claim(
        self,
        claim_id: str,
        image_input: str | Image.Image,
        narrative: str,
        claim_fields: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Run all specialist agents and return their JSON messages + final decision."""
        vision_msg = self.vision_agent.predict(claim_id=claim_id, image_input=image_input)
        text_msg = self.text_agent.predict(claim_id=claim_id, narrative=narrative)
        tabular_msg = self.tabular_agent.predict(claim_id=claim_id, claim_fields=claim_fields)

        decision_msg = self.decision_agent.predict(
            claim_id=claim_id,
            vision_msg=vision_msg,
            text_msg=text_msg,
            tabular_msg=tabular_msg,
            claim_fields=claim_fields,
        )

        return {
            "vision": vision_msg.to_dict(),
            "text": text_msg.to_dict(),
            "tabular": tabular_msg.to_dict(),
            "decision": decision_msg.to_dict(),
        }
