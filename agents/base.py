"""Shared message contract, severity mappings, and base agent interface.

Every specialist agent (Vision, Text, Tabular) and the final Decision agent
communicate exclusively through the JSON envelope defined in `AgentMessage`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import json
from typing import Any, Dict, List

# Canonical severity levels shared across Vision (A) and Tabular/Decision (C)
SEVERITY_CLASSES: List[str] = ["minor", "moderate", "severe"]
SEVERITY_TO_INDEX: Dict[str, int] = {label: idx for idx, label in enumerate(SEVERITY_CLASSES)}
INDEX_TO_SEVERITY: Dict[int, str] = {idx: label for idx, label in enumerate(SEVERITY_CLASSES)}

# Mapping from common folder names or Kaggle tabular `incident_severity` values
# to our canonical 3-class severity labels.
TABULAR_SEVERITY_MAP: Dict[str, str] = {
    # Image folder variants
    "01-minor": "minor",
    "02-moderate": "moderate",
    "03-severe": "severe",
    "minor": "minor",
    "moderate": "moderate",
    "severe": "severe",
    # Kaggle Car Insurance Claim Fraud Detection `incident_severity` values
    "trivial damage": "minor",
    "minor damage": "minor",
    "major damage": "moderate",
    "total loss": "severe",
}


def normalize_severity_label(raw_label: str) -> str:
    """Normalize any folder or tabular severity string into {'minor', 'moderate', 'severe'}."""
    cleaned = str(raw_label).strip().lower()
    if cleaned in TABULAR_SEVERITY_MAP:
        return TABULAR_SEVERITY_MAP[cleaned]
    for key, canonical in TABULAR_SEVERITY_MAP.items():
        if key in cleaned:
            return canonical
    raise ValueError(
        f"Unrecognized severity label: '{raw_label}'. "
        f"Expected one of {list(TABULAR_SEVERITY_MAP.keys())}."
    )


def compute_severity_mismatch(
    declared_severity: str,
    vision_probabilities: Dict[str, float],
) -> Dict[str, float]:
    """Compute the fraud signal comparing declared severity vs. photo-predicted severity.

    Returns both:
    - `severity_gap_discrete`: declared_rank - predicted_rank (positive = claim exaggerates damage)
    - `severity_gap_continuous`: declared_rank - E[photo_rank] using class probabilities
    """
    declared_norm = normalize_severity_label(declared_severity)
    declared_rank = float(SEVERITY_TO_INDEX[declared_norm])

    pred_label = max(SEVERITY_CLASSES, key=lambda c: vision_probabilities.get(c, 0.0))
    pred_rank = float(SEVERITY_TO_INDEX[pred_label])

    expected_photo_rank = sum(
        SEVERITY_TO_INDEX[c] * float(vision_probabilities.get(c, 0.0))
        for c in SEVERITY_CLASSES
    )

    return {
        "declared_rank": declared_rank,
        "predicted_rank": pred_rank,
        "expected_photo_rank": round(expected_photo_rank, 4),
        "severity_gap_discrete": round(declared_rank - pred_rank, 4),
        "severity_gap_continuous": round(declared_rank - expected_photo_rank, 4),
    }


@dataclass
class AgentMessage:
    """Standardized JSON envelope returned by every agent in the system.

    Example (Vision Agent):
    {
        "claim_id": "C-0412",
        "agent": "vision",
        "output": {
            "severity": "minor",
            "probabilities": {"minor": 0.71, "moderate": 0.22, "severe": 0.07}
        },
        "confidence": 0.71,
        "model_version": "resnet50-v1"
    }
    """

    claim_id: str
    agent: str
    output: Dict[str, Any]
    confidence: float
    model_version: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        return cls(
            claim_id=str(data["claim_id"]),
            agent=str(data["agent"]),
            output=dict(data["output"]),
            confidence=float(data["confidence"]),
            model_version=str(data["model_version"]),
        )


class BaseAgent(ABC):
    """Abstract base class that all specialist and decision agents inherit from."""

    agent_name: str = "base"
    model_version: str = "v1"

    @abstractmethod
    def predict(self, claim_id: str, **kwargs: Any) -> AgentMessage:
        """Score a single claim and return a standardized `AgentMessage`."""
        raise NotImplementedError
