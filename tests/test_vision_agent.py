"""Unit and integration tests for Part A (Vision Agent, Grad-CAM, and JSON schema)."""

from __future__ import annotations

from pathlib import Path
import sys

from PIL import Image
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.base import (
    SEVERITY_CLASSES,
    compute_severity_mismatch,
    normalize_severity_label,
)
from agents.vision_agent import VisionAgent
from orchestrator.coordinator import ClaimCoordinator


def test_severity_normalization_and_mismatch() -> None:
    assert normalize_severity_label("01-minor") == "minor"
    assert normalize_severity_label("Major Damage") == "moderate"
    assert normalize_severity_label("Total Loss") == "severe"

    mismatch = compute_severity_mismatch(
        declared_severity="Total Loss",
        vision_probabilities={"minor": 0.70, "moderate": 0.20, "severe": 0.10},
    )
    assert mismatch["declared_rank"] == 2.0
    assert mismatch["predicted_rank"] == 0.0
    assert mismatch["severity_gap_discrete"] == 2.0
    assert mismatch["severity_gap_continuous"] > 1.5


def test_vision_agent_contract_and_gradcam() -> None:
    # Use mobilenet_v3_large without downloading ImageNet weights for fast offline unit testing
    agent = VisionAgent(
        weights_path=None,
        backbone_name="mobilenet_v3_large",
        pretrained_fallback=False,
    )
    dummy_img = Image.new("RGB", (256, 256), color=(120, 80, 90))

    msg = agent.predict(claim_id="C-0412", image_input=dummy_img)
    payload = msg.to_dict()

    assert payload["claim_id"] == "C-0412"
    assert payload["agent"] == "vision"
    assert payload["output"]["severity"] in SEVERITY_CLASSES

    probs = payload["output"]["probabilities"]
    assert set(probs.keys()) == set(SEVERITY_CLASSES)
    assert pytest.approx(sum(probs.values()), abs=1e-2) == 1.0
    assert 0.0 <= payload["confidence"] <= 1.0

    overlay, heatmap, _ = agent.explain(dummy_img)
    assert isinstance(overlay, Image.Image)
    assert overlay.size == dummy_img.size
    assert heatmap.shape == (256, 256)


def test_end_to_end_coordinator() -> None:
    vision_agent = VisionAgent(
        weights_path=None,
        backbone_name="mobilenet_v3_large",
        pretrained_fallback=False,
    )
    coord = ClaimCoordinator(vision_agent=vision_agent)
    dummy_img = Image.new("RGB", (224, 224), color=(90, 120, 150))

    result = coord.audit_claim(
        claim_id="C-9999",
        image_input=dummy_img,
        narrative="Urgent payout needed ASAP, no witnesses, dark road.",
        claim_fields={
            "incident_severity": "Total Loss",
            "total_claim_amount": 35000,
            "months_as_customer": 2,
            "witnesses": 0,
            "police_report_available": "NO",
        },
    )

    assert set(result.keys()) == {"vision", "text", "tabular", "decision"}
    assert result["decision"]["output"]["decision"] in ("validate", "audit")
