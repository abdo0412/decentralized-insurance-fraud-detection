"""Streamlit Demo Application — Decentralized Insurance Fraud Detection.

Provides:
1. Input panel + 3 prepared example claims (Clearly Honest, Clearly Fraudulent, Borderline).
2. Specialist agent cards (Vision + Grad-CAM heatmap, Text, Tabular).
3. Final Decision panel with cost threshold and plain-English reason.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from orchestrator.coordinator import ClaimCoordinator
from agents.decision_agent import DecisionAgent
from agents.tabular_agent import TabularAgent


def make_demo_photo(case_name: str) -> Image.Image:
    """Load the real held-out collision photo prepared for each demo case."""
    assets_dir = ROOT_DIR / "app" / "demo_assets"
    file_map = {
        "Clearly Honest": assets_dir / "honest_severe.jpg",
        "Clearly Fraudulent": assets_dir / "fraud_minor_scratch.jpg",
        "Borderline": assets_dir / "borderline_moderate.jpg",
    }
    asset_path = file_map.get(case_name)
    if asset_path is not None and asset_path.exists():
        return Image.open(asset_path).convert("RGB")

    # Fallback synthetic image if demo_assets is missing
    img = Image.new("RGB", (320, 224), color=(215, 222, 230))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 80, 280, 170], fill=(70, 110, 180), outline=(30, 50, 90), width=3)
    return img


PREPARED_CASES: Dict[str, Dict[str, Any]] = {
    "Clearly Honest": {
        "claim_id": "C-0101",
        "incident_severity": "Total Loss",
        "total_claim_amount": 22500,
        "months_as_customer": 64,
        "witnesses": 2,
        "police_report_available": "YES",
        "narrative": (
            "On Tuesday October 12 at 08:30 AM at the Route 9 intersection, a delivery "
            "truck ran a red light and struck the passenger side of my vehicle at high speed. "
            "Two witnesses gave statements and the police officer filed report #PR-8821 on site."
        ),
    },
    "Clearly Fraudulent": {
        "claim_id": "C-0412",
        "incident_severity": "Total Loss",
        "total_claim_amount": 38500,
        "months_as_customer": 2,
        "witnesses": 0,
        "police_report_available": "NO",
        "narrative": (
            "Car was completely totaled somewhere on a dark road late at night, cannot "
            "remember the exact street. No witnesses and no police report. Urgent payout "
            "needed immediately via wire transfer ASAP."
        ),
    },
    "Borderline": {
        "claim_id": "C-0289",
        "incident_severity": "Total Loss",
        "total_claim_amount": 16200,
        "months_as_customer": 14,
        "witnesses": 0,
        "police_report_available": "YES",
        "narrative": (
            "Side collision in a parking garage. Rear right passenger door was dented "
            "and rear tire deflated. No witnesses stopped, police report was filed the next day."
        ),
    },
}


@st.cache_resource
def load_coordinator() -> ClaimCoordinator:
    from agents.vision_agent import VisionAgent, resolve_default_weights_path

    weights_path = resolve_default_weights_path(ROOT_DIR / "models")
    vision_agent = VisionAgent(
        weights_path=weights_path,
        backbone_name="efficientnet_v2_s",
        pretrained_fallback=False,
    )
    tabular_path = ROOT_DIR / "models" / "tabular_agent.joblib"
    decision_path = ROOT_DIR / "models" / "decision_agent.joblib"
    tabular_agent = TabularAgent.load(tabular_path) if tabular_path.exists() else TabularAgent()
    decision_agent = DecisionAgent.load(decision_path) if decision_path.exists() else DecisionAgent()
    return ClaimCoordinator(
        vision_agent=vision_agent,
        tabular_agent=tabular_agent,
        decision_agent=decision_agent,
    )


def main() -> None:
    st.set_page_config(page_title="Project 7 — Decentralized Fraud Detection", layout="wide")
    st.title("Decentralized Insurance Fraud Detection — Multi-Agent Audit")

    coordinator = load_coordinator()

    with st.sidebar:
        st.header("1. Load Prepared Case")
        selected_case = st.selectbox("Prepared Demo Claims", list(PREPARED_CASES.keys()))
        preset = PREPARED_CASES[selected_case]

        st.header("2. Claim Metadata")
        claim_id = st.text_input("Claim ID", value=preset["claim_id"])
        declared_severity = st.selectbox(
            "Declared Severity (`incident_severity`)",
            ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"],
            index=["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"].index(
                preset["incident_severity"]
            ),
        )
        claim_amount = st.number_input(
            "Claim Amount ($)", value=int(preset["total_claim_amount"]), step=500
        )
        policy_months = st.number_input(
            "Months as Customer", value=int(preset["months_as_customer"]), step=1
        )
        witnesses = st.number_input("Witnesses", value=int(preset["witnesses"]), min_value=0, max_value=5)
        police_report = st.selectbox(
            "Police Report Available",
            ["YES", "NO", "?"],
            index=["YES", "NO", "?"].index(preset["police_report_available"]),
        )
        threshold = st.slider("Decision Audit Threshold", 0.10, 0.90, 0.45, 0.05)
        coordinator.decision_agent.audit_threshold = threshold

    col_input_img, col_input_txt = st.columns([1, 1])
    with col_input_img:
        st.subheader("Collision Photo Input (Part A)")
        uploaded_file = st.file_uploader("Upload collision photo (optional)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file).convert("RGB")
        else:
            input_image = make_demo_photo(selected_case)

    with col_input_txt:
        st.subheader("Claim Narrative Input (Part B)")
        narrative = st.text_area("Written Claim Report", value=preset["narrative"], height=160)

    claim_fields = {
        "incident_severity": declared_severity,
        "total_claim_amount": claim_amount,
        "months_as_customer": policy_months,
        "witnesses": witnesses,
        "police_report_available": police_report,
    }

    results = coordinator.audit_claim(
        claim_id=claim_id,
        image_input=input_image,
        narrative=narrative,
        claim_fields=claim_fields,
    )
    gradcam_overlay, _, _ = coordinator.vision_agent.explain(input_image)
    fallback_agents = [
        name
        for name, message in results.items()
        if any(
            marker in str(message.get("model_version", "")).lower()
            for marker in ("fallback", "placeholder", "untrained")
        )
    ]
    if fallback_agents:
        st.warning(
            "Fallback or untrained model active for: "
            + ", ".join(fallback_agents)
            + ". Treat this demo output as illustrative, not a trained-system result."
        )

    st.divider()
    st.subheader("Specialist Agent Messages (Decentralized JSON Envelopes)")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### Vision Agent (CNN + Grad-CAM)")
        img_col1, img_col2 = st.columns(2)
        img_col1.image(input_image, caption="Input Photo", use_container_width=True)
        img_col2.image(gradcam_overlay, caption="Grad-CAM Focus", use_container_width=True)
        st.json(results["vision"])

    with c2:
        st.markdown("#### Text Agent (NB / DistilBERT)")
        st.json(results["text"])

    with c3:
        st.markdown("#### Tabular Context Agent")
        st.json(results["tabular"])

    st.divider()
    st.subheader("Final Decision Agent (Random Forest + Cost Threshold)")
    dec_out = results["decision"]["output"]
    if dec_out["decision"] == "audit":
        st.error(f"**FINAL DECISION: SEND TO HUMAN AUDIT** — {dec_out['reason']}")
    else:
        st.success(f"**FINAL DECISION: VALIDATE CLAIM** — {dec_out['reason']}")

    st.json(results["decision"])
    factors = dec_out.get("top_factors", {})
    if factors:
        explanation_method = dec_out.get("explanation_method", "unspecified")
        st.subheader("Decision factors")
        st.caption(f"Explanation method: {explanation_method}")
        st.dataframe(
            pd.DataFrame(
                [
                    {"factor": name, "contribution": value}
                    for name, value in factors.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
