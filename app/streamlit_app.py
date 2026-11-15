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
from PIL import Image, ImageDraw
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from orchestrator.coordinator import ClaimCoordinator


def make_demo_photo(case_name: str) -> Image.Image:
    """Create a synthetic demo collision image if no photo is uploaded."""
    img = Image.new("RGB", (320, 224), color=(215, 222, 230))
    draw = ImageDraw.Draw(img)
    # Car body silhouette
    draw.rectangle([40, 80, 280, 170], fill=(70, 110, 180), outline=(30, 50, 90), width=3)
    if case_name == "Clearly Fraudulent":
        # Tiny scratch (minor photo vs severe claim)
        draw.line([230, 115, 255, 125], fill=(240, 200, 40), width=4)
    elif case_name == "Borderline":
        # Moderate dent
        draw.ellipse([200, 100, 265, 155], fill=(220, 120, 40), outline=(140, 50, 20), width=3)
    else:
        # Honest minor scratch
        draw.line([225, 110, 250, 130], fill=(230, 190, 50), width=4)
    return img


PREPARED_CASES: Dict[str, Dict[str, Any]] = {
    "Clearly Honest": {
        "claim_id": "C-0101",
        "incident_severity": "Minor Damage",
        "total_claim_amount": 1850,
        "months_as_customer": 48,
        "witnesses": 2,
        "police_report_available": "YES",
        "narrative": (
            "On Tuesday October 12 at 08:30 AM at the Main St parking lot, another driver "
            "backed into my rear bumper at low speed. Two witnesses saw the bump and a "
            "police officer filed a report on site. Only minor paint scratch on the bumper."
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
        "incident_severity": "Major Damage",
        "total_claim_amount": 14200,
        "months_as_customer": 14,
        "witnesses": 0,
        "police_report_available": "YES",
        "narrative": (
            "Side collision at an intersection during rain. Front right fender was dented "
            "and headlight cracked. No witnesses stopped, police report was filed next day."
        ),
    },
}


@st.cache_resource
def load_coordinator() -> ClaimCoordinator:
    weights_path = ROOT_DIR / "models" / "vision_resnet50_v1.pt"
    from agents.vision_agent import VisionAgent

    vision_agent = VisionAgent(
        weights_path=weights_path if weights_path.exists() else None,
        backbone_name="resnet50",
        pretrained_fallback=False,
    )
    return ClaimCoordinator(vision_agent=vision_agent)


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


if __name__ == "__main__":
    main()
