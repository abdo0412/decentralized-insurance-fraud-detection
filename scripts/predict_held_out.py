"""Step 3 (Part A): Batch Inference on Linked Held-Out Photos for Person C.

Runs the trained `VisionAgent` (4-Fold EfficientNetV2-S Ensemble + Flip TTA)
across all claims in `linked_claims.csv` (or `held_out_images.csv`)
and exports probabilities, predicted severity, confidence, and severity mismatch.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.base import compute_severity_mismatch
from agents.vision_agent import VisionAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Score held-out photos for Person C.")
    parser.add_argument(
        "--claims-csv",
        type=Path,
        default=Path("data/processed/held_out_images.csv"),
        help="CSV with `image_path` and optional `claim_id` and `declared_severity`.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="auto",
        help="Path to trained Vision CNN checkpoint, or 'auto' to pick strongest checkpoint.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_v2_s",
        help="CNN backbone name (overridden by checkpoint metadata when present).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/vision_predictions.csv"),
        help="Output CSV path for Person C's Decision Agent.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.claims_csv)
    agent = VisionAgent(weights_path=args.weights, backbone_name=args.backbone)
    print(f"[INFO] Loaded VisionAgent ({agent.model_version}, {len(agent.models)} fold(s), TTA={agent.use_tta})")

    rows: List[Dict] = []
    for idx, row in df.iterrows():
        claim_id = str(row.get("claim_id", row.get("image_id", f"C-{idx:04d}")))
        img_path = str(row["image_path"])
        msg = agent.predict(claim_id=claim_id, image_input=img_path)
        probs = msg.output["probabilities"]

        record: Dict = {
            "claim_id": claim_id,
            "image_path": img_path,
            "pred_severity": msg.output["severity"],
            "prob_minor": probs["minor"],
            "prob_moderate": probs["moderate"],
            "prob_severe": probs["severe"],
            "vision_confidence": msg.confidence,
            "model_version": msg.model_version,
        }

        if "declared_severity" in row or "incident_severity" in row:
            declared = str(row.get("declared_severity", row.get("incident_severity")))
            mismatch = compute_severity_mismatch(declared, probs)
            record["declared_severity"] = declared
            record["severity_gap_discrete"] = mismatch["severity_gap_discrete"]
            record["severity_gap_continuous"] = mismatch["severity_gap_continuous"]

        rows.append(record)

    out_df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"[INFO] Exported {len(out_df)} vision predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
