"""Train the tabular fraud agent and write train-only OOF / validation scores.

This script intentionally does not read the final test split. Its OOF scores
can be joined with A/B's matching claim-ID predictions when those handoffs are
available for training the final decision agent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.tabular_agent import TabularAgent


def _require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def train_tabular_agent(
    claims_dir: Path,
    model_path: Path,
    output_dir: Path,
    target_column: str = "fraud_reported",
    threshold: float = 0.5,
) -> dict[str, Any]:
    train_path = claims_dir / "claims_train.csv"
    validation_path = claims_dir / "claims_validation.csv"
    folds_path = claims_dir / "claim_folds.csv"
    for path in (train_path, validation_path, folds_path):
        if not path.exists():
            raise FileNotFoundError(f"Required claims input is missing: {path}")

    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    folds = pd.read_csv(folds_path)
    for frame, path in ((train, train_path), (validation, validation_path)):
        _require_columns(frame, {"claim_id", target_column}, path)
    _require_columns(folds, {"claim_id", "fold"}, folds_path)
    if train["claim_id"].astype(str).duplicated().any():
        raise ValueError("Training claim IDs must be unique.")
    if folds["claim_id"].astype(str).duplicated().any():
        raise ValueError("The fold manifest must have exactly one row per training claim.")

    train = train.copy()
    train["claim_id"] = train["claim_id"].astype(str)
    folds = folds.copy()
    folds["claim_id"] = folds["claim_id"].astype(str)
    fold_table = train[["claim_id"]].merge(folds, on="claim_id", how="left", validate="one_to_one")
    if fold_table["fold"].isna().any() or len(folds) != len(train):
        raise ValueError("The fold manifest must cover every training claim exactly once.")
    train["fold"] = fold_table["fold"].astype(int).to_numpy()
    fold_values = sorted(train["fold"].unique().tolist())
    if len(fold_values) < 2:
        raise ValueError("At least two distinct folds are required for out-of-fold predictions.")

    feature_columns = [column for column in train.columns if column not in {target_column, "fold"}]
    oof_scores = np.full(len(train), np.nan, dtype=float)
    for fold in fold_values:
        train_mask = train["fold"] != fold
        score_mask = train["fold"] == fold
        fold_model = TabularAgent().fit(
            train.loc[train_mask, feature_columns],
            train.loc[train_mask, target_column],
        )
        oof_scores[score_mask.to_numpy()] = fold_model.predict_proba(
            train.loc[score_mask, feature_columns]
        )

    if np.isnan(oof_scores).any():
        raise RuntimeError("Some training rows did not receive an out-of-fold tabular score.")
    oof_output = pd.DataFrame(
        {
            "claim_id": train["claim_id"],
            target_column: train[target_column],
            "fold": train["fold"],
            "tabular_risk_score": oof_scores,
            "model_version": "rf-tabular-oof-v1",
        }
    )

    full_model = TabularAgent().fit(train[feature_columns], train[target_column])
    full_model.save(model_path)
    validation["claim_id"] = validation["claim_id"].astype(str)
    validation_features = [column for column in feature_columns if column in validation.columns]
    validation_scores = full_model.predict_proba(validation[validation_features])
    validation_output = pd.DataFrame(
        {
            "claim_id": validation["claim_id"],
            target_column: validation[target_column],
            "tabular_risk_score": validation_scores,
            "model_version": full_model.model_version,
        }
    )
    actual = validation[target_column].astype(int).to_numpy()
    predicted = (validation_scores >= threshold).astype(np.int64)
    metrics = {
        "threshold": float(threshold),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "pr_auc": float(average_precision_score(actual, validation_scores)),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
        "validation_rows": int(len(validation)),
        "training_rows": int(len(train)),
        "folds": int(len(fold_values)),
        "note": "Scores use the validation split; the final test split was not read.",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    oof_output.to_csv(output_dir / "tabular_oof_predictions.csv", index=False)
    validation_output.to_csv(output_dir / "tabular_validation_predictions.csv", index=False)
    (output_dir / "tabular_validation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"Saved tabular OOF scores and validation scores beneath {output_dir.resolve()}")
    print(f"Saved trained TabularAgent to {model_path.resolve()}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--claims-dir",
        type=Path,
        default=Path("data/processed/claims_fraud_detection"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/claims_fraud_detection"),
    )
    parser.add_argument("--model-path", type=Path, default=Path("models/tabular_agent.joblib"))
    parser.add_argument("--target-column", default="fraud_reported")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    metrics = train_tabular_agent(
        claims_dir=args.claims_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        target_column=args.target_column,
        threshold=args.threshold,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
