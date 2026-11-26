"""Rebuild C's Decision Agent train/validation tables from specialist outputs.

The script joins C claim splits to B's text scores, C's tabular scores, and A's
vision predictions using claim IDs and linked image paths. It never builds a
Decision Agent table from the frozen test split.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.base import compute_severity_mismatch


def _read_unique(path: Path, key: str, *, string_key: bool = False) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    dtype = {key: "string"} if string_key else None
    frame = pd.read_csv(path, dtype=dtype)
    if key not in frame:
        raise ValueError(f"{path} must contain `{key}`.")
    if frame[key].isna().any() or frame[key].duplicated().any():
        raise ValueError(f"{path} must have one non-empty row per `{key}`.")
    frame[key] = frame[key].astype(str)
    return frame


def _assert_id_coverage(frame: pd.DataFrame, expected_ids: Iterable[str], source: str) -> None:
    expected = set(expected_ids)
    actual = set(frame["claim_id"].astype(str))
    if actual != expected:
        missing = sorted(expected - actual)[:5]
        extra = sorted(actual - expected)[:5]
        raise ValueError(f"{source} claim IDs do not match the claim split; missing={missing}, extra={extra}.")


def _numeric(frame: pd.DataFrame, column: str, source: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"{source} is missing `{column}`.")
    values = pd.to_numeric(frame[column], errors="raise")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{source} has missing or non-finite `{column}` values.")
    return values


def _check_probability(values: pd.Series, column: str, source: str) -> pd.Series:
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError(f"{source} `{column}` scores must be within [0, 1].")
    return values.astype(float)


def _build_split(
    split_name: str,
    claims_path: Path,
    linked: pd.DataFrame,
    text: pd.DataFrame,
    tabular: pd.DataFrame,
    vision: pd.DataFrame,
    folds: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    claims = _read_unique(claims_path, "claim_id", string_key=True)
    if "fraud_reported" not in claims or "incident_severity" not in claims:
        raise ValueError(f"{claims_path} must include `fraud_reported` and `incident_severity`.")
    claims["fraud_reported"] = pd.to_numeric(claims["fraud_reported"], errors="raise").astype(int)
    if not set(claims["fraud_reported"].unique()).issubset({0, 1}):
        raise ValueError(f"{claims_path} has fraud labels outside 0/1.")

    linked_split = linked.loc[linked["split"].astype(str) == split_name].copy()
    _assert_id_coverage(linked_split, claims["claim_id"], f"Linked {split_name} data")
    linked_split = linked_split.set_index("claim_id").loc[claims["claim_id"]].reset_index()
    if not linked_split["fraud_reported"].astype(int).equals(claims["fraud_reported"]):
        raise ValueError(f"Linked {split_name} labels do not match {claims_path}.")

    _assert_id_coverage(text, claims["claim_id"], f"Text {split_name} predictions")
    _assert_id_coverage(tabular, claims["claim_id"], f"Tabular {split_name} predictions")
    text = text.set_index("claim_id").loc[claims["claim_id"]].reset_index()
    tabular = tabular.set_index("claim_id").loc[claims["claim_id"]].reset_index()

    if "suspicion_score" not in text:
        raise ValueError("Text predictions must include `suspicion_score`.")
    text_score = _check_probability(_numeric(text, "suspicion_score", "Text predictions"), "suspicion_score", "Text predictions")
    tabular_score = _check_probability(_numeric(tabular, "tabular_risk_score", "Tabular predictions"), "tabular_risk_score", "Tabular predictions")
    if "model_version" not in text or "model_version" not in tabular:
        raise ValueError("Text and Tabular predictions must include `model_version`.")

    vision_join = linked_split[["claim_id", "image_path", "declared_severity", "linked_image_id"]].merge(
        vision,
        on="image_path",
        how="left",
        validate="many_to_one",
        suffixes=("", "_vision"),
    )
    if vision_join[["prob_minor", "prob_moderate", "prob_severe", "model_version_vision"]].isna().any().any():
        raise ValueError(f"Some linked {split_name} images lack Vision prediction scores.")
    if "claim_id_vision" in vision_join:
        mismatch = vision_join["linked_image_id"].astype(str) != vision_join["claim_id_vision"].astype(str)
        if mismatch.any():
            raise ValueError(f"Linked image IDs do not match the Vision prediction paths for {split_name}.")

    probability_columns = ["prob_minor", "prob_moderate", "prob_severe"]
    for column in probability_columns:
        vision_join[column] = _check_probability(
            _numeric(vision_join, column, "Vision predictions"), column, "Vision predictions"
        )
    probability_sum = vision_join[probability_columns].sum(axis=1)
    if ((probability_sum - 1.0).abs() > 0.003).any():
        raise ValueError(f"Vision probabilities for {split_name} do not sum to approximately 1.")

    rows: list[dict[str, object]] = []
    for index, claim in claims.iterrows():
        vision_row = vision_join.iloc[index]
        probabilities = {
            "minor": float(vision_row["prob_minor"]),
            "moderate": float(vision_row["prob_moderate"]),
            "severe": float(vision_row["prob_severe"]),
        }
        mismatch = compute_severity_mismatch(claim["incident_severity"], probabilities)
        row: dict[str, object] = {
            "claim_id": claim["claim_id"],
            "fraud_reported": int(claim["fraud_reported"]),
            "text_suspicion_score": float(text_score.iloc[index]),
            "tabular_risk_score": float(tabular_score.iloc[index]),
            "vision_prob_minor": probabilities["minor"],
            "vision_prob_moderate": probabilities["moderate"],
            "vision_prob_severe": probabilities["severe"],
            "severity_gap_discrete": mismatch["severity_gap_discrete"],
            "severity_gap_continuous": mismatch["severity_gap_continuous"],
            "positive_severity_gap": max(0.0, mismatch["severity_gap_continuous"]) / 2.0,
            "text_model_version": str(text.iloc[index]["model_version"]),
            "tabular_model_version": str(tabular.iloc[index]["model_version"]),
            "vision_model_version": str(vision_row["model_version_vision"]),
        }
        if split_name == "train":
            row["fold"] = int(folds.set_index("claim_id").loc[claim["claim_id"], "fold"])
        for column, value in claim.items():
            if column not in {"claim_id", "fraud_reported", "split", "_c39"}:
                row[f"claim_{column}"] = value
        rows.append(row)

    result = pd.DataFrame(rows)
    if split_name == "train":
        # Preserve the established table order with the fold immediately after the ID.
        ordered = ["claim_id", "fold"] + [column for column in result if column not in {"claim_id", "fold"}]
        result = result[ordered]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Wrote {len(result)} {split_name} Decision Agent rows -> {output_path.resolve()}")
    return result


def prepare_decision_agent_data(data_dir: Path, linked_path: Path, vision_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild training and validation feature tables without reading test claims."""
    folds = _read_unique(data_dir / "claim_folds.csv", "claim_id", string_key=True)
    folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
    if sorted(folds["fold"].unique().tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError("claim_folds.csv must contain folds 0 through 4.")
    train_claims = _read_unique(data_dir / "claims_train.csv", "claim_id", string_key=True)
    _assert_id_coverage(folds, train_claims["claim_id"], "Fold assignments")

    linked = _read_unique(linked_path, "claim_id", string_key=True)
    if "split" not in linked or not {"image_path", "declared_severity", "linked_image_id"}.issubset(linked.columns):
        raise ValueError("Linked claim file needs split, image_path, declared_severity, and linked_image_id.")

    text_train = _read_unique(data_dir / "text_agent_train_predictions.csv", "claim_id", string_key=True)
    text_validation = _read_unique(data_dir / "text_agent_validation_predictions.csv", "claim_id", string_key=True)
    tabular_train = _read_unique(data_dir / "tabular_oof_predictions.csv", "claim_id", string_key=True)
    tabular_validation = _read_unique(data_dir / "tabular_validation_predictions.csv", "claim_id", string_key=True)
    vision = _read_unique(vision_path, "image_path")
    if "claim_id" not in vision:
        raise ValueError("Vision predictions must contain their image ID in `claim_id`.")
    vision = vision.rename(columns={"claim_id": "claim_id_vision", "model_version": "model_version_vision"})
    if "model_version_vision" not in vision:
        raise ValueError("Vision predictions must include `model_version`.")

    if "fold" not in text_train or "fold" not in tabular_train:
        raise ValueError("Training specialist predictions must carry their fold assignment.")
    text_folds = pd.to_numeric(text_train["fold"], errors="raise").astype(int)
    tabular_folds = pd.to_numeric(tabular_train["fold"], errors="raise").astype(int)
    expected_folds = folds.set_index("claim_id").loc[text_train["claim_id"], "fold"].reset_index(drop=True)
    if not np.array_equal(text_folds.to_numpy(), expected_folds.to_numpy()):
        raise ValueError("Text training prediction folds do not match claim_folds.csv.")
    tabular_fold_lookup = tabular_train.set_index("claim_id")["fold"].astype(int)
    expected_tabular_folds = folds.set_index("claim_id").loc[tabular_train["claim_id"], "fold"].reset_index(drop=True)
    if not np.array_equal(tabular_folds.to_numpy(), expected_tabular_folds.to_numpy()):
        raise ValueError("Tabular OOF prediction folds do not match claim_folds.csv.")
    if set(text_train["claim_id"]) != set(tabular_fold_lookup.index):
        raise ValueError("Text and Tabular training predictions cover different claims.")

    train = _build_split(
        "train",
        data_dir / "claims_train.csv",
        linked,
        text_train,
        tabular_train,
        vision,
        folds,
        data_dir / "decision_agent_train.csv",
    )
    validation = _build_split(
        "validation",
        data_dir / "claims_validation.csv",
        linked,
        text_validation,
        tabular_validation,
        vision,
        folds,
        data_dir / "decision_agent_validation.csv",
    )
    return train, validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/claims_fraud_detection"))
    parser.add_argument("--linked-csv", type=Path, default=Path("data/processed/linked_claims.csv"))
    parser.add_argument("--vision-predictions", type=Path, default=Path("data/processed/vision_predictions.csv"))
    args = parser.parse_args()
    train, validation = prepare_decision_agent_data(args.data_dir, args.linked_csv, args.vision_predictions)
    print(f"Training folds: {train['fold'].value_counts().sort_index().to_dict()}")
    print(f"Validation labels: {validation['fraud_reported'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
