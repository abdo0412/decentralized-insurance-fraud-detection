"""Tune, train, and validate the final Decision Agent.

This script uses only the prepared training and validation feature tables. It
uses C's predefined five training folds for hyperparameter selection, chooses
the audit threshold from validation costs, and never reads the frozen test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.decision_agent import DecisionAgent


def _load_table(path: Path, *, require_fold: bool = False) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required Decision Agent input is missing: {path}")
    frame = pd.read_csv(path, dtype={"claim_id": "string"})
    required = {"claim_id", "fraud_reported", "text_suspicion_score", "tabular_risk_score",
                "vision_prob_minor", "vision_prob_moderate", "vision_prob_severe"}
    if require_fold:
        required.add("fold")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if frame["claim_id"].isna().any() or frame["claim_id"].duplicated().any():
        raise ValueError(f"{path} must contain one non-empty row per claim_id.")
    if frame["fraud_reported"].isna().any():
        raise ValueError(f"{path} contains missing fraud_reported labels.")
    frame["fraud_reported"] = pd.to_numeric(frame["fraud_reported"], errors="raise").astype(int)
    if not set(frame["fraud_reported"].unique()).issubset({0, 1}):
        raise ValueError(f"{path} has fraud_reported values other than 0 and 1.")

    score_columns = [
        "text_suspicion_score",
        "tabular_risk_score",
        "vision_prob_minor",
        "vision_prob_moderate",
        "vision_prob_severe",
    ]
    for column in score_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if frame[column].isna().any() or ((frame[column] < 0) | (frame[column] > 1)).any():
            raise ValueError(f"{path} has missing or out-of-range {column} values.")
    probability_sum = frame[["vision_prob_minor", "vision_prob_moderate", "vision_prob_severe"]].sum(axis=1)
    if ((probability_sum - 1.0).abs() > 0.003).any():
        raise ValueError(f"{path} contains Vision probabilities that do not sum to approximately 1.")
    return frame


def train_decision_agent(
    data_dir: Path,
    model_path: Path,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
) -> dict[str, Any]:
    train_path = data_dir / "decision_agent_train.csv"
    validation_path = data_dir / "decision_agent_validation.csv"
    train = _load_table(train_path, require_fold=True)
    validation = _load_table(validation_path)

    if len(train) != 700 or len(validation) != 150:
        raise ValueError(
            f"Expected 700 training and 150 validation claims; got {len(train)} and {len(validation)}."
        )
    if set(train["claim_id"].astype(str)) & set(validation["claim_id"].astype(str)):
        raise ValueError("Training and validation claim IDs overlap.")
    folds = pd.to_numeric(train["fold"], errors="raise").astype(int)
    fold_values = sorted(folds.unique().tolist())
    if fold_values != [0, 1, 2, 3, 4]:
        raise ValueError(f"Expected five folds numbered 0-4; found {fold_values}.")
    fold_counts = folds.value_counts().sort_index()
    if not (fold_counts == 140).all():
        raise ValueError(f"Expected 140 claims in each training fold; got {fold_counts.to_dict()}.")
    if set(train["fraud_reported"].unique()) != {0, 1} or set(validation["fraud_reported"].unique()) != {0, 1}:
        raise ValueError("Both training and validation splits must contain fraud and legitimate claims.")
    if false_negative_cost <= 0 or false_positive_cost <= 0:
        raise ValueError("False-negative and false-positive costs must be positive.")

    target = "fraud_reported"
    train_labels = train[target].to_numpy(dtype=np.int64)
    validation_labels = validation[target].to_numpy(dtype=np.int64)
    train_features = DecisionAgent._prepare_training_frame(train.drop(columns=[target]))
    validation_features = validation.drop(columns=[target])

    # PredefinedSplit honors C's shared fold assignments. The grid is small and
    # includes the agent's default settings while tuning tree count, leaf size,
    # and the feature sampling strategy by average precision.
    cv = PredefinedSplit(test_fold=folds.to_numpy())
    parameter_grid = {
        "classifier__n_estimators": [300, 500],
        "classifier__min_samples_leaf": [1, 2],
        "classifier__max_features": ["sqrt", 0.7],
    }
    search = GridSearchCV(
        estimator=DecisionAgent._build_pipeline(train_features),
        param_grid=parameter_grid,
        scoring="average_precision",
        cv=cv,
        refit=True,
        # Serial fitting avoids Windows worker-process permission failures in
        # restricted environments while keeping the fold protocol unchanged.
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(train_features, train_labels)

    agent = DecisionAgent(
        model=search.best_estimator_,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    agent.model_version = "rf-decision-cv-v1"
    threshold_info = agent.tune_threshold(
        validation_features=validation_features,
        validation_labels=validation_labels,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    validation_metrics = agent.evaluate(validation_features, validation_labels)

    prepared_validation = DecisionAgent._prepare_training_frame(
        validation_features, drop_high_cardinality=False
    ).reindex(columns=agent.feature_columns)
    classifier = agent.model.named_steps["classifier"]
    positive_index = list(classifier.classes_).index(1)
    probabilities = agent.model.predict_proba(prepared_validation)[:, positive_index]
    predictions = (probabilities >= agent.audit_threshold).astype(np.int64)
    validation_predictions = pd.DataFrame(
        {
            "claim_id": validation["claim_id"].astype(str),
            "fraud_reported": validation_labels,
            "fraud_probability": probabilities,
            "audit_threshold": agent.audit_threshold,
            "predicted_label": np.where(predictions == 1, "audit", "validate"),
            "model_version": agent.model_version,
        }
    )

    metrics: dict[str, Any] = {
        **validation_metrics,
        "validation_cost": threshold_info["validation_cost"],
        "false_negatives_at_threshold": int(threshold_info["false_negatives"]),
        "false_positives_at_threshold": int(threshold_info["false_positives"]),
        "false_negative_cost": false_negative_cost,
        "false_positive_cost": false_positive_cost,
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "training_fold_count": int(len(fold_values)),
        "training_fold_sizes": {str(k): int(v) for k, v in fold_counts.items()},
        "cv_scoring": "average_precision",
        "best_cv_average_precision": float(search.best_score_),
        "best_parameters": search.best_params_,
        "feature_count": int(len(agent.feature_columns)),
        "model_version": agent.model_version,
        "test_set_used": False,
        "note": (
            "Threshold selected on validation using the configured cost ratio. "
            "Specialist scores are precomputed input artifacts; provenance of the "
            "Text Agent training data is external to this repository. "
            "The claim-to-photo association is a synthetic severity-based assumption."
        ),
    }

    data_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(model_path)
    validation_predictions.to_csv(data_dir / "decision_agent_validation_predictions.csv", index=False)
    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(
        data_dir / "decision_agent_cv_results.csv", index=False
    )
    (data_dir / "decision_agent_validation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (data_dir / "decision_agent_threshold.json").write_text(
        json.dumps(threshold_info, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/claims_fraud_detection"),
    )
    parser.add_argument("--model-path", type=Path, default=Path("models/decision_agent.joblib"))
    parser.add_argument("--false-negative-cost", type=float, default=5.0)
    parser.add_argument("--false-positive-cost", type=float, default=1.0)
    args = parser.parse_args()
    result = train_decision_agent(
        data_dir=args.data_dir,
        model_path=args.model_path,
        false_negative_cost=args.false_negative_cost,
        false_positive_cost=args.false_positive_cost,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
