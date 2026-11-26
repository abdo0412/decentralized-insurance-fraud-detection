"""Part C: Tabular risk agent.

The agent can be trained on the structured claim table. Until a trained model is
provided, it keeps a small deterministic fallback so the demo remains usable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from agents.base import AgentMessage, BaseAgent


_EXCLUDED_FIELD_PARTS = (
    "claim_id",
    "policy_number",
    "policy_id",
    "fraud_reported",
    "fraud_label",
    "fraud",
    "is_fraud",
    "fraudulent",
    "image_path",
    "image_id",
    "linked_photo_severity",
    "synthetic_severity_mismatch",
    "narrative",
    "report_text",
    "description",
    "split",
    "fold",
)
_POSITIVE_LABELS = {"1", "y", "yes", "true", "fraud", "fraudulent"}
_NEGATIVE_LABELS = {"0", "n", "no", "false", "honest", "legitimate", "normal"}


def _binary_labels(labels: Iterable[Any]) -> np.ndarray:
    """Convert common Kaggle fraud labels to 0/1 and reject ambiguous values."""
    normalized = []
    for value in labels:
        if pd.isna(value):
            raise ValueError("Training labels cannot be missing.")
        if isinstance(value, (bool, np.bool_)):
            normalized.append(int(value))
            continue
        if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
            normalized.append(int(value))
            continue
        key = str(value).strip().lower()
        if key in _POSITIVE_LABELS:
            normalized.append(1)
        elif key in _NEGATIVE_LABELS:
            normalized.append(0)
        else:
            raise ValueError(f"Unrecognized fraud label: {value!r}; expected 0/1 or Y/N.")
    result = np.asarray(normalized, dtype=np.int64)
    if set(np.unique(result)) != {0, 1}:
        raise ValueError("TabularAgent.fit requires both honest (0) and fraudulent (1) examples.")
    return result


def _is_model_feature(name: Any) -> bool:
    key = str(name).strip().lower()
    return bool(key) and not any(part in key for part in _EXCLUDED_FIELD_PARTS)


class TabularAgent(BaseAgent):
    """Estimate fraud probability from structured claim fields."""

    agent_name = "tabular"
    model_version = "rf-tabular-untrained-v1"

    def __init__(self, model: Pipeline | None = None) -> None:
        self.model = model
        self.feature_columns: list[str] = []
        if model is not None:
            self.feature_columns = list(model.feature_names_in_)
            self.model_version = "rf-tabular-v1"

    @staticmethod
    def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
        cleaned = frame.copy()
        cleaned.columns = [str(column) for column in cleaned.columns]
        cleaned = cleaned.loc[:, [_is_model_feature(column) for column in cleaned.columns]]
        for column in cleaned.select_dtypes(include=["object", "string"]).columns:
            values = cleaned[column].astype("string").str.strip()
            values = values.replace({"?": pd.NA, "": pd.NA})
            numeric = pd.to_numeric(values, errors="coerce")
            if values.notna().any() and numeric[values.notna()].notna().all():
                cleaned[column] = numeric.astype("float64")
            else:
                categorical = values.astype(object)
                cleaned[column] = categorical.where(pd.notna(categorical), np.nan)
        cleaned = cleaned.dropna(axis=1, how="all")
        return cleaned

    @staticmethod
    def _preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
        numeric_columns = [
            column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])
        ]
        categorical_columns = [column for column in frame.columns if column not in numeric_columns]
        transformers = []
        if numeric_columns:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ]
                    ),
                    numeric_columns,
                )
            )
        if categorical_columns:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_columns,
                )
            )
        if not transformers:
            raise ValueError("No usable claim features were supplied to TabularAgent.fit.")
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def fit(self, claim_table: pd.DataFrame, labels: Iterable[Any]) -> "TabularAgent":
        """Fit the tabular Random Forest using training rows only.

        The caller is responsible for passing only the training partition. The
        learned imputation and category encodings are fitted inside this method.
        """
        frame = self._clean_frame(claim_table)
        if frame.empty:
            raise ValueError("No usable claim features were supplied to TabularAgent.fit.")
        self.feature_columns = list(frame.columns)
        preprocessor = self._preprocessor(frame)
        self.model = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced_subsample",
                        min_samples_leaf=2,
                        max_features="sqrt",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        self.model.fit(frame, _binary_labels(labels))
        self.model_version = "rf-tabular-v1"
        return self

    def save(self, path: str | Path) -> None:
        """Persist the fitted agent with joblib."""
        if self.model is None:
            raise RuntimeError("Cannot save TabularAgent before fitting it.")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)

    @classmethod
    def load(cls, path: str | Path) -> "TabularAgent":
        loaded = joblib.load(Path(path))
        if not isinstance(loaded, cls):
            raise TypeError(f"Saved object at {path} is not a TabularAgent.")
        return loaded

    def predict_proba(self, claim_table: pd.DataFrame) -> np.ndarray:
        """Return fraud probabilities for rows, preserving the fitted feature schema."""
        if self.model is None:
            raise RuntimeError("TabularAgent has no trained Random Forest.")
        frame = self._clean_frame(claim_table).reindex(columns=self.feature_columns)
        probabilities = self.model.predict_proba(frame)
        classes = list(self.model.named_steps["classifier"].classes_)
        return probabilities[:, classes.index(1)]

    @staticmethod
    def _fallback_score(fields: Dict[str, Any]) -> float:
        """Small transparent fallback used before a trained model is available."""
        try:
            amount = float(fields.get("total_claim_amount", 5000.0))
        except (TypeError, ValueError):
            amount = 5000.0
        try:
            policy_months = float(fields.get("months_as_customer", 24.0))
        except (TypeError, ValueError):
            policy_months = 24.0
        try:
            witnesses = int(fields.get("witnesses", 1))
        except (TypeError, ValueError):
            witnesses = 1
        police_report = str(fields.get("police_report_available", "YES")).strip().upper()

        risk = 0.20
        if amount > 20000:
            risk += 0.25
        if policy_months < 6:
            risk += 0.20
        if witnesses == 0:
            risk += 0.10
        if police_report in ("NO", "?"):
            risk += 0.15
        return round(min(0.95, max(0.05, risk)), 4)

    def predict(
        self,
        claim_id: str,
        claim_fields: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentMessage:
        """Return the predicted fraud risk in the shared JSON message envelope."""
        fields = claim_fields or {}
        if self.model is not None:
            risk_score = float(self.predict_proba(pd.DataFrame([fields]))[0])
            version = self.model_version
        else:
            risk_score = self._fallback_score(fields)
            version = "heuristic-fallback-v1"

        risk_score = round(float(np.clip(risk_score, 0.0, 1.0)), 4)
        confidence = round(max(risk_score, 1.0 - risk_score), 4)
        return AgentMessage(
            claim_id=str(claim_id),
            agent=self.agent_name,
            output={"risk_score": risk_score},
            confidence=confidence,
            model_version=version,
        )
