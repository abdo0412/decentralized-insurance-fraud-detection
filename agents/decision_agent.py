"""Part C: Final decision model and cost-based audit threshold.

Training requires out-of-fold specialist predictions plus claim features. The
agent retains a transparent weighted fallback until that training data exists.
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
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from agents.base import AgentMessage, BaseAgent, compute_severity_mismatch


_EXCLUDED_FEATURE_PARTS = (
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
    "model_version",
)
_POSITIVE_LABELS = {"1", "y", "yes", "true", "fraud", "fraudulent"}
_NEGATIVE_LABELS = {"0", "n", "no", "false", "honest", "legitimate", "normal"}


def _binary_labels(labels: Iterable[Any]) -> np.ndarray:
    normalized = []
    for value in labels:
        if pd.isna(value):
            raise ValueError("Training labels cannot be missing.")
        if isinstance(value, (bool, np.bool_)):
            normalized.append(int(value))
        elif isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
            normalized.append(int(value))
        else:
            key = str(value).strip().lower()
            if key in _POSITIVE_LABELS:
                normalized.append(1)
            elif key in _NEGATIVE_LABELS:
                normalized.append(0)
            else:
                raise ValueError(f"Unrecognized fraud label: {value!r}; expected 0/1 or Y/N.")
    result = np.asarray(normalized, dtype=np.int64)
    if set(np.unique(result)) != {0, 1}:
        raise ValueError("DecisionAgent.fit requires both honest (0) and fraudulent (1) examples.")
    return result


def _allowed_feature(name: Any) -> bool:
    key = str(name).strip().lower()
    return bool(key) and not any(part in key for part in _EXCLUDED_FEATURE_PARTS)


class DecisionAgent(BaseAgent):
    """Fuse out-of-fold agent scores and claim fields into validate/audit."""

    agent_name = "decision"
    model_version = "rf-decision-untrained-v1"

    def __init__(
        self,
        audit_threshold: float = 0.45,
        false_negative_cost: float = 5.0,
        false_positive_cost: float = 1.0,
        model: Pipeline | None = None,
    ) -> None:
        if not 0.0 <= audit_threshold <= 1.0:
            raise ValueError("audit_threshold must be between 0 and 1, inclusive.")
        self.audit_threshold = float(audit_threshold)
        self.false_negative_cost = float(false_negative_cost)
        self.false_positive_cost = float(false_positive_cost)
        self.model = model
        self.feature_columns: list[str] = []
        self.last_explanation_method = "unavailable"
        if model is not None:
            self.feature_columns = list(model.feature_names_in_)
            self.model_version = "rf-decision-v1"

    @staticmethod
    def _prepare_training_frame(
        frame: pd.DataFrame, drop_high_cardinality: bool = True
    ) -> pd.DataFrame:
        cleaned = frame.copy()
        cleaned.columns = [str(column) for column in cleaned.columns]
        cleaned = cleaned.loc[:, [_allowed_feature(column) for column in cleaned.columns]]
        for column in cleaned.select_dtypes(include=["object", "string"]).columns:
            values = cleaned[column].astype("string").str.strip()
            values = values.replace({"?": pd.NA, "": pd.NA})
            numeric = pd.to_numeric(values, errors="coerce")
            if values.notna().any() and numeric[values.notna()].notna().all():
                cleaned[column] = numeric.astype("float64")
                continue
            categorical = values.astype(object)
            cleaned[column] = categorical.where(pd.notna(categorical), np.nan)
            distinct = cleaned[column].nunique(dropna=True)
            if drop_high_cardinality and distinct > 100 and distinct / max(1, len(cleaned)) > 0.5:
                cleaned = cleaned.drop(columns=[column])
        cleaned = cleaned.dropna(axis=1, how="all")
        if cleaned.empty:
            raise ValueError("No usable features were supplied to DecisionAgent.fit.")
        return cleaned

    @staticmethod
    def _build_pipeline(frame: pd.DataFrame) -> Pipeline:
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
                        [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
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
            raise ValueError("No usable features were supplied to DecisionAgent.fit.")
        return Pipeline(
            [
                (
                    "preprocessor",
                    ColumnTransformer(transformers=transformers, remainder="drop"),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=500,
                        class_weight="balanced_subsample",
                        min_samples_leaf=2,
                        max_features="sqrt",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    @staticmethod
    def build_feature_row(
        vision_msg: AgentMessage | Dict[str, Any],
        text_msg: AgentMessage | Dict[str, Any],
        tabular_msg: AgentMessage | Dict[str, Any] | None,
        claim_fields: Dict[str, Any] | None,
        mismatch: Dict[str, float],
    ) -> Dict[str, Any]:
        v_out = vision_msg.output if isinstance(vision_msg, AgentMessage) else vision_msg["output"]
        t_out = text_msg.output if isinstance(text_msg, AgentMessage) else text_msg["output"]
        if isinstance(tabular_msg, AgentMessage):
            tab_out = tabular_msg.output
        elif tabular_msg:
            tab_out = tabular_msg["output"]
        else:
            tab_out = {}

        probabilities = v_out.get("probabilities", {})
        row: Dict[str, Any] = {
            "vision_prob_minor": float(probabilities.get("minor", 1.0 / 3.0)),
            "vision_prob_moderate": float(probabilities.get("moderate", 1.0 / 3.0)),
            "vision_prob_severe": float(probabilities.get("severe", 1.0 / 3.0)),
            "severity_gap_discrete": float(mismatch["severity_gap_discrete"]),
            "severity_gap_continuous": float(mismatch["severity_gap_continuous"]),
            "positive_severity_gap": max(0.0, float(mismatch["severity_gap_continuous"])) / 2.0,
            "text_suspicion_score": float(t_out.get("suspicion_score", 0.2)),
            "tabular_risk_score": float(tab_out.get("risk_score", 0.2)),
        }
        for key, value in (claim_fields or {}).items():
            if _allowed_feature(key):
                row[f"claim_{key}"] = value
        return row

    def fit(self, feature_table: pd.DataFrame, labels: Iterable[Any]) -> "DecisionAgent":
        """Fit on training rows containing only out-of-fold specialist outputs.

        Validation/test rows and in-sample specialist predictions must not be
        passed here. The caller owns that split discipline.
        """
        frame = self._prepare_training_frame(feature_table)
        self.feature_columns = list(frame.columns)
        self.model = self._build_pipeline(frame)
        self.model.fit(frame, _binary_labels(labels))
        self.model_version = "rf-decision-v1"
        return self

    def _probabilities(self, feature_frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("DecisionAgent has no trained Random Forest.")
        row = self._prepare_training_frame(feature_frame, drop_high_cardinality=False)
        row = row.reindex(columns=self.feature_columns)
        probabilities = self.model.predict_proba(row)[0]
        classes = list(self.model.named_steps["classifier"].classes_)
        return probabilities[classes.index(1)]

    def tune_threshold(
        self,
        validation_features: pd.DataFrame,
        validation_labels: Iterable[Any],
        false_negative_cost: float | None = None,
        false_positive_cost: float | None = None,
    ) -> Dict[str, float]:
        """Choose the threshold minimizing validation-set FN/FP cost."""
        if self.model is None:
            raise RuntimeError("Fit DecisionAgent before tuning its threshold.")
        fn_cost = self.false_negative_cost if false_negative_cost is None else float(false_negative_cost)
        fp_cost = self.false_positive_cost if false_positive_cost is None else float(false_positive_cost)
        if fn_cost <= 0.0 or fp_cost <= 0.0:
            raise ValueError("False-negative and false-positive costs must be positive.")

        frame = self._prepare_training_frame(
            validation_features, drop_high_cardinality=False
        ).reindex(columns=self.feature_columns)
        labels = _binary_labels(validation_labels)
        classifier = self.model.named_steps["classifier"]
        classes = list(classifier.classes_)
        fraud_index = classes.index(1)
        scores = self.model.predict_proba(frame)[:, fraud_index]
        candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))

        best = None
        for threshold in candidates:
            predictions = (scores >= threshold).astype(np.int64)
            false_negatives = int(np.sum((labels == 1) & (predictions == 0)))
            false_positives = int(np.sum((labels == 0) & (predictions == 1)))
            cost = fn_cost * false_negatives + fp_cost * false_positives
            candidate = (float(cost), -float(threshold), int(false_negatives), int(false_positives))
            if best is None or candidate < best:
                best = candidate

        assert best is not None
        self.false_negative_cost = fn_cost
        self.false_positive_cost = fp_cost
        self.audit_threshold = -best[1]
        return {
            "threshold": self.audit_threshold,
            "validation_cost": best[0],
            "false_negatives": float(best[2]),
            "false_positives": float(best[3]),
            "false_negative_cost": fn_cost,
            "false_positive_cost": fp_cost,
        }

    def compare_cost_ratios(
        self,
        validation_features: pd.DataFrame,
        validation_labels: Iterable[Any],
        false_negative_to_positive_ratios: Iterable[float] = (1.0, 3.0, 5.0),
        false_positive_cost: float = 1.0,
    ) -> list[Dict[str, float]]:
        """Tune and report thresholds for several missed-fraud cost ratios."""
        if self.model is None:
            raise RuntimeError("Fit DecisionAgent before comparing cost ratios.")
        ratios = [float(ratio) for ratio in false_negative_to_positive_ratios]
        if not ratios or any(ratio <= 0.0 for ratio in ratios):
            raise ValueError("Provide one or more positive false-negative/false-positive cost ratios.")
        previous = (
            self.audit_threshold,
            self.false_negative_cost,
            self.false_positive_cost,
        )
        normalized_labels = _binary_labels(validation_labels)
        results = []
        try:
            for ratio in ratios:
                result = self.tune_threshold(
                    validation_features=validation_features,
                    validation_labels=normalized_labels,
                    false_negative_cost=ratio * false_positive_cost,
                    false_positive_cost=false_positive_cost,
                )
                result["false_negative_to_positive_ratio"] = ratio
                results.append(result)
        finally:
            self.audit_threshold, self.false_negative_cost, self.false_positive_cost = previous
        return results

    def evaluate(
        self,
        features: pd.DataFrame,
        labels: Iterable[Any],
        threshold: float | None = None,
    ) -> Dict[str, Any]:
        """Return thresholded metrics for one explicitly supplied evaluation split."""
        if self.model is None:
            raise RuntimeError("Fit DecisionAgent before evaluating it.")
        frame = self._prepare_training_frame(features, drop_high_cardinality=False).reindex(
            columns=self.feature_columns
        )
        actual = _binary_labels(labels)
        classifier = self.model.named_steps["classifier"]
        classes = list(classifier.classes_)
        fraud_index = classes.index(1)
        scores = self.model.predict_proba(frame)[:, fraud_index]
        chosen_threshold = self.audit_threshold if threshold is None else float(threshold)
        predicted = (scores >= chosen_threshold).astype(np.int64)
        return {
            "threshold": chosen_threshold,
            "recall": float(recall_score(actual, predicted, zero_division=0)),
            "precision": float(precision_score(actual, predicted, zero_division=0)),
            "f1": float(f1_score(actual, predicted, zero_division=0)),
            "pr_auc": float(average_precision_score(actual, scores)),
            "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
        }

    def _encoded_feature_names(self) -> list[str]:
        if self.model is None:
            return []
        preprocessor = self.model.named_steps["preprocessor"]
        try:
            return [str(name) for name in preprocessor.get_feature_names_out()]
        except (AttributeError, ValueError):
            return list(self.feature_columns)

    def explain(self, feature_row: Dict[str, Any], max_features: int = 6) -> Dict[str, float]:
        """Return per-claim SHAP contributions when available, else RF importances."""
        if self.model is None:
            return {}
        frame = self._prepare_training_frame(
            pd.DataFrame([feature_row]), drop_high_cardinality=False
        ).reindex(
            columns=self.feature_columns
        )
        names = self._encoded_feature_names()
        preprocessor = self.model.named_steps["preprocessor"]
        classifier = self.model.named_steps["classifier"]
        try:
            import shap

            transformed = preprocessor.transform(frame)
            explainer = shap.TreeExplainer(classifier)
            raw_values = explainer.shap_values(transformed)
            classes = list(classifier.classes_)
            positive_index = classes.index(1)
            if isinstance(raw_values, list):
                values = np.asarray(raw_values[positive_index]).reshape(-1)
            else:
                array = np.asarray(raw_values)
                if array.ndim == 3:
                    if array.shape[-1] == len(classes):
                        values = array[0, :, positive_index]
                    else:
                        values = array[positive_index, 0, :]
                else:
                    values = array.reshape(-1)
            contributions = {names[i]: float(values[i]) for i in range(min(len(names), len(values)))}
            ranked = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
            self.last_explanation_method = "shap"
            return {name: round(value, 4) for name, value in ranked[:max_features]}
        except Exception:
            # SHAP is an optional explanation layer; retain a model-level fallback.
            importances = np.asarray(classifier.feature_importances_)
            ranked = sorted(
                ((names[i], float(importances[i])) for i in range(min(len(names), len(importances)))),
                key=lambda item: item[1],
                reverse=True,
            )
            self.last_explanation_method = "global_feature_importance"
            return {name: round(value, 4) for name, value in ranked[:max_features]}

    def save(self, path: str | Path) -> None:
        """Persist the trained agent with joblib."""
        if self.model is None:
            raise RuntimeError("Cannot save DecisionAgent before fitting it.")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)

    @classmethod
    def load(cls, path: str | Path) -> "DecisionAgent":
        loaded = joblib.load(Path(path))
        if not isinstance(loaded, cls):
            raise TypeError(f"Saved object at {path} is not a DecisionAgent.")
        return loaded

    def predict(
        self,
        claim_id: str,
        vision_msg: AgentMessage | Dict[str, Any],
        text_msg: AgentMessage | Dict[str, Any],
        tabular_msg: AgentMessage | Dict[str, Any] | None = None,
        claim_fields: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentMessage:
        v_out = vision_msg.output if isinstance(vision_msg, AgentMessage) else vision_msg["output"]
        t_out = text_msg.output if isinstance(text_msg, AgentMessage) else text_msg["output"]
        declared_severity = str((claim_fields or {}).get("incident_severity", "minor"))
        vision_probabilities = v_out.get(
            "probabilities", {"minor": 0.34, "moderate": 0.33, "severe": 0.33}
        )
        mismatch = compute_severity_mismatch(declared_severity, vision_probabilities)
        feature_row = self.build_feature_row(
            vision_msg=vision_msg,
            text_msg=text_msg,
            tabular_msg=tabular_msg,
            claim_fields=claim_fields,
            mismatch=mismatch,
        )

        if self.model is not None:
            audit_probability = round(float(self._probabilities(pd.DataFrame([feature_row]))), 4)
            top_factors = self.explain(feature_row)
            model_version = self.model_version
            explanation_method = self.last_explanation_method
        else:
            text_suspicion = float(t_out.get("suspicion_score", 0.2))
            if isinstance(tabular_msg, AgentMessage):
                tab_out = tabular_msg.output
            elif tabular_msg:
                tab_out = tabular_msg["output"]
            else:
                tab_out = {"risk_score": 0.25}
            tabular_risk = float(tab_out.get("risk_score", 0.2))
            positive_gap = max(0.0, float(mismatch["severity_gap_continuous"])) / 2.0
            audit_probability = round(
                min(0.99, max(0.01, 0.45 * positive_gap + 0.35 * text_suspicion + 0.20 * tabular_risk)),
                4,
            )
            top_factors = {
                "severity_mismatch": round(0.45 * positive_gap, 4),
                "text_suspicion_score": round(0.35 * text_suspicion, 4),
                "tabular_risk_score": round(0.20 * tabular_risk, 4),
            }
            model_version = "weighted-fallback-v1"
            explanation_method = "weighted_fallback_components"

        decision_label = "audit" if audit_probability >= self.audit_threshold else "validate"
        reasons = []
        if mismatch["severity_gap_discrete"] >= 1:
            reasons.append(
                f"declared severity is {declared_severity} but photo shows {v_out.get('severity')} damage"
            )
        if float(t_out.get("suspicion_score", 0.2)) >= 0.5:
            reasons.append("report wording scores as suspicious")
        if float(feature_row.get("tabular_risk_score", 0.2)) >= 0.5:
            reasons.append("claim metadata indicates elevated risk")
        if not reasons and top_factors and explanation_method in (
            "shap",
            "weighted_fallback_components",
        ):
            strongest = next(iter(top_factors))
            reasons.append(f"the strongest contributing factor is {strongest.replace('_', ' ')}")
        if not reasons and top_factors:
            reasons.append("the model's learned claim patterns support this decision")
        if not reasons:
            reasons.append("photo severity, narrative, and claim metadata are consistent")

        comparator = ">=" if decision_label == "audit" else "<"
        prefix = (
            f"Audit (fraud probability {audit_probability:.2f} {comparator} threshold {self.audit_threshold:.2f})"
            if decision_label == "audit"
            else f"Validate (fraud probability {audit_probability:.2f} {comparator} threshold {self.audit_threshold:.2f})"
        )
        reason_sentence = f"{prefix}: {', and '.join(reasons)}."
        return AgentMessage(
            claim_id=str(claim_id),
            agent=self.agent_name,
            output={
                "decision": decision_label,
                "audit_probability": audit_probability,
                "threshold": self.audit_threshold,
                "severity_mismatch": mismatch,
                "top_factors": top_factors,
                "explanation_method": explanation_method,
                "reason": reason_sentence,
                "false_negative_cost": self.false_negative_cost,
                "false_positive_cost": self.false_positive_cost,
            },
            confidence=round(max(audit_probability, 1.0 - audit_probability), 4),
            model_version=model_version,
        )
