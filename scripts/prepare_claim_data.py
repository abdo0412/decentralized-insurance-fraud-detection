"""Prepare insurance-claim splits and the claim-to-fold manifest.

This script does not read or rewrite the Vision Agent's image manifests. It
expects the raw Kaggle claim CSV as an input and writes new files beneath
``data/processed/claims_fraud_detection``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


TARGET_ALIASES = ("fraud_reported", "fraud_label", "is_fraud", "fraud")
CLAIM_ID_ALIASES = ("claim_id", "claim_number", "policy_number", "policy_id")
POSITIVE_LABELS = {"1", "y", "yes", "true", "fraud", "fraudulent"}
NEGATIVE_LABELS = {"0", "n", "no", "false", "honest", "legitimate", "normal"}


def _resolve_column(frame: pd.DataFrame, requested: str | None, aliases: tuple[str, ...]) -> str | None:
    columns = {str(column).strip().lower(): str(column) for column in frame.columns}
    if requested:
        match = columns.get(requested.strip().lower())
        if match is None:
            raise ValueError(f"Column {requested!r} not found. Available columns: {list(frame.columns)}")
        return match
    for alias in aliases:
        if alias in columns:
            return columns[alias]
    return None


def _clean_cells(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        values = cleaned[column].astype("string").str.strip()
        cleaned[column] = values.replace({"?": pd.NA, "": pd.NA})
    return cleaned


def _normalize_fraud_labels(values: pd.Series) -> pd.Series:
    def normalize(value: Any) -> Any:
        if pd.isna(value):
            return pd.NA
        if isinstance(value, (bool, np.bool_)):
            return int(value)
        if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
            return int(value)
        key = str(value).strip().lower()
        if key in POSITIVE_LABELS:
            return 1
        if key in NEGATIVE_LABELS:
            return 0
        raise ValueError(f"Unrecognized fraud label {value!r}; expected Y/N or 0/1.")

    return values.map(normalize).astype("Int64")


def _make_claim_ids(frame: pd.DataFrame, source_column: str | None) -> pd.Series:
    if source_column is None:
        base = pd.Series([f"C-{index + 1:06d}" for index in range(len(frame))], index=frame.index)
    else:
        base = frame[source_column].astype("string").str.strip()
        missing = base.isna() | base.eq("")
        if missing.any():
            generated = pd.Series(
                [f"C-{index + 1:06d}" for index in range(len(frame))], index=frame.index
            )
            base = base.where(~missing, generated)

    # A source field such as policy_number can repeat. Keep every claim row and
    # create deterministic unique IDs so downstream joins remain one-to-one.
    duplicate_ordinals = base.groupby(base, dropna=False).cumcount()
    duplicate_counts = base.value_counts(dropna=False)
    duplicated = base.map(duplicate_counts).gt(1)
    ids = base.astype(str)
    ids.loc[duplicated] = ids.loc[duplicated] + "-" + (duplicate_ordinals[duplicated] + 1).astype(str)
    if ids.duplicated().any():
        ids = pd.Series([f"C-{index + 1:06d}" for index in range(len(frame))], index=frame.index)
    return ids.astype(str)


def prepare_claim_data(
    input_csv: Path,
    output_dir: Path,
    target_column: str | None = None,
    claim_id_column: str | None = None,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    folds: int = 5,
    seed: int = 42,
) -> dict[str, int]:
    """Clean a claims table, make stratified splits, and assign training folds."""
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than 1.")
    if folds < 2:
        raise ValueError("At least two folds are required.")
    if not input_csv.exists():
        raise FileNotFoundError(f"Claims CSV not found: {input_csv}")

    raw = pd.read_csv(input_csv, low_memory=False)
    if raw.empty:
        raise ValueError(f"Claims CSV is empty: {input_csv}")
    cleaned = _clean_cells(raw).drop_duplicates().reset_index(drop=True)
    source_target = _resolve_column(cleaned, target_column, TARGET_ALIASES)
    if source_target is None:
        raise ValueError(
            "Could not find a fraud target column. Expected one of "
            f"{list(TARGET_ALIASES)}; pass --target-column if it has another name."
        )
    source_claim_id = _resolve_column(cleaned, claim_id_column, CLAIM_ID_ALIASES)

    labels = _normalize_fraud_labels(cleaned[source_target])
    labeled = labels.notna()
    dropped_unlabeled = int((~labeled).sum())
    cleaned = cleaned.loc[labeled].copy().reset_index(drop=True)
    labels = labels.loc[labeled].astype(int).reset_index(drop=True)
    if labels.nunique() != 2:
        raise ValueError("The claims CSV must contain both honest and fraudulent examples.")
    if int(labels.value_counts().min()) < 3:
        raise ValueError("Each fraud class needs at least three rows for stratified train/val/test splits.")

    claim_ids = _make_claim_ids(cleaned, source_claim_id)
    cleaned["claim_id"] = claim_ids
    cleaned["fraud_reported"] = labels
    if source_target != "fraud_reported":
        cleaned = cleaned.drop(columns=[source_target])

    test_size = test_fraction
    train_validation, test = train_test_split(
        cleaned,
        test_size=test_size,
        stratify=cleaned["fraud_reported"],
        random_state=seed,
    )
    relative_validation_size = validation_fraction / (1.0 - test_fraction)
    train, validation = train_test_split(
        train_validation,
        test_size=relative_validation_size,
        stratify=train_validation["fraud_reported"],
        random_state=seed,
    )
    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)

    if int(train["fraud_reported"].value_counts().min()) < folds:
        raise ValueError(
            f"The training partition has fewer than {folds} examples in its minority class; "
            "reduce --folds or use a larger claims dataset."
        )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_ids = np.full(len(train), -1, dtype=np.int64)
    for fold_index, (_, validation_indices) in enumerate(
        splitter.split(train, train["fraud_reported"])
    ):
        fold_ids[validation_indices] = fold_index
    fold_manifest = pd.DataFrame({"claim_id": train["claim_id"], "fold": fold_ids})

    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_dir / "claims_clean.csv", index=False)
    train.assign(split="train").to_csv(output_dir / "claims_train.csv", index=False)
    validation.assign(split="validation").to_csv(output_dir / "claims_validation.csv", index=False)
    test.assign(split="test").to_csv(output_dir / "claims_test.csv", index=False)
    fold_manifest.to_csv(output_dir / "claim_folds.csv", index=False)

    counts = {
        "clean": len(cleaned),
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
        "folds": folds,
        "unlabeled_rows_dropped": dropped_unlabeled,
    }
    print("Insurance claim data preparation complete")
    print(f"Rows: {counts}")
    print(f"Outputs: {output_dir.resolve()}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True, help="Raw insurance claim table.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/claims_fraud_detection"),
        help="Processed claims output directory; does not overwrite shared image manifests.",
    )
    parser.add_argument("--target-column", default=None, help="Fraud target column; auto-detected by default.")
    parser.add_argument("--claim-id-column", default=None, help="Claim ID source; auto-detected by default.")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_claim_data(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        target_column=args.target_column,
        claim_id_column=args.claim_id_column,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        folds=args.folds,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
