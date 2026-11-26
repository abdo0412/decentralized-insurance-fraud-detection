"""Create the documented synthetic claim-to-held-out-photo linkage.

This consumes C's photo-pool manifest built from A's held-out predictions,
without modifying A's files. Photo pools are partitioned across
train/validation/test before claims are linked, so a photo path is never shared
across those claim splits. Sampling with replacement is allowed inside one
split when there are more claims than held-out photos.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.base import SEVERITY_CLASSES, SEVERITY_TO_INDEX, normalize_severity_label


def _partition_photos(
    photos: pd.DataFrame,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Stratify the held-out photo pool and keep each path in one claim split."""
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than 1.")
    if photos["severity"].value_counts().min() < 3:
        raise ValueError("The held-out image pool needs at least three photos per severity class.")
    train_validation, test = train_test_split(
        photos,
        test_size=test_fraction,
        stratify=photos["severity"],
        random_state=seed,
    )
    validation_relative = validation_fraction / (1.0 - test_fraction)
    train, validation = train_test_split(
        train_validation,
        test_size=validation_relative,
        stratify=train_validation["severity"],
        random_state=seed,
    )
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def _desired_photo_class(
    declared: str,
    rng: np.random.Generator,
    severity_noise_probability: float,
) -> str:
    """Choose a photo severity without access to the claim's fraud label."""
    rank = SEVERITY_TO_INDEX[declared]
    if rng.random() < severity_noise_probability:
        neighbors = [index for index in (rank - 1, rank + 1) if 0 <= index < len(SEVERITY_CLASSES)]
        return SEVERITY_CLASSES[int(rng.choice(neighbors))]
    return declared


def link_claims_to_photos(
    claims_dir: Path,
    held_out_csv: Path,
    output_csv: Path,
    declared_severity_column: str = "incident_severity",
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    severity_noise_probability: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Link C claims to A's held-out images with label-independent severity sampling."""
    if not 0.0 <= severity_noise_probability <= 1.0:
        raise ValueError("severity_noise_probability must be between 0 and 1.")
    if not held_out_csv.exists():
        raise FileNotFoundError(f"Held-out image manifest not found: {held_out_csv}")

    photos = pd.read_csv(held_out_csv)
    if not {"image_path", "severity"}.issubset(photos.columns):
        raise ValueError("A's held-out manifest must have `image_path` and `severity` columns.")
    photos = photos.dropna(subset=["image_path", "severity"]).drop_duplicates("image_path").copy()
    photos["severity"] = photos["severity"].map(normalize_severity_label)
    photos = photos.reset_index(drop=True)
    image_id_column = "image_id" if "image_id" in photos.columns else None
    pools = _partition_photos(photos, validation_fraction, test_fraction, seed)

    split_files = {
        "train": claims_dir / "claims_train.csv",
        "validation": claims_dir / "claims_validation.csv",
        "test": claims_dir / "claims_test.csv",
    }
    missing = [str(path) for path in split_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Claim splits are missing: " + ", ".join(missing))

    rng = np.random.default_rng(seed)
    linked_splits = []
    for split_name, claims_path in split_files.items():
        claims = pd.read_csv(claims_path)
        required = {"claim_id", declared_severity_column}
        if not required.issubset(claims.columns):
            raise ValueError(
                f"{claims_path} needs columns {sorted(required)}; found {list(claims.columns)}."
            )
        pool = pools[split_name]
        by_severity = {label: pool[pool["severity"] == label] for label in SEVERITY_CLASSES}
        if any(group.empty for group in by_severity.values()):
            raise ValueError(
                f"The {split_name} photo partition has an empty severity class; "
                "use a larger held-out pool or adjust split fractions."
            )

        selected_paths = []
        selected_ids = []
        selected_severities = []
        mismatches = []
        for row in claims.to_dict(orient="records"):
            raw_declared = row[declared_severity_column]
            declared = normalize_severity_label(str(raw_declared))
            desired = _desired_photo_class(
                declared=declared,
                rng=rng,
                severity_noise_probability=severity_noise_probability,
            )
            candidates = by_severity[desired]
            photo_index = int(rng.integers(0, len(candidates)))
            photo = candidates.iloc[photo_index]
            selected_paths.append(str(photo["image_path"]))
            selected_ids.append(str(photo[image_id_column]) if image_id_column else "")
            selected_severities.append(str(photo["severity"]))
            mismatches.append(int(SEVERITY_TO_INDEX[declared] > SEVERITY_TO_INDEX[str(photo["severity"])]))

        claims = claims.copy()
        claims["image_path"] = selected_paths
        if image_id_column:
            claims["linked_image_id"] = selected_ids
        claims["declared_severity"] = claims[declared_severity_column].astype(str)
        claims["linked_photo_severity"] = selected_severities
        claims["synthetic_severity_mismatch"] = mismatches
        claims["split"] = split_name
        linked_splits.append(claims)

    linked = pd.concat(linked_splits, ignore_index=True)
    if linked["claim_id"].astype(str).duplicated().any():
        raise ValueError("Claim IDs must be unique across train, validation, and test files.")
    path_sets = {
        split_name: set(linked.loc[linked["split"] == split_name, "image_path"].astype(str))
        for split_name in split_files
    }
    split_names = list(path_sets)
    for left_index, left_name in enumerate(split_names):
        for right_name in split_names[left_index + 1 :]:
            if path_sets[left_name].intersection(path_sets[right_name]):
                raise RuntimeError(
                    f"A held-out image path appeared in both {left_name} and {right_name} splits."
                )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    linked.to_csv(output_csv, index=False)
    print(f"Linked {len(linked)} claims to held-out photos -> {output_csv.resolve()}")
    print("Photo paths are partitioned across train, validation, and test claim splits.")
    print(
        "Photo severity uses declared claim severity plus a uniform, label-independent "
        f"neighbor-noise probability of {severity_noise_probability:.3f}."
    )
    print("The severity-based claim/photo relationship is synthetic and must be disclosed in the report.")
    return linked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--claims-dir",
        type=Path,
        default=Path("data/processed/claims_fraud_detection"),
    )
    parser.add_argument(
        "--held-out-csv",
        type=Path,
        default=Path("data/processed/claims_fraud_detection/vision_photo_pool.csv"),
        help="Aligned photo manifest built from the vision prediction paths.",
    )
    parser.add_argument("--output-csv", type=Path, default=Path("data/processed/linked_claims.csv"))
    parser.add_argument("--declared-severity-column", default="incident_severity")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument(
        "--severity-noise-probability",
        type=float,
        default=0.05,
        help="Probability of selecting an adjacent photo severity, independent of fraud labels.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    link_claims_to_photos(
        claims_dir=args.claims_dir,
        held_out_csv=args.held_out_csv,
        output_csv=args.output_csv,
        declared_severity_column=args.declared_severity_column,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        severity_noise_probability=args.severity_noise_probability,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
