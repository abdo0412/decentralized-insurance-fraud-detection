"""Build the aligned photo pool manifest from vision predictions.

The source prediction file is left unchanged. Ground-truth image severity is
derived from the canonical class folder in each image path, never from the
model's predicted severity.
"""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
FOLDER_TO_SEVERITY = {
    "01-minor": "minor",
    "02-moderate": "moderate",
    "03-severe": "severe",
}
SEVERITY_TO_INDEX = {"minor": 0, "moderate": 1, "severe": 2}


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT_DIR / path


def build_vision_photo_pool_manifest(
    predictions_csv: Path,
    output_csv: Path,
) -> pd.DataFrame:
    """Create a project-owned image manifest from the held-out prediction paths."""
    predictions_path = _resolve_path(predictions_csv)
    output_path = _resolve_path(output_csv)
    if not predictions_path.exists():
        raise FileNotFoundError(f"Vision predictions not found: {predictions_path}")

    predictions = pd.read_csv(predictions_path)
    if "image_path" not in predictions.columns:
        raise ValueError("Vision predictions must include an `image_path` column.")
    image_id_column = "image_id" if "image_id" in predictions.columns else "claim_id"
    if image_id_column not in predictions.columns:
        raise ValueError("Vision predictions must include `image_id` or `claim_id`.")
    if predictions.empty:
        raise ValueError("Vision predictions contain no image rows.")

    paths = predictions["image_path"].astype("string").str.strip().str.replace("\\", "/", regex=False)
    if paths.isna().any() or paths.eq("").any():
        raise ValueError("Vision predictions contain a blank image path.")
    if paths.duplicated().any():
        duplicates = paths[paths.duplicated(keep=False)].unique().tolist()[:5]
        raise ValueError(f"Vision predictions contain duplicate image paths: {duplicates}")

    image_ids = predictions[image_id_column].astype("string").str.strip()
    if image_ids.isna().any() or image_ids.eq("").any() or image_ids.duplicated().any():
        raise ValueError(f"`{image_id_column}` must contain unique, non-empty image IDs.")

    severities = []
    missing_images = []
    for image_path in paths.tolist():
        posix_path = PurePosixPath(image_path)
        severity = FOLDER_TO_SEVERITY.get(posix_path.parent.name)
        if severity is None:
            raise ValueError(
                f"Image path does not contain a recognized severity folder: {image_path}"
            )
        severities.append(severity)
        local_path = Path(image_path)
        if not local_path.is_absolute():
            local_path = ROOT_DIR / local_path
        if not local_path.is_file():
            missing_images.append(image_path)

    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} prediction image files are missing locally; "
            f"examples: {missing_images[:5]}"
        )

    manifest = pd.DataFrame(
        {
            "image_id": image_ids.astype(str),
            "image_path": paths.astype(str),
            "severity": severities,
        }
    )
    manifest["label_idx"] = manifest["severity"].map(SEVERITY_TO_INDEX).astype(int)
    if set(manifest["severity"].unique()) != set(SEVERITY_TO_INDEX):
        raise ValueError("The A image pool must contain minor, moderate, and severe photos.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    print(f"Built C photo pool manifest with {len(manifest)} images -> {output_path}")
    print("Severity counts:")
    print(manifest["severity"].value_counts().reindex(SEVERITY_TO_INDEX).to_string())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("data/processed/vision_predictions.csv"),
        help="Member A's held-out prediction export.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/claims_fraud_detection/vision_photo_pool.csv"),
        help="Aligned photo pool manifest.",
    )
    args = parser.parse_args()
    build_vision_photo_pool_manifest(args.predictions_csv, args.output_csv)


if __name__ == "__main__":
    main()
