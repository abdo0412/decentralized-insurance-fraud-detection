"""Step 1 (Part A): Deduplicated, Leak-Free Image Dataset Split & K-Fold CV Setup.

Scans `data/raw/vehicle_damage`, removes exact duplicate images via MD5 hashing
so zero duplicate leakage can occur across splits, and creates:
1. `data/processed/held_out_images.csv` (default 15% = 243 images):
   Untouched by the CNN; handed to Person C for linking photos to claims.
2. `data/processed/cnn_splits/test.csv` (12% of CNN pool = 166 images):
   Untouched evaluation set used once at the end to report Vision metrics.
3. `data/processed/cnn_splits/cv_folds.csv` (remaining 1,211 images with 4 stratified folds):
   Used to train the K-Fold CNN Ensemble (+52% more unique training images than v1),
   plus `train.csv` and `val.csv` (Fold 0 split) for single-model compatibility.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import List, Set, Tuple

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.base import SEVERITY_TO_INDEX, normalize_severity_label

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan_and_deduplicate_images(raw_dir: Path) -> Tuple[List[Tuple[str, str, int]], int]:
    """Scan `raw_dir` recursively, verify images, and drop exact MD5 duplicates."""
    records: List[Tuple[str, str, int]] = []
    seen_hashes: Set[str] = set()
    duplicates_dropped = 0

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw image directory not found: {raw_dir}")

    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parent_name = path.parent.name
        try:
            severity = normalize_severity_label(parent_name)
        except ValueError:
            continue

        try:
            raw_bytes = path.read_bytes()
            img_hash = hashlib.md5(raw_bytes).hexdigest()
            if img_hash in seen_hashes:
                duplicates_dropped += 1
                continue
            with Image.open(path) as img:
                img.verify()
        except Exception:
            print(f"[WARN] Skipping corrupted image: {path}")
            continue

        seen_hashes.add(img_hash)
        try:
            rel_posix = path.relative_to(ROOT_DIR).as_posix()
        except ValueError:
            rel_posix = path.as_posix()
        records.append((rel_posix, severity, SEVERITY_TO_INDEX[severity]))

    return records, duplicates_dropped


def create_synthetic_sample_images(raw_dir: Path, per_class: int = 30) -> None:
    """Create a small synthetic image dataset for testing before Kaggle download."""
    import numpy as np

    colors = {
        "01-minor": (60, 160, 80),
        "02-moderate": (220, 160, 40),
        "03-severe": (200, 50, 50),
    }
    rng = np.random.default_rng(42)
    for folder, base_rgb in colors.items():
        folder_path = raw_dir / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            arr = rng.integers(0, 40, size=(224, 224, 3), dtype=np.uint8)
            arr[:, :, 0] = np.clip(arr[:, :, 0] + base_rgb[0], 0, 255)
            arr[:, :, 1] = np.clip(arr[:, :, 1] + base_rgb[1], 0, 255)
            arr[:, :, 2] = np.clip(arr[:, :, 2] + base_rgb[2], 0, 255)
            Image.fromarray(arr).save(folder_path / f"sample_{i:03d}.jpg")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deduplicated, leak-free Vision splits (Part A v2).")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/vehicle_damage"),
        help="Path to raw Kaggle Vehicle Damage dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for CSV manifests.",
    )
    parser.add_argument(
        "--held-out-ratio",
        type=float,
        default=0.15,
        help="Fraction of unique images reserved for Person C's claim linking (untouched by CNN).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.12,
        help="Fraction of the CNN pool reserved for untouched final CNN test metrics.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=4,
        help="Number of stratified CV folds inside the CNN training pool.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible stratified splits.",
    )
    parser.add_argument(
        "--create-dummy-if-empty",
        action="store_true",
        help="Generate synthetic sample images if raw-dir does not exist yet.",
    )
    args = parser.parse_args()

    if not args.raw_dir.exists() and args.create_dummy_if_empty:
        print(f"[INFO] Creating synthetic sample dataset at {args.raw_dir}...")
        create_synthetic_sample_images(args.raw_dir, per_class=30)

    records, dups_dropped = scan_and_deduplicate_images(args.raw_dir)
    if not records:
        raise RuntimeError(
            f"No valid images found under {args.raw_dir}. "
            "Ensure subfolders are named e.g. 01-minor, 02-moderate, 03-severe."
        )

    df = pd.DataFrame(records, columns=["image_path", "severity", "label_idx"])
    df.insert(0, "image_id", [f"IMG-{i:05d}" for i in range(len(df))])

    # 1. Disjoint split: CNN Development Pool (85%) vs Held-Out Claims Pool (15%)
    cnn_df, held_out_df = train_test_split(
        df,
        test_size=args.held_out_ratio,
        stratify=df["severity"],
        random_state=args.seed,
    )

    # 2. Hold out untouched CNN Test Set (12% of CNN pool)
    cv_df, test_df = train_test_split(
        cnn_df,
        test_size=args.test_ratio,
        stratify=cnn_df["severity"],
        random_state=args.seed,
    )
    cv_df = cv_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    held_out_df = held_out_df.reset_index(drop=True)

    # 3. Create Stratified K-Fold assignments on cv_df so 100% of cv_df trains the ensemble
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    cv_df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(skf.split(cv_df, cv_df["label_idx"])):
        cv_df.loc[val_idx, "fold"] = fold_idx

    train_fold0_df = cv_df[cv_df["fold"] != 0].drop(columns=["fold"]).reset_index(drop=True)
    val_fold0_df = cv_df[cv_df["fold"] == 0].drop(columns=["fold"]).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = args.output_dir / "cnn_splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    held_out_path = args.output_dir / "held_out_images.csv"
    held_out_df.to_csv(held_out_path, index=False)

    cv_df.to_csv(splits_dir / "cv_folds.csv", index=False)
    train_fold0_df.to_csv(splits_dir / "train.csv", index=False)
    val_fold0_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    print("=== Part A (v2): Deduplicated Leak-Free Split Complete ===")
    print(f"Unique valid images          : {len(df)} (dropped {dups_dropped} exact MD5 duplicates)")
    print(f"Held-out pool (for Person C) : {len(held_out_df)} -> {held_out_path}")
    print(f"CNN {args.num_folds}-Fold CV Pool           : {len(cv_df)} ({len(train_fold0_df)} train / {len(val_fold0_df)} val per fold) -> {splits_dir / 'cv_folds.csv'}")
    print(f"CNN Untouched Test split     : {len(test_df)} -> {splits_dir / 'test.csv'}")
    print("\nSeverity distribution in Held-Out Pool:")
    print(held_out_df["severity"].value_counts())


if __name__ == "__main__":
    main()
