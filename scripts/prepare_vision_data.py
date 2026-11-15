"""Step 1 (Part A): Leak-Free Image Dataset Preparation & Two-Pool Split.

Reads a directory of vehicle damage images organized by severity folders
(e.g., `01-minor`, `02-moderate`, `03-severe` or `minor`, `moderate`, `severe`)
and performs the mandatory disjoint split:
1. Held-Out Claims Pool (default 30%): Never touched by the CNN during training.
   Exported to `data/processed/held_out_images.csv` for Person C to link to claims.
2. CNN Development Pool (default 70%): Further split into stratified `train.csv` (70%),
   `val.csv` (15%), and `test.csv` (15%) inside `data/processed/cnn_splits/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Tuple

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# Add repo root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.base import SEVERITY_TO_INDEX, normalize_severity_label

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan_image_folders(raw_dir: Path) -> List[Tuple[str, str, int]]:
    """Scan `raw_dir` recursively for images inside severity-named parent folders."""
    records: List[Tuple[str, str, int]] = []
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

        # Verify image can be opened and is not corrupted
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            print(f"[WARN] Skipping corrupted image: {path}")
            continue

        records.append((str(path.resolve()), severity, SEVERITY_TO_INDEX[severity]))

    return records


def create_synthetic_sample_images(raw_dir: Path, per_class: int = 20) -> None:
    """Create a small synthetic image dataset for testing the pipeline before Kaggle download."""
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
    parser = argparse.ArgumentParser(description="Prepare leak-free Vision splits (Part A).")
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
        default=0.30,
        help="Fraction of images reserved for Person C's claim linking (never seen by CNN).",
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

    records = scan_image_folders(args.raw_dir)
    if not records:
        raise RuntimeError(
            f"No valid images found under {args.raw_dir}. "
            "Ensure subfolders are named e.g. 01-minor, 02-moderate, 03-severe."
        )

    df = pd.DataFrame(records, columns=["image_path", "severity", "label_idx"])
    df.insert(0, "image_id", [f"IMG-{i:05d}" for i in range(len(df))])

    # 1. Disjoint split: CNN Development Pool (70%) vs Held-Out Claims Pool (30%)
    cnn_df, held_out_df = train_test_split(
        df,
        test_size=args.held_out_ratio,
        stratify=df["severity"],
        random_state=args.seed,
    )

    # 2. Split CNN Development Pool into Train (70%), Val (15%), Test (15%)
    train_df, temp_df = train_test_split(
        cnn_df,
        test_size=0.30,
        stratify=cnn_df["severity"],
        random_state=args.seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["severity"],
        random_state=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = args.output_dir / "cnn_splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    held_out_path = args.output_dir / "held_out_images.csv"
    held_out_df.to_csv(held_out_path, index=False)

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    print("=== Part A: Leak-Free Image Split Complete ===")
    print(f"Total valid images scanned : {len(df)}")
    print(f"Held-out pool (for Person C): {len(held_out_df)} -> {held_out_path}")
    print(f"CNN Train split             : {len(train_df)} -> {splits_dir / 'train.csv'}")
    print(f"CNN Val split               : {len(val_df)} -> {splits_dir / 'val.csv'}")
    print(f"CNN Test split              : {len(test_df)} -> {splits_dir / 'test.csv'}")
    print("\nSeverity distribution in Held-Out Pool:")
    print(held_out_df["severity"].value_counts())


if __name__ == "__main__":
    main()
