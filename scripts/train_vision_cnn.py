"""Step 2 (Part A): Two-Stage CNN Transfer Learning, Evaluation & Grad-CAM Grid.

Trains ResNet50 (or MobileNetV3-Large) exclusively on the CNN Development Pool:
- Stage 1 (Warmup): Frozen convolutional backbone, train 3-class head (`lr=1e-3`).
- Stage 2 (Fine-Tune): Unfreeze top blocks (`layer3`+`layer4`), train with `lr=1e-4`
  and inverse-frequency class weights.
Evaluates on `cnn_splits/test.csv` and saves:
- `models/vision_<backbone>_v1.pt`
- `report/figures/vision_confusion_matrix.png`
- `report/figures/vision_gradcam_examples.png`
- `report/figures/vision_metrics.json`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.base import SEVERITY_CLASSES
from agents.vision_agent import (
    VisionAgent,
    build_cnn_backbone,
    build_vision_transforms,
    set_backbone_trainable,
)


class DamageManifestDataset(Dataset):
    """PyTorch Dataset backed by a split CSV (`image_path, severity, label_idx`)."""

    def __init__(self, csv_path: Path, transform) -> None:
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        img_path = str(row["image_path"])
        label = int(row["label_idx"])
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img)
        return tensor, label, img_path


def compute_class_weights(train_csv: Path, num_classes: int = 3) -> torch.Tensor:
    """Compute inverse-frequency class weights to handle class imbalance."""
    df = pd.read_csv(train_csv)
    counts = df["label_idx"].value_counts().reindex(range(num_classes), fill_value=1).values
    total = float(counts.sum())
    weights = total / (num_classes * counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, float, float, List[int], List[int]]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.set_grad_enabled(is_train):
        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, targets)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item()) * images.size(0)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    avg_loss = total_loss / max(1, len(all_targets))
    acc = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    return avg_loss, acc, macro_f1, all_targets, all_preds


def save_confusion_matrix(y_true: List[int], y_pred: List[int], out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(SEVERITY_CLASSES))))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=SEVERITY_CLASSES,
        yticklabels=SEVERITY_CLASSES,
        ax=ax,
    )
    ax.set_xlabel("Predicted Severity")
    ax.set_ylabel("True Severity")
    ax.set_title("Vision Agent (CNN) — Test Confusion Matrix")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_gradcam_grid(
    weights_path: Path,
    test_csv: Path,
    backbone_name: str,
    out_path: Path,
    max_samples: int = 3,
) -> None:
    agent = VisionAgent(weights_path=weights_path, backbone_name=backbone_name)
    df = pd.read_csv(test_csv).head(max_samples)
    if df.empty:
        return

    n = len(df)
    fig, axes = plt.subplots(n, 2, figsize=(8, 3.5 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, (_, row) in enumerate(df.iterrows()):
        img_path = str(row["image_path"])
        true_sev = str(row["severity"])
        orig_img = Image.open(img_path).convert("RGB").resize((224, 224))
        overlay_img, _, probs = agent.explain(img_path)
        pred_sev = max(probs, key=probs.get)

        axes[idx, 0].imshow(orig_img)
        axes[idx, 0].set_title(f"Original (True: {true_sev})")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(overlay_img.resize((224, 224)))
        axes[idx, 1].set_title(f"Grad-CAM (Pred: {pred_sev}, conf={probs[pred_sev]:.2f})")
        axes[idx, 1].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train & evaluate Part A Vision CNN.")
    parser.add_argument("--splits-dir", type=Path, default=Path("data/processed/cnn_splits"))
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet50", "mobilenet_v3_large", "efficientnet_b0"],
    )
    parser.add_argument("--epochs-warmup", type=int, default=5)
    parser.add_argument("--epochs-finetune", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-warmup", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--output-weights", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training Vision CNN ({args.backbone}) on device: {device}")

    tfs = build_vision_transforms(224)
    train_ds = DamageManifestDataset(args.splits_dir / "train.csv", tfs["train"])
    val_ds = DamageManifestDataset(args.splits_dir / "val.csv", tfs["eval"])
    test_ds = DamageManifestDataset(args.splits_dir / "test.csv", tfs["eval"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = build_cnn_backbone(
        backbone_name=args.backbone,
        num_classes=len(SEVERITY_CLASSES),
        pretrained=not args.no_pretrained,
    ).to(device)

    class_weights = compute_class_weights(args.splits_dir / "train.csv").to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    weights_path = args.output_weights or Path(f"models/vision_{args.backbone}_v1.pt")
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_f1 = -1.0

    # --- Stage 1: Warmup Head (Backbone Frozen) ---
    if args.epochs_warmup > 0:
        print(f"\n--- Stage 1: Frozen Backbone Warmup ({args.epochs_warmup} epochs) ---")
        set_backbone_trainable(model, args.backbone, stage="warmup")
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr_warmup,
            weight_decay=1e-4,
        )
        for epoch in range(1, args.epochs_warmup + 1):
            tr_loss, tr_acc, tr_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer, device)
            va_loss, va_acc, va_f1, _, _ = run_epoch(model, val_loader, criterion, None, device)
            print(
                f"[Warmup {epoch:02d}/{args.epochs_warmup:02d}] "
                f"train_loss={tr_loss:.4f} train_f1={tr_f1:.4f} | val_loss={va_loss:.4f} val_f1={va_f1:.4f}"
            )
            if va_f1 >= best_val_f1:
                best_val_f1 = va_f1
                torch.save(
                    {
                        "backbone_name": args.backbone,
                        "model_version": f"{args.backbone}-v1",
                        "model_state_dict": model.state_dict(),
                        "best_val_macro_f1": best_val_f1,
                    },
                    weights_path,
                )

    # --- Stage 2: Fine-Tune Top Blocks ---
    if args.epochs_finetune > 0:
        print(f"\n--- Stage 2: Fine-Tuning Top Blocks ({args.epochs_finetune} epochs) ---")
        set_backbone_trainable(model, args.backbone, stage="finetune")
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr_finetune,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_finetune)
        for epoch in range(1, args.epochs_finetune + 1):
            tr_loss, tr_acc, tr_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer, device)
            va_loss, va_acc, va_f1, _, _ = run_epoch(model, val_loader, criterion, None, device)
            scheduler.step()
            print(
                f"[FineTune {epoch:02d}/{args.epochs_finetune:02d}] "
                f"train_loss={tr_loss:.4f} train_f1={tr_f1:.4f} | val_loss={va_loss:.4f} val_f1={va_f1:.4f}"
            )
            if va_f1 >= best_val_f1:
                best_val_f1 = va_f1
                torch.save(
                    {
                        "backbone_name": args.backbone,
                        "model_version": f"{args.backbone}-v1",
                        "model_state_dict": model.state_dict(),
                        "best_val_macro_f1": best_val_f1,
                    },
                    weights_path,
                )

    # --- Final Evaluation on Untouched CNN Test Split ---
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    te_loss, te_acc, te_f1, y_true, y_pred = run_epoch(model, test_loader, criterion, None, device)

    report_dict: Dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(SEVERITY_CLASSES))),
        target_names=SEVERITY_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    figures_dir = Path("report/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    cm_path = figures_dir / "vision_confusion_matrix.png"
    gradcam_path = figures_dir / "vision_gradcam_examples.png"
    metrics_path = figures_dir / "vision_metrics.json"

    save_confusion_matrix(y_true, y_pred, cm_path)
    save_gradcam_grid(weights_path, args.splits_dir / "test.csv", args.backbone, gradcam_path)

    metrics_payload = {
        "backbone": args.backbone,
        "test_accuracy": round(te_acc, 4),
        "test_macro_f1": round(te_f1, 4),
        "best_val_macro_f1": round(best_val_f1, 4),
        "classification_report": report_dict,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("\n=== Test Set Evaluation Complete ===")
    print(f"Test Accuracy : {te_acc:.4f}")
    print(f"Test Macro F1 : {te_f1:.4f}")
    print(f"Saved weights : {weights_path}")
    print(f"Saved figures : {cm_path}, {gradcam_path}")


if __name__ == "__main__":
    main()
