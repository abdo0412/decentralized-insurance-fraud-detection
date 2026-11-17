"""Step 2 (Part A v2): High-Accuracy K-Fold CNN Transfer Learning, TTA & Grad-CAM.

Upgrades over v1 to maximize 3-class damage severity accuracy without data leakage:
1. Backbone: `efficientnet_v2_s` (IMAGENET1K_V1) at 288x288 resolution (aspect-preserving).
2. Data utilization: Stratified 4-Fold Cross-Validation across all 1,211 CV images
   (+52% more unique training images than v1's single 798-image split).
3. Regularization for noisy `moderate` labels: Label Smoothing (0.08) + RandAugment +
   Mixup (alpha=0.2) + final 4-epoch clean cooldown.
4. Optimization: Two-stage transfer learning (frozen warmup -> full unfrozen backbone
   with layer-wise discriminative learning rates + CosineAnnealingLR + FP16 AMP).
5. Inference: 4-Fold Ensemble + Horizontal-Flip Test-Time Augmentation (TTA).
"""

from __future__ import annotations

import argparse
import copy
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.base import SEVERITY_CLASSES
from agents.vision_agent import (
    VisionAgent,
    build_cnn_backbone,
    build_optimizer_param_groups,
    build_vision_transforms,
    set_backbone_trainable,
)


class DamageDataFrameDataset(Dataset):
    """PyTorch Dataset backed by an in-memory DataFrame (`image_path, severity, label_idx`)."""

    def __init__(self, df: pd.DataFrame, transform) -> None:
        self.df = df.reset_index(drop=True)
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


def compute_class_weights_from_df(df: pd.DataFrame, num_classes: int = 3) -> torch.Tensor:
    counts = df["label_idx"].value_counts().reindex(range(num_classes), fill_value=1).values
    total = float(counts.sum())
    weights = total / (num_classes * counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    use_amp: bool,
    mixup_alpha: float = 0.0,
    mixup_prob: float = 0.4,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        do_mixup = mixup_alpha > 0.0 and images.size(0) > 1 and np.random.rand() < mixup_prob
        if do_mixup:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            perm = torch.randperm(images.size(0), device=device)
            mixed_images = lam * images + (1.0 - lam) * images[perm]
            targets_b = targets[perm]
        else:
            lam = 1.0
            mixed_images = images
            targets_b = targets

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(mixed_images)
            if do_mixup:
                loss = lam * criterion(logits, targets) + (1.0 - lam) * criterion(logits, targets_b)
            else:
                loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item()) * images.size(0)
        preds = torch.argmax(logits.detach(), dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    avg_loss = total_loss / max(1, len(all_targets))
    acc = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    return avg_loss, acc, macro_f1


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_tta: bool = True,
) -> Tuple[float, float, float, List[int], List[int]]:
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        probs = F.softmax(logits, dim=-1)

        if use_tta:
            logits_flip = model(torch.flip(images, dims=[-1]))
            probs = 0.5 * (probs + F.softmax(logits_flip, dim=-1))

        total_loss += float(loss.item()) * images.size(0)
        preds = torch.argmax(probs, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    avg_loss = total_loss / max(1, len(all_targets))
    acc = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    return avg_loss, acc, macro_f1, all_targets, all_preds


def compress_state_dict_fp16(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Store floating-point weights in FP16 for compact checkpoints while keeping int buffers intact."""
    return {
        k: (v.detach().half().cpu() if v.is_floating_point() else v.detach().cpu())
        for k, v in state_dict.items()
    }


def save_confusion_matrix(y_true: List[int], y_pred: List[int], out_path: Path, title_suffix: str = "") -> None:
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
    ax.set_title(f"Vision Agent — Test Confusion Matrix{title_suffix}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_gradcam_grid(
    agent: VisionAgent,
    test_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Render one representative test photo from each severity class (minor, moderate, severe)."""
    samples = []
    for cls_name in SEVERITY_CLASSES:
        sub = test_df[test_df["severity"] == cls_name]
        if not sub.empty:
            samples.append(sub.iloc[0])
    if not samples:
        return

    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(8, 3.5 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, row in enumerate(samples):
        img_path = str(row["image_path"])
        true_sev = str(row["severity"])
        orig_img = Image.open(img_path).convert("RGB").resize((288, 288))
        overlay_img, _, probs = agent.explain(img_path)
        pred_sev = max(probs, key=probs.get)

        axes[idx, 0].imshow(orig_img)
        axes[idx, 0].set_title(f"Original (True: {true_sev})")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(overlay_img.resize((288, 288)))
        axes[idx, 1].set_title(f"Ensemble Grad-CAM (Pred: {pred_sev}, conf={probs[pred_sev]:.2f})")
        axes[idx, 1].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def train_single_fold(
    fold_idx: int,
    num_folds: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], float, float]:
    tfs = build_vision_transforms(args.image_size)
    train_ds = DamageDataFrameDataset(train_df, tfs["train"])
    val_ds = DamageDataFrameDataset(val_df, tfs["eval"])

    num_workers = 2 if device.type == "cuda" else 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_cnn_backbone(
        backbone_name=args.backbone,
        num_classes=len(SEVERITY_CLASSES),
        pretrained=not args.no_pretrained,
        dropout=0.3,
    ).to(device)

    class_weights = compute_class_weights_from_df(train_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    val_criterion = nn.CrossEntropyLoss(weight=class_weights)

    use_amp = device.type == "cuda"
    scaler = make_grad_scaler(enabled=use_amp)

    best_val_f1 = -1.0
    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    print(f"\n================ FOLD {fold_idx + 1}/{num_folds} (train={len(train_df)}, val={len(val_df)}) ================")

    # Stage 1: Frozen Backbone Head Warmup
    if args.epochs_warmup > 0:
        set_backbone_trainable(model, args.backbone, stage="warmup")
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr_warmup,
            weight_decay=1e-4,
        )
        for epoch in range(1, args.epochs_warmup + 1):
            tr_loss, _, tr_f1 = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, use_amp, mixup_alpha=0.0
            )
            va_loss, va_acc, va_f1, _, _ = evaluate_model(
                model, val_loader, val_criterion, device, use_tta=True
            )
            print(
                f"  [Fold {fold_idx + 1} | Warmup {epoch:02d}/{args.epochs_warmup:02d}] "
                f"tr_loss={tr_loss:.4f} tr_f1={tr_f1:.4f} | val_acc={va_acc:.4f} val_f1={va_f1:.4f}"
            )
            if va_f1 > best_val_f1:
                best_val_f1 = va_f1
                best_val_acc = va_acc
                best_state = copy.deepcopy(model.state_dict())

    # Stage 2: Discriminative Full-Backbone Fine-Tuning + Clean Cooldown
    if args.epochs_finetune > 0:
        set_backbone_trainable(model, args.backbone, stage="finetune")
        param_groups = build_optimizer_param_groups(
            model, args.backbone, base_lr=args.lr_finetune, weight_decay=2e-4
        )
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs_finetune, eta_min=1e-6
        )
        cooldown_start = max(1, args.epochs_finetune - args.cooldown_epochs + 1)

        for epoch in range(1, args.epochs_finetune + 1):
            in_cooldown = epoch >= cooldown_start
            if in_cooldown:
                train_ds.transform = tfs["cooldown"]
            mix_alpha = 0.0 if in_cooldown else args.mixup_alpha

            tr_loss, _, tr_f1 = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                use_amp,
                mixup_alpha=mix_alpha,
            )
            scheduler.step()
            va_loss, va_acc, va_f1, _, _ = evaluate_model(
                model, val_loader, val_criterion, device, use_tta=True
            )
            tag = "Cooldown" if in_cooldown else "FineTune"
            print(
                f"  [Fold {fold_idx + 1} | {tag:8s} {epoch:02d}/{args.epochs_finetune:02d}] "
                f"tr_loss={tr_loss:.4f} tr_f1={tr_f1:.4f} | val_acc={va_acc:.4f} val_f1={va_f1:.4f}"
            )
            if va_f1 > best_val_f1:
                best_val_f1 = va_f1
                best_val_acc = va_acc
                best_state = copy.deepcopy(model.state_dict())

    print(f"--> Fold {fold_idx + 1} Best Val Accuracy: {best_val_acc:.4f} | Val Macro F1: {best_val_f1:.4f}")
    return compress_state_dict_fp16(best_state), best_val_acc, best_val_f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Train & evaluate Part A Vision CNN Ensemble (v2).")
    parser.add_argument("--splits-dir", type=Path, default=Path("data/processed/cnn_splits"))
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_v2_s",
        choices=["efficientnet_v2_s", "resnet50", "mobilenet_v3_large", "efficientnet_b0"],
    )
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--num-folds", type=int, default=4, help="Number of CV folds to ensemble (1 = single split).")
    parser.add_argument("--epochs-warmup", type=int, default=4)
    parser.add_argument("--epochs-finetune", type=int, default=18)
    parser.add_argument("--cooldown-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-warmup", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1.5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.08)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--output-weights", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training Vision Ensemble ({args.backbone}, {args.image_size}px, folds={args.num_folds}) on {device}")

    cv_csv = args.splits_dir / "cv_folds.csv"
    test_df = pd.read_csv(args.splits_dir / "test.csv")

    fold_states: List[Dict[str, torch.Tensor]] = []
    fold_val_f1s: List[float] = []
    fold_val_accs: List[float] = []

    if cv_csv.exists() and args.num_folds > 1:
        cv_df = pd.read_csv(cv_csv)
        available_folds = sorted(cv_df["fold"].unique().tolist())[: args.num_folds]
        for f_idx in available_folds:
            tr_df = cv_df[cv_df["fold"] != f_idx].reset_index(drop=True)
            va_df = cv_df[cv_df["fold"] == f_idx].reset_index(drop=True)
            sd_fp16, v_acc, v_f1 = train_single_fold(f_idx, len(available_folds), tr_df, va_df, args, device)
            fold_states.append(sd_fp16)
            fold_val_accs.append(v_acc)
            fold_val_f1s.append(v_f1)
    else:
        tr_df = pd.read_csv(args.splits_dir / "train.csv")
        va_df = pd.read_csv(args.splits_dir / "val.csv")
        sd_fp16, v_acc, v_f1 = train_single_fold(0, 1, tr_df, va_df, args, device)
        fold_states.append(sd_fp16)
        fold_val_accs.append(v_acc)
        fold_val_f1s.append(v_f1)

    best_fold_idx = int(np.argmax(fold_val_f1s))
    weights_path = args.output_weights or Path(f"models/vision_{args.backbone}.pt")
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    model_version = f"{args.backbone}-{len(fold_states)}fold-tta-v2"
    torch.save(
        {
            "backbone_name": args.backbone,
            "image_size": args.image_size,
            "use_tta": True,
            "model_version": model_version,
            "fold_state_dicts": fold_states,
            "model_state_dict": fold_states[best_fold_idx],
            "fold_val_macro_f1s": [round(x, 4) for x in fold_val_f1s],
            "cv_val_macro_f1_mean": round(float(np.mean(fold_val_f1s)), 4),
        },
        weights_path,
    )

    # Evaluate on the untouched CNN Test Split using VisionAgent (Ensemble + Flip TTA)
    agent = VisionAgent(weights_path=weights_path, device=str(device), use_tta=True)
    eval_tf = build_vision_transforms(args.image_size)["eval"]
    test_ds = DamageDataFrameDataset(test_df, eval_tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    y_true: List[int] = []
    y_pred: List[int] = []
    for images, targets, _ in test_loader:
        probs = agent.predict_proba_tensor(images)
        preds = torch.argmax(probs, dim=-1).tolist()
        y_pred.extend(preds)
        y_true.extend(targets.tolist())

    te_acc = float(accuracy_score(y_true, y_pred))
    te_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

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

    save_confusion_matrix(y_true, y_pred, cm_path, title_suffix=f" ({model_version})")
    save_gradcam_grid(agent, test_df, gradcam_path)

    metrics_payload = {
        "backbone": args.backbone,
        "model_version": model_version,
        "image_size": args.image_size,
        "num_folds": len(fold_states),
        "use_tta": True,
        "cv_val_accuracy_mean": round(float(np.mean(fold_val_accs)), 4),
        "cv_val_macro_f1_mean": round(float(np.mean(fold_val_f1s)), 4),
        "fold_val_macro_f1s": [round(x, 4) for x in fold_val_f1s],
        "test_accuracy": round(te_acc, 4),
        "test_macro_f1": round(te_f1, 4),
        "classification_report": report_dict,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("\n================ FINAL UNTOUCHED TEST SET EVALUATION ================")
    print(f"Model Version          : {model_version}")
    print(f"Mean CV Val Accuracy   : {np.mean(fold_val_accs):.4f}")
    print(f"Mean CV Val Macro F1   : {np.mean(fold_val_f1s):.4f}")
    print(f"Test Accuracy (TTA)    : {te_acc:.4f}")
    print(f"Test Macro F1 (TTA)    : {te_f1:.4f}")
    print(f"Saved weights          : {weights_path}")
    print(f"Saved figures          : {cm_path}, {gradcam_path}")


if __name__ == "__main__":
    main()
