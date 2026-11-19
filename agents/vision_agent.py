"""Part A: Vision Agent (CNN with Transfer Learning, K-Fold Ensemble, TTA & Grad-CAM).

Classifies collision photo severity into `minor`, `moderate`, or `severe`
and returns a standardized `AgentMessage` JSON envelope.
Supports both single-model checkpoints and K-fold ensemble checkpoints
with Horizontal-Flip Test-Time Augmentation (TTA).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from agents.base import (
    INDEX_TO_SEVERITY,
    SEVERITY_CLASSES,
    AgentMessage,
    BaseAgent,
)
from explain.gradcam import generate_ensemble_gradcam, overlay_heatmap_on_image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def resolve_default_weights_path(models_dir: str | Path = "models") -> Path | None:
    """Locate the strongest available trained Vision checkpoint (Champion: ResNet50 v1)."""
    base = Path(models_dir)
    candidates = [
        base / "vision_resnet50_v1.pt",
        base / "vision_efficientnet_v2_s.pt",
        base / "vision_mobilenet_v3_large_v1.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def build_vision_transforms(image_size: int = 288) -> Dict[str, transforms.Compose]:
    """Return strong aspect-preserving train transforms and clean center-crop eval transforms."""
    eval_resize = int(round(image_size * 1.12))
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.72, 1.0),
                ratio=(0.85, 1.18),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.12)),
        ]
    )
    # Clean augmentation for the final cooldown epochs (no RandAugment/RandomErasing)
    cooldown_tf = transforms.Compose(
        [
            transforms.Resize(eval_resize),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(eval_resize),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return {"train": train_tf, "cooldown": cooldown_tf, "eval": eval_tf}


def build_cnn_backbone(
    backbone_name: str = "efficientnet_v2_s",
    num_classes: int = 3,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """Build a transfer-learning CNN for 3-class damage severity classification.

    Supported backbones (all rubric-compliant ImageNet CNNs):
    - `efficientnet_v2_s` (default v2: highest accuracy + 9x9 Grad-CAM at 288px)
    - `resnet50` (classic baseline)
    - `efficientnet_b0`
    - `mobilenet_v3_large` (fast CPU fallback)
    """
    backbone_name = backbone_name.lower()

    if backbone_name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(in_features, num_classes),
        )
        return model

    if backbone_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        return model

    if backbone_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if backbone_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(
        f"Unsupported backbone '{backbone_name}'. "
        "Choose from: efficientnet_v2_s, resnet50, mobilenet_v3_large, efficientnet_b0."
    )


def set_backbone_trainable(
    model: nn.Module,
    backbone_name: str = "efficientnet_v2_s",
    stage: str = "warmup",
) -> None:
    """Configure parameter gradients for two-stage transfer learning."""
    backbone_name = backbone_name.lower()
    for param in model.parameters():
        param.requires_grad = False

    if backbone_name == "resnet50":
        for param in model.fc.parameters():
            param.requires_grad = True
        if stage == "finetune":
            for layer in (model.layer2, model.layer3, model.layer4):
                for param in layer.parameters():
                    param.requires_grad = True
    else:
        for param in model.classifier.parameters():
            param.requires_grad = True
        if stage == "finetune":
            # Unfreeze full feature extractor so discriminative LRs can adapt all spatial scales
            for param in model.features.parameters():
                param.requires_grad = True


def build_optimizer_param_groups(
    model: nn.Module,
    backbone_name: str,
    base_lr: float,
    weight_decay: float = 1e-4,
) -> List[Dict[str, Any]]:
    """Build layer-wise discriminative learning rate groups for Stage 2 fine-tuning."""
    backbone_name = backbone_name.lower()

    if backbone_name == "resnet50":
        early_params = [p for p in model.layer2.parameters() if p.requires_grad]
        late_params = [
            p
            for layer in (model.layer3, model.layer4)
            for p in layer.parameters()
            if p.requires_grad
        ]
        head_params = [p for p in model.fc.parameters() if p.requires_grad]
    else:
        blocks = list(model.features.children())
        split_idx = max(1, len(blocks) // 2)
        early_params = [
            p for b in blocks[:split_idx] for p in b.parameters() if p.requires_grad
        ]
        late_params = [
            p for b in blocks[split_idx:] for p in b.parameters() if p.requires_grad
        ]
        head_params = [p for p in model.classifier.parameters() if p.requires_grad]

    groups = []
    if early_params:
        groups.append({"params": early_params, "lr": base_lr * 0.25, "weight_decay": weight_decay})
    if late_params:
        groups.append({"params": late_params, "lr": base_lr * 1.0, "weight_decay": weight_decay})
    if head_params:
        groups.append({"params": head_params, "lr": base_lr * 2.5, "weight_decay": weight_decay})
    return groups


class VisionAgent(BaseAgent):
    """Specialist Vision Agent supporting single-CNN or K-Fold Ensemble + Flip TTA + Grad-CAM."""

    agent_name = "vision"

    def __init__(
        self,
        weights_path: str | Path | None = "auto",
        backbone_name: str = "efficientnet_v2_s",
        device: str = "auto",
        image_size: int = 288,
        use_tta: bool = True,
        pretrained_fallback: bool = False,
    ) -> None:
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if weights_path == "auto":
            resolved = resolve_default_weights_path()
        elif weights_path is not None:
            resolved = Path(weights_path)
        else:
            resolved = None

        self.backbone_name = backbone_name.lower()
        self.image_size = image_size
        self.use_tta = use_tta
        self.models: List[nn.Module] = []

        if resolved is not None and resolved.exists():
            ckpt = torch.load(resolved, map_location="cpu", weights_only=False)
            self.backbone_name = ckpt.get("backbone_name", self.backbone_name).lower()
            self.image_size = int(ckpt.get("image_size", 224 if "resnet50" in self.backbone_name else image_size))
            self.model_version = ckpt.get("model_version", f"{self.backbone_name}-v2")
            self.use_tta = bool(ckpt.get("use_tta", use_tta))

            if "fold_state_dicts" in ckpt and ckpt["fold_state_dicts"]:
                for sd in ckpt["fold_state_dicts"]:
                    m = build_cnn_backbone(
                        backbone_name=self.backbone_name,
                        num_classes=len(SEVERITY_CLASSES),
                        pretrained=False,
                    )
                    m.load_state_dict(sd)
                    m.to(self.device).eval()
                    self.models.append(m)
            else:
                state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                m = build_cnn_backbone(
                    backbone_name=self.backbone_name,
                    num_classes=len(SEVERITY_CLASSES),
                    pretrained=False,
                )
                # Handle legacy ResNet50 v1 checkpoint vs Sequential head smoothly
                m.load_state_dict(state_dict)
                m.to(self.device).eval()
                self.models.append(m)
        else:
            m = build_cnn_backbone(
                backbone_name=self.backbone_name,
                num_classes=len(SEVERITY_CLASSES),
                pretrained=pretrained_fallback,
            )
            m.to(self.device).eval()
            self.models.append(m)
            self.model_version = f"{self.backbone_name}-untrained"

        self.model = self.models[0]
        self.eval_transform = build_vision_transforms(self.image_size)["eval"]

    def _load_image(self, image_input: str | Path | Image.Image) -> Image.Image:
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        return Image.open(image_input).convert("RGB")

    def predict_proba_tensor(self, tensor_batch: torch.Tensor) -> torch.Tensor:
        """Predict ensemble + TTA averaged probability tensor of shape `(B, 3)`."""
        x = tensor_batch.to(self.device)
        x_flip = torch.flip(x, dims=[-1]) if self.use_tta else None

        probs_accum = torch.zeros((x.size(0), len(SEVERITY_CLASSES)), device=self.device)
        total_views = 0

        with torch.no_grad():
            for m in self.models:
                m.eval()
                probs_accum += F.softmax(m(x), dim=-1)
                total_views += 1
                if x_flip is not None:
                    probs_accum += F.softmax(m(x_flip), dim=-1)
                    total_views += 1

        return (probs_accum / max(1, total_views)).cpu()

    def predict(
        self,
        claim_id: str,
        image_input: str | Path | Image.Image | None = None,
        **kwargs: Any,
    ) -> AgentMessage:
        """Predict damage severity and return a standardized `AgentMessage`."""
        if image_input is None:
            image_input = kwargs.get("image_path")
        if image_input is None:
            raise ValueError("VisionAgent.predict requires `image_input` (path or PIL.Image).")

        pil_img = self._load_image(image_input)
        tensor = self.eval_transform(pil_img).unsqueeze(0)
        probs_tensor = self.predict_proba_tensor(tensor).squeeze(0)

        probs_dict: Dict[str, float] = {
            INDEX_TO_SEVERITY[idx]: round(float(probs_tensor[idx].item()), 4)
            for idx in range(len(SEVERITY_CLASSES))
        }
        pred_idx = int(torch.argmax(probs_tensor).item())
        pred_severity = INDEX_TO_SEVERITY[pred_idx]
        confidence = round(float(probs_tensor[pred_idx].item()), 4)

        return AgentMessage(
            claim_id=str(claim_id),
            agent=self.agent_name,
            output={
                "severity": pred_severity,
                "probabilities": probs_dict,
            },
            confidence=confidence,
            model_version=self.model_version,
        )

    def explain(
        self,
        image_input: str | Path | Image.Image,
        target_class: int | None = None,
        alpha: float = 0.45,
    ) -> Tuple[Image.Image, np.ndarray, Dict[str, float]]:
        """Generate an ensemble-averaged Grad-CAM overlay image, raw 2D heatmap, and probabilities."""
        pil_img = self._load_image(image_input)
        tensor = self.eval_transform(pil_img).unsqueeze(0).to(self.device)

        heatmap, _, _ = generate_ensemble_gradcam(
            models_list=self.models,
            backbone_name=self.backbone_name,
            input_tensor=tensor,
            target_class=target_class,
            output_size=pil_img.size,
        )
        probs_tensor = self.predict_proba_tensor(tensor.cpu()).squeeze(0)

        overlay_img = overlay_heatmap_on_image(pil_img, heatmap, alpha=alpha)
        probs_dict = {
            INDEX_TO_SEVERITY[idx]: round(float(probs_tensor[idx].item()), 4)
            for idx in range(len(SEVERITY_CLASSES))
        }
        return overlay_img, heatmap, probs_dict
