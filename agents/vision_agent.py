"""Part A: Vision Agent (CNN with Transfer Learning + Grad-CAM).

Classifies collision photo severity into `minor`, `moderate`, or `severe`
and returns a standardized `AgentMessage` JSON envelope.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

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
from explain.gradcam import GradCAM, get_target_layer, overlay_heatmap_on_image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_vision_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
    """Return train (with augmentation) and eval transforms matching ImageNet stats."""
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return {"train": train_tf, "eval": eval_tf}


def build_cnn_backbone(
    backbone_name: str = "resnet50",
    num_classes: int = 3,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """Build a transfer-learning CNN for 3-class damage severity classification.

    Supported backbones:
    - `resnet50` (default, strong accuracy + clean 7x7 Grad-CAM on `layer4[-1]`)
    - `mobilenet_v3_large` (fast CPU/low-GPU fallback)
    - `efficientnet_b0`
    """
    backbone_name = backbone_name.lower()

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
        f"Unsupported backbone '{backbone_name}'. Choose from: resnet50, mobilenet_v3_large, efficientnet_b0."
    )


def set_backbone_trainable(
    model: nn.Module,
    backbone_name: str = "resnet50",
    stage: str = "warmup",
) -> None:
    """Configure parameter gradients for two-stage transfer learning.

    - `stage="warmup"`: Freeze entire convolutional backbone; train only classification head.
    - `stage="finetune"`: Unfreeze top convolutional blocks (`layer3` + `layer4` for ResNet50)
      plus classification head.
    """
    backbone_name = backbone_name.lower()

    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    if backbone_name == "resnet50":
        for param in model.fc.parameters():
            param.requires_grad = True
        if stage == "finetune":
            for param in model.layer3.parameters():
                param.requires_grad = True
            for param in model.layer4.parameters():
                param.requires_grad = True
    elif backbone_name in ("mobilenet_v3_large", "efficientnet_b0"):
        for param in model.classifier.parameters():
            param.requires_grad = True
        if stage == "finetune":
            # Unfreeze the last 3 feature blocks
            for block in list(model.features.children())[-3:]:
                for param in block.parameters():
                    param.requires_grad = True


class VisionAgent(BaseAgent):
    """Specialist Vision Agent that scores collision photos and produces Grad-CAM maps."""

    agent_name = "vision"

    def __init__(
        self,
        weights_path: str | Path | None = None,
        backbone_name: str = "resnet50",
        device: str = "auto",
        image_size: int = 224,
        pretrained_fallback: bool = False,
    ) -> None:
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.backbone_name = backbone_name.lower()
        self.image_size = image_size
        self.model_version = f"{self.backbone_name}-v1"
        self.eval_transform = build_vision_transforms(image_size)["eval"]

        # Load checkpoint if available; otherwise initialize architecture
        if weights_path is not None and Path(weights_path).exists():
            ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
            self.backbone_name = ckpt.get("backbone_name", self.backbone_name)
            self.model_version = ckpt.get("model_version", f"{self.backbone_name}-v1")
            self.model = build_cnn_backbone(
                backbone_name=self.backbone_name,
                num_classes=len(SEVERITY_CLASSES),
                pretrained=False,
            )
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(state_dict)
        else:
            self.model = build_cnn_backbone(
                backbone_name=self.backbone_name,
                num_classes=len(SEVERITY_CLASSES),
                pretrained=pretrained_fallback,
            )
            self.model_version = f"{self.backbone_name}-untrained"

        self.model.to(self.device)
        self.model.eval()

    def _load_image(self, image_input: str | Path | Image.Image) -> Image.Image:
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        return Image.open(image_input).convert("RGB")

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
        tensor = self.eval_transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs_tensor = F.softmax(logits, dim=-1).squeeze(0).cpu()

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
        """Generate a Grad-CAM overlay image, raw 2D heatmap, and class probabilities."""
        pil_img = self._load_image(image_input)
        tensor = self.eval_transform(pil_img).unsqueeze(0).to(self.device)

        target_layer = get_target_layer(self.model, self.backbone_name)
        gradcam = GradCAM(self.model, target_layer)
        try:
            heatmap, _, probs_tensor = gradcam.generate(
                tensor,
                target_class=target_class,
                output_size=pil_img.size,
            )
        finally:
            gradcam.remove_hooks()

        overlay_img = overlay_heatmap_on_image(pil_img, heatmap, alpha=alpha)
        probs_dict = {
            INDEX_TO_SEVERITY[idx]: round(float(probs_tensor[idx].item()), 4)
            for idx in range(len(SEVERITY_CLASSES))
        }
        return overlay_img, heatmap, probs_dict
