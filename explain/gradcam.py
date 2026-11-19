"""Grad-CAM (Gradient-weighted Class Activation Mapping) for the Vision Agent.

Generates spatial heatmaps over collision photos to verify that the CNN focuses
on vehicle damage (dents, scratches, shattered glass) rather than background noise.
Uses tensor-level gradient hooks so backbones with inplace SiLU/ReLU activations
(such as EfficientNetV2, MobileNetV3, and ResNet) work without inplace view errors.
"""

from __future__ import annotations

from typing import List, Tuple

import matplotlib.cm as cm
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_target_layer(model: nn.Module, backbone_name: str = "efficientnet_v2_s") -> nn.Module:
    """Return the final convolutional layer of the chosen backbone for Grad-CAM."""
    backbone_name = backbone_name.lower()
    if backbone_name == "resnet50":
        return model.layer4[-1]
    if backbone_name in ("efficientnet_v2_s", "efficientnet_b0", "mobilenet_v3_large"):
        return model.features[-1]
    raise ValueError(f"Unsupported backbone for automatic target layer lookup: {backbone_name}")


class GradCAM:
    """Computes Grad-CAM heatmaps for a PyTorch CNN classifier (or ensemble of classifiers)."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self._fwd_hook = self.target_layer.register_forward_hook(self._save_activation)

    def _save_activation(
        self,
        module: nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        # Clone and return output so backward pass flows through `act` without inplace view conflicts
        act = output.clone()
        self.activations = act
        if act.requires_grad:
            act.register_hook(self._save_gradient)
        return act

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self.gradients = grad.detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
        output_size: Tuple[int, int] = (288, 288),
    ) -> Tuple[np.ndarray, int, torch.Tensor]:
        """Compute a normalized [0, 1] 2D Grad-CAM heatmap.

        Args:
            input_tensor: Preprocessed tensor of shape `(1, 3, H, W)`.
            target_class: Class index to explain. If None, uses the predicted argmax class.
            output_size: `(width, height)` to upsample the heatmap to.

        Returns:
            Tuple of `(heatmap_2d_np, explained_class_idx, probabilities_1d_tensor)`.
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        x = input_tensor.clone().detach().requires_grad_(True)
        logits = self.model(x)
        probs = F.softmax(logits, dim=-1).squeeze(0)

        if target_class is None:
            target_class = int(torch.argmax(probs).item())

        score = logits[0, target_class]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        acts = self.activations.detach()
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(output_size[1], output_size[0]),
            mode="bilinear",
            align_corners=False,
        )
        cam_np = cam.squeeze().cpu().numpy()

        cam_min, cam_max = float(cam_np.min()), float(cam_np.max())
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np, target_class, probs.detach().cpu()

    def remove_hooks(self) -> None:
        self._fwd_hook.remove()


def generate_ensemble_gradcam(
    models_list: List[nn.Module],
    backbone_name: str,
    input_tensor: torch.Tensor,
    target_class: int | None = None,
    output_size: Tuple[int, int] = (288, 288),
) -> Tuple[np.ndarray, int, torch.Tensor]:
    """Average Grad-CAM heatmaps and probabilities across a K-fold CNN ensemble."""
    if not models_list:
        raise ValueError("models_list must contain at least one model.")

    # First determine ensemble argmax class if target_class is None
    if target_class is None:
        with torch.no_grad():
            probs_sum = torch.zeros(3, device=input_tensor.device)
            for m in models_list:
                m.eval()
                probs_sum += F.softmax(m(input_tensor), dim=-1).squeeze(0)
            avg_probs = (probs_sum / len(models_list)).cpu()
            target_class = int(torch.argmax(avg_probs).item())

    cam_accum = np.zeros((output_size[1], output_size[0]), dtype=np.float32)
    probs_accum = torch.zeros(3, dtype=torch.float32)

    for m in models_list:
        layer = get_target_layer(m, backbone_name)
        gcam = GradCAM(m, layer)
        try:
            cam_i, _, probs_i = gcam.generate(
                input_tensor,
                target_class=target_class,
                output_size=output_size,
            )
            cam_accum += cam_i
            probs_accum += probs_i
        finally:
            gcam.remove_hooks()

    cam_avg = cam_accum / len(models_list)
    c_min, c_max = float(cam_avg.min()), float(cam_avg.max())
    if c_max - c_min > 1e-8:
        cam_avg = (cam_avg - c_min) / (c_max - c_min)

    probs_avg = probs_accum / len(models_list)
    return cam_avg, int(target_class), probs_avg


def overlay_heatmap_on_image(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> Image.Image:
    """Blend a [0, 1] 2D Grad-CAM heatmap over an RGB PIL Image."""
    rgb_img = original_image.convert("RGB")
    w, h = rgb_img.size

    if heatmap.shape != (h, w):
        heatmap_pil = Image.fromarray(np.uint8(255 * heatmap)).resize((w, h), Image.BILINEAR)
        heatmap_resized = np.asarray(heatmap_pil, dtype=np.float32) / 255.0
    else:
        heatmap_resized = heatmap

    cmap = cm.get_cmap(colormap)
    colored_heatmap = cmap(heatmap_resized)[:, :, :3]
    img_np = np.asarray(rgb_img, dtype=np.float32) / 255.0

    blended = (1.0 - alpha) * img_np + alpha * colored_heatmap
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)
