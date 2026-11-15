"""Grad-CAM (Gradient-weighted Class Activation Mapping) for the Vision Agent.

Generates spatial heatmaps over collision photos to verify that the CNN focuses
on vehicle damage (dents, scratches, shattered glass) rather than background noise.
"""

from __future__ import annotations

from typing import Tuple

import matplotlib.cm as cm
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_target_layer(model: nn.Module, backbone_name: str = "resnet50") -> nn.Module:
    """Return the final convolutional layer of the chosen backbone for Grad-CAM."""
    backbone_name = backbone_name.lower()
    if backbone_name == "resnet50":
        return model.layer4[-1]
    if backbone_name == "mobilenet_v3_large":
        return model.features[-1]
    if backbone_name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"Unsupported backbone for automatic target layer lookup: {backbone_name}")


class GradCAM:
    """Computes Grad-CAM heatmaps for a PyTorch CNN classifier."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self._fwd_hook = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(
        self,
        module: nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output.detach()

    def _save_gradient(
        self,
        module: nn.Module,
        grad_input: Tuple[torch.Tensor | None, ...],
        grad_output: Tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
        output_size: Tuple[int, int] = (224, 224),
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

        # Ensure input requires grad so backward pass flows through the target layer
        x = input_tensor.clone().detach().requires_grad_(True)
        logits = self.model(x)
        probs = F.softmax(logits, dim=-1).squeeze(0)

        if target_class is None:
            target_class = int(torch.argmax(probs).item())

        score = logits[0, target_class]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        # Global average pooling over spatial dimensions (H_feat, W_feat) -> (1, C, 1, 1)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to original image size (height=output_size[1], width=output_size[0])
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
        self._bwd_hook.remove()


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
    colored_heatmap = cmap(heatmap_resized)[:, :, :3]  # Drop alpha channel -> (H, W, 3) in [0, 1]
    img_np = np.asarray(rgb_img, dtype=np.float32) / 255.0

    blended = (1.0 - alpha) * img_np + alpha * colored_heatmap
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)
