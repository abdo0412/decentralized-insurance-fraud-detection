"""Explainability helpers (Grad-CAM for Vision, SHAP for Decision)."""

from explain.gradcam import GradCAM, get_target_layer, overlay_heatmap_on_image

__all__ = ["GradCAM", "get_target_layer", "overlay_heatmap_on_image"]
