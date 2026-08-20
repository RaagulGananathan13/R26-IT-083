"""
Grad-CAM for the Stage 5 ConvNeXt classifier, plus image encoding helpers.

Hooks the last convolutional stage (12x12x1024). The gradient of the chosen
pathology logit is backpropagated to that layer, giving a spatial map of which
regions drove the prediction, upsampled and overlaid on the radiograph.

⚠️ Grad-CAM shows WHERE the model looked, not WHETHER it was right. Arun et al.
(Radiology: AI 2021) measured Grad-CAM repeatability at SSIM 0.12 on chest
radiographs. Treat these overlays as a sanity check -- "did it look at the
heart?" -- never as localisation evidence.
"""
from __future__ import annotations

import base64
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, img_size: int = 384):
        self.model = model
        self.img_size = img_size
        self.gradients = None
        self.activations = None
        target = model.features[-1]
        target.register_forward_hook(self._fwd)
        target.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        self.activations = out.detach()

    def _bwd(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x: torch.Tensor, target_class: int) -> np.ndarray:
        """Returns a (img_size, img_size) map in [0, 1]."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            onehot = torch.zeros_like(logits)
            onehot[0, target_class] = 1.0
            logits.backward(gradient=onehot, retain_graph=False)

        if self.gradients is None or self.activations is None:
            return np.zeros((self.img_size, self.img_size), dtype=np.float32)

        w = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((w * self.activations).sum(dim=1, keepdim=True))
        cam = cam - cam.amin()
        mx = cam.amax()
        if float(mx) > 0:
            cam = cam / mx
        cam = F.interpolate(cam.float(), size=(self.img_size, self.img_size),
                            mode="bilinear", align_corners=False)
        return cam.squeeze().cpu().numpy()


def overlay_heatmap(img_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> str:
    cam_u8 = (np.clip(cam, 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    h, w = img_bgr.shape[:2]
    if heat.shape[:2] != (h, w):
        heat = cv2.resize(heat, (w, h))
    blended = cv2.addWeighted(img_bgr, 1 - alpha, heat, alpha, 0)
    ok, buf = cv2.imencode(".png", blended)
    return base64.b64encode(buf.tobytes()).decode("utf-8") if ok else ""


def image_to_base64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf.tobytes()).decode("utf-8") if ok else ""
