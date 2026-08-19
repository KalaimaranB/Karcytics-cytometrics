import math

import cv2
import numpy as np

_STACKED_CHANNELS_NDIM = 3  # (channel, height, width) — a dual-channel stack, not a flat 2D image


class CellposePipeline:
    def __init__(self):
        self.name = "AI Smart Detect (Cellpose)"
        self.model = None
        self.device = None

    def _ensure_model(self):
        """Lazy loader for the heavy AI model."""
        if self.model is not None:
            return

        import torch
        from cellpose import models

        # --- ENTERPRISE HARDWARE CHECK ---
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            use_gpu = True
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            use_gpu = True
        else:
            self.device = torch.device("cpu")
            use_gpu = False

        # Cellpose v4.2+ dropped the old cyto/cyto2/cyto3 model zoo in favor of the
        # Cellpose-SAM generalist model, loaded via `pretrained_model` (not `model`).
        self.model = models.CellposeModel(pretrained_model="cpsam", gpu=use_gpu, device=self.device)

    def run(self, image_stack, params, scale=1.0):
        self._ensure_model()
        target_name = params.get("target_channel")
        use_dual = params.get("use_dual_channel", False)
        seed_name = params.get("seed_channel")
        min_area = params.get("min_area_px", 100)
        max_area = params.get("max_area_px", 100000)

        diameter = params.get("diameter", None)
        flow_threshold = params.get("flow_threshold", 0.4)

        # Grab the new parameter from the UI (defaulting to True)
        exclude_borders = params.get("exclude_borders", True)

        target_img = next((ch.data for ch in image_stack.channels if ch.name == target_name), None)
        if target_img is None:
            return []

        img_h, img_w = target_img.shape

        # Cellpose-SAM ("cpsam") takes a (Channels, Height, Width) array and is
        # order-invariant between cytoplasm/nuclei channels — the old `channels=[cyto, nuclei]`
        # index argument from Cellpose v1-v3 is deprecated and silently ignored in v4+;
        # `channel_axis` is how the channel dimension is now identified.
        if use_dual and seed_name:
            seed_img = next((ch.data for ch in image_stack.channels if ch.name == seed_name), None)
            stacked_img = np.array([target_img, seed_img]) if seed_img is not None else target_img
        else:
            stacked_img = target_img

        channel_axis = 0 if stacked_img.ndim == _STACKED_CHANNELS_NDIM else None

        # Run the AI
        masks, flows, styles = self.model.eval(
            stacked_img,
            diameter=diameter,
            channel_axis=channel_axis,
            flow_threshold=flow_threshold,
        )

        detected_cells = []
        unique_labels = np.unique(masks)

        for label in unique_labels:
            if label == 0:
                continue

            cell_mask = np.zeros_like(masks, dtype=np.uint8)
            cell_mask[masks == label] = 255

            contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            cnt = contours[0]

            # --- APPLY BORDER EXCLUSION IF CHECKED ---
            if exclude_borders:
                x, y, w, h = cv2.boundingRect(cnt)
                # If the bounding box is within 1 pixel of the image edge, skip it entirely
                if x <= 1 or y <= 1 or (x + w) >= (img_w - 1) or (y + h) >= (img_h - 1):
                    continue

            area_px = cv2.contourArea(cnt)
            if not (min_area <= area_px <= max_area):
                continue
            perim_px = cv2.arcLength(cnt, True)

            # FORCE STANDARD PYTHON TYPES HERE
            area_um2 = float(area_px * (scale**2))
            perim_um = float(perim_px * scale)
            circularity = float((4 * math.pi * area_um2) / (perim_um**2) if perim_um > 0 else 0.0)

            # FORCE STANDARD INT FOR COORDINATES
            points = [[int(pt[0][0]), int(pt[0][1])] for pt in cnt]

            detected_cells.append(
                {"points": points, "area": area_um2, "perim": perim_um, "circ": circularity}
            )

        return detected_cells
