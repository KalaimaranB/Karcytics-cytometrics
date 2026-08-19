import cv2
import numpy as np
import math


class WatershedPipeline:
    def __init__(self):
        self.name = "Watershed (Clump Splitting)"

    def run(self, image_stack, params, scale=1.0):
        target_name = params.get("target_channel")
        use_dual = params.get("use_dual_channel", False)
        seed_name = params.get("seed_channel")
        min_area = params.get("min_area_px", 100)
        max_area = params.get("max_area_px", 50000)

        target_img = next((ch.data for ch in image_stack.channels if ch.name == target_name), None)
        if target_img is None:
            return []

        img_h, img_w = target_img.shape

        # ==========================================
        # LAYER 1: THE SEEDS (Start Positions)
        # ==========================================
        sure_fg = None
        if use_dual and seed_name:
            seed_img = next((ch.data for ch in image_stack.channels if ch.name == seed_name), None)
            if seed_img is not None:
                # Find nuclei and split any touching ones
                blur_seed = cv2.GaussianBlur(seed_img, (5, 5), 0)
                _, seed_mask = cv2.threshold(blur_seed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                dist_seed = cv2.distanceTransform(seed_mask, cv2.DIST_L2, 3)
                _, sure_fg = cv2.threshold(dist_seed, 0.2 * dist_seed.max(), 255, 0)
                sure_fg = np.uint8(sure_fg)

        # ==========================================
        # LAYER 2: THE FOOTPRINT (Cytoplasm)
        # ==========================================
        # Slight blur to smooth the target
        blur_target = cv2.GaussianBlur(target_img, (5, 5), 0)

        # Calculate Otsu, but use 30% of it so we catch the dim extensions!
        otsu_val, _ = cv2.threshold(blur_target, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        sensitive_val = max(5, otsu_val * 0.3)
        _, footprint = cv2.threshold(blur_target, sensitive_val, 255, cv2.THRESH_BINARY)

        # Close tiny gaps
        kernel = np.ones((3, 3), np.uint8)
        footprint = cv2.morphologyEx(footprint, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Fallback if no dual-channel is used
        if sure_fg is None:
            dist_target = cv2.distanceTransform(footprint, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_target, 0.5 * dist_target.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)

        # ==========================================
        # LAYER 3: WATERSHED EXECUTION
        # ==========================================
        # Define the absolute background boundary
        sure_bg = cv2.dilate(footprint, kernel, iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Set up the markers array for OpenCV
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Run watershed on the smoothed target image
        target_bgr = cv2.cvtColor(blur_target, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(target_bgr, markers)

        # ==========================================
        # FILTERING & METRICS
        # ==========================================
        detected_cells = []
        unique_labels = np.unique(markers)

        for label in unique_labels:
            if label <= 1:  # Skip background and boundary lines
                continue

            cell_mask = np.zeros_like(target_img, dtype=np.uint8)
            cell_mask[markers == label] = 255

            contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            cnt = contours[0]

            # Area Filter
            area_px = cv2.contourArea(cnt)
            if not (min_area <= area_px <= max_area):
                continue

            # Edge Filter (Only drop if touching the absolute camera edge)
            x, y, w, h = cv2.boundingRect(cnt)
            if x <= 1 or y <= 1 or (x + w) >= (img_w - 1) or (y + h) >= (img_h - 1):
                continue

            perim_px = cv2.arcLength(cnt, True)
            area_um2 = area_px * (scale ** 2)
            perim_um = perim_px * scale
            circularity = (4 * math.pi * area_um2) / (perim_um ** 2) if perim_um > 0 else 0.0

            points = [[pt[0][0], pt[0][1]] for pt in cnt]

            detected_cells.append({
                "points": points,
                "area": area_um2,
                "perim": perim_um,
                "circ": circularity
            })

        return detected_cells