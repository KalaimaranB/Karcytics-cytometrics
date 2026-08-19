import cv2
import math
from .base import SegmentationPipeline


class OtsuPipeline(SegmentationPipeline):
    """Classic computer vision segmentation using Otsu's thresholding."""

    @property
    def name(self) -> str:
        return "Basic Threshold (Otsu)"

    def get_parameters(self) -> dict:
        return {
            "target_channel": "channel_select",
            "min_area_px": "int",
            "max_area_px": "int"
        }

    def run(self, image_stack, parameters: dict, scale: float) -> list:
        # Find the specific channel the user wants to analyze
        target_name = parameters.get("target_channel")
        channel = next((c for c in image_stack.channels if c.name == target_name), None)

        if not channel:
            return []

        # 1. Processing
        blurred = cv2.GaussianBlur(channel.data, (5, 5), 0)
        ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cells = []
        min_area = parameters.get("min_area_px", 50)
        max_area = parameters.get("max_area_px", 100000)

        # 2. Metric Extraction
        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            if area_px < min_area or area_px > max_area:
                continue

            perim_px = cv2.arcLength(cnt, True)
            points = [(float(pt[0][0]), float(pt[0][1])) for pt in cnt]

            area_um2 = area_px * (scale ** 2)
            perim_um = perim_px * scale
            circ = (4 * math.pi * area_um2) / (perim_um ** 2) if perim_um > 0 else 0.0

            cells.append({
                "points": points,
                "area": area_um2,
                "perim": perim_um,
                "circ": circ
            })

        return cells