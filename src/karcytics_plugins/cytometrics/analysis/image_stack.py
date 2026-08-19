import cv2
import numpy as np

# Standard fluorescent mapping
COLOR_MAPS = {
    "gray": (1.0, 1.0, 1.0),
    "red": (0.0, 0.0, 1.0),  # OpenCV is BGR
    "green": (0.0, 1.0, 0.0),
    "blue": (1.0, 0.0, 0.0),
    "magenta": (1.0, 0.0, 1.0),  # Great for Actin!
    "cyan": (1.0, 1.0, 0.0),
    "yellow": (0.0, 1.0, 1.0)
}


class Channel:
    """Represents a single biological stain/marker."""

    def __init__(self, name: str, data: np.ndarray, color: str = "gray"):
        self.name = name
        self.data = data  # 2D Grayscale Numpy Array (0-255)
        self.color = color
        self.visible = True


class ImageStack:
    """Manages multiple channels and renders the composite."""

    def __init__(self):
        self.channels = []

    def add_channel(self, file_path: str, name: str, default_color: str = "gray") -> list:
        """Loads an image. Splits multi-channel images. Returns list of (ch_name, ch_color) added."""
        # Use UNCHANGED to preserve color channels and bit-depth
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return []

        added_info = []

        # Helper to normalize 16-bit TIFFs to 8-bit for UI/processing
        def to_8bit(channel_data):
            if channel_data.dtype == np.uint16:
                return (channel_data / 256).astype(np.uint8)
            return channel_data

        # If it's a multi-channel image (e.g. RGB)
        if len(img.shape) == 3:
            h, w, c = img.shape
            # OpenCV loads color as BGR (Blue, Green, Red)
            color_presets = ["blue", "green", "red"] if c == 3 else ["gray"] * c

            for i in range(c):
                single_channel = to_8bit(img[:, :, i])
                ch_name = f"{name} (Ch {i + 1})"
                ch_color = color_presets[i] if i < len(color_presets) else "gray"

                self.channels.append(Channel(ch_name, single_channel, ch_color))
                added_info.append((ch_name, ch_color))
        else:
            # Single channel grayscale
            img = to_8bit(img)
            self.channels.append(Channel(name, img, default_color))
            added_info.append((name, default_color))

        return added_info

    def get_composite(self) -> np.ndarray:
        """Blends all visible channels into a single BGR image for the UI."""
        if not self.channels:
            return None

        # Base black canvas
        h, w = self.channels[0].data.shape
        composite = np.zeros((h, w, 3), dtype=np.float32)

        for ch in self.channels:
            if not ch.visible:
                continue

            # Normalize to 0.0 - 1.0
            normalized = ch.data.astype(np.float32) / 255.0

            # Apply color tint
            b, g, r = COLOR_MAPS.get(ch.color, (1.0, 1.0, 1.0))
            colorized = np.dstack([normalized * b, normalized * g, normalized * r])

            # Additive blending (how light works in microscopes)
            composite += colorized

        # Clip highlights and convert back to 8-bit image
        composite = np.clip(composite * 255.0, 0, 255).astype(np.uint8)
        return composite