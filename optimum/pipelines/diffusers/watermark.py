import numpy as np
from imwatermark import WatermarkEncoder


WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]


# Adapted from https://github.com/huggingface/diffusers/blob/v0.18.1/src/diffusers/pipelines/stable_diffusion_xl/watermark.py#L12
class StableDiffusionXLWatermarker:
    def __init__(self):
        self.watermark = WATERMARK_BITS
        self.encoder = WatermarkEncoder()
        self.encoder.set_watermark("bits", self.watermark)

    def apply_watermark(self, images: np.array):
        # can't encode images that are smaller than 256
        if images.shape[-1] < 256:
            return images

        # cv2 doesn't support float16
        if images.dtype == np.float16:
            images = images.astype(np.float32)

        images = (255 * (images / 2 + 0.5)).transpose((0, 2, 3, 1))

        images = np.array([self.encoder.encode(image, "dwtDct") for image in images]).transpose((0, 3, 1, 2))

        np.clip(2 * (images / 255 - 0.5), -1.0, 1.0, out=images)

        return images
