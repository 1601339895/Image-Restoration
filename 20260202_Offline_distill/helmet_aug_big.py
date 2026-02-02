import cv2
import numpy as np
import random
import albumentations as A
from loguru import logger

class HelmetOcclusionAug_big(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(HelmetOcclusionAug_big, self).__init__(always_apply, p)

    def apply(self, img, **params):
        img = self._validate_and_fix_input(img)
        
        if img.shape[0] != 128 or img.shape[1] != 128:
            img = cv2.resize(img, (128, 128))
        
        img = self._ensure_uint8_bgr(img)
        
        result = img.copy()
        result = self._add_forehead_cover(result)
        result = self._add_ear_cover(result)
        
        mode = random.choice([1, 2, 3, 4])
        
        if mode == 1:
            result = self._add_mouth_mask(result)
        elif mode == 2:
            result = self._add_transparent_mask(result, transparency=0.6, blur_amount=1.5)
        elif mode == 3:
            result = self._add_mouth_nose_mask(result)
        elif mode == 4:
            result = self._add_mouth_nose_mask(result)
            result = self._add_transparent_mask(result, transparency=0.6, blur_amount=1.5)
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        if random.random() < 0.001:
            logger.debug(f"HelmetAug applied mode: {mode}")
        
        return result

    def _validate_and_fix_input(self, img):
        if img is None:
            return np.zeros((128, 128, 3), dtype=np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] != 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            else:
                img = img[:, :, :3]
        return img

    def _ensure_uint8_bgr(self, img):
        if img.dtype != np.uint8:
            img = np.clip(img * 255 if np.max(img) <= 1.0 else img, 0, 255).astype(np.uint8)
        if img.shape[2] == 3:
            mean = np.mean(img, axis=(0,1))
            if mean[1] > mean[0] and mean[1] > mean[2]:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def _shadow(self, input_image, light=30):
        input_image = np.clip(input_image, 0, 255).astype(np.uint8)
        f = input_image.astype(np.float32) / 255.0
        gray = 0.299 * f[:,:,0] + 0.587 * f[:,:,1] + 0.114 * f[:,:,2]
        thresh = (1.0 - gray) ** 2
        t = np.mean(thresh) + 1e-6
        mask = np.where(thresh >= t, 255, 0).astype(np.uint8)

        max_val = 4  
        bright = light / 100.0 / max_val  
        mid = 1.0 + max_val * bright  
        midrate = np.where(mask == 255, mid, ((mid - 1.0) / t * thresh) + 1.0)
        brightrate = np.where(mask == 255, bright, (1.0 / t * thresh) * bright)
        
        result = np.clip(np.power(f, 1.0 / (midrate[:, :, np.newaxis] + 1e-6)), 0.0, 1.0)
        result = np.clip(result * (1.0 / (1 - brightrate[:, :, np.newaxis] + 1e-6)), 0.0, 1.0) * 255
        return np.clip(result, 0, 255).astype(np.uint8)

    def _add_forehead_cover(self, img, alpha=1.0):
        overlay = img.copy()
        pts = np.array([[0, 0], [128, 0], [120, 30], [8, 30]], np.int32)
        cv2.fillPoly(overlay, [pts], (0, 0, 0))
        result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        return result

    def _add_ear_cover(self, img):
        result = img.copy()
        left = np.array([[0, 45], [12, 35], [22, 75], [8, 95]], np.int32)
        right = np.array([[128, 45], [116, 35], [106, 75], [120, 95]], np.int32)
        cv2.fillPoly(result, [left], (0, 0, 0))
        cv2.fillPoly(result, [right], (0, 0, 0))
        return self._shadow(result, light=20)

    def _add_mouth_mask(self, img, alpha=1.0):
        mask = np.zeros((128, 128), dtype=np.uint8)
        cv2.ellipse(mask, (64, 95), (40, 20), 0, 0, 180, 255, -1)
        gradient = np.zeros((128, 128), dtype=np.float32)
        for i in range(128): gradient[:, i] = 0.7 + 0.3 * (i / 128.0)
        result = img.copy()
        for c in range(3):
            mk = (mask.astype(np.float32) / 255.0) * gradient
            result[:, :, c] = (img[:, :, c] * (1 - mk * alpha)).astype(np.uint8)
        return result

    def _add_mouth_nose_mask(self, img, alpha=1.0):
        mask = np.zeros((128, 128), dtype=np.uint8)
        cv2.ellipse(mask, (64, 75), (48, 40), 0, 0, 180, 255, -1)
        gradient = np.zeros((128, 128), dtype=np.float32)
        for i in range(128): gradient[:, i] = 0.8 + 0.2 * (i / 128.0)
        result = img.copy()
        for c in range(3):
            mk = (mask.astype(np.float32) / 255.0) * gradient
            result[:, :, c] = (img[:, :, c] * (1 - mk * alpha)).astype(np.uint8)
        return result
        
    def _add_transparent_mask(self, img, transparency=0.7, blur_amount=1.5):
        k = max(1, int(blur_amount * 2) | 1)
        blurred = cv2.GaussianBlur(img, (k, k), blur_amount)
        layer = np.ones_like(img, dtype=np.float32) * 240
        strength = 1.0 - max(0.0, min(1.0, transparency))
        result = img.astype(np.float32) * (1 - strength) + layer * strength
        return np.clip(result, 0, 255).astype(np.uint8)