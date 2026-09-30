"""Image I/O and normalisation for QA (8/16-bit PNG via OpenCV; RGB float32 in [0,1])."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

Img = NDArray[np.float32]


def decode(data: bytes) -> Img:
    arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError("cannot decode image")
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    scale = 65535.0 if arr.dtype == np.uint16 else 255.0
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB).astype(np.float32) / scale
    return rgb


def encode_png16(img: Img) -> bytes:
    arr = np.clip(np.rint(img * 65535.0), 0, 65535).astype(np.uint16)
    ok, buf = cv2.imencode(
        ".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6]
    )
    if not ok:
        raise ValueError("PNG encode failed")
    return bytes(buf)


def encode_jpeg(img: Img, quality: int = 95) -> bytes:
    arr = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise ValueError("JPEG encode failed")
    return bytes(buf)


def resize_to_width(img: Img, width: int) -> Img:
    h, w = img.shape[:2]
    if w == width:
        return img
    height = max(1, round(h * width / w))
    interp = cv2.INTER_AREA if width < w else cv2.INTER_CUBIC
    return np.asarray(cv2.resize(img, (width, height), interpolation=interp), dtype=np.float32)


def luminance(img: Img) -> NDArray[np.float32]:
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    return np.asarray(lum, dtype=np.float32)
