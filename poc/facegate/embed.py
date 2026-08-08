"""Дескриптор лица для прототипа.

Это не face recognition модель. Считаем энергию спектра области лица по кольцам и
секторам средних частот: дескриптор устойчив к яркости, слабеет при размытии и
разваливается на плохих кадрах, то есть ведёт себя качественно так же, как
настоящий эмбеддинг. В целевой системе сюда встаёт ArcFace (512-D), вызывающий
код не меняется.
"""

import cv2
import numpy as np

DIM = 84
_RINGS = [(3, 5), (5, 7), (7, 9), (9, 11), (11, 13), (13, 16), (16, 20)]
_SECTORS = 12


def encode(face_crop: np.ndarray) -> np.ndarray:
    """Возвращает L2-нормированный вектор длины DIM."""
    patch = cv2.resize(face_crop, (64, 64)).astype(np.float32)
    patch = (patch - patch.mean()) / (patch.std() + 1e-6)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(patch * np.outer(np.hanning(64), np.hanning(64)))))

    yy, xx = np.mgrid[0:64, 0:64] - 32
    radius = np.sqrt(yy**2 + xx**2)
    angle = np.arctan2(yy, xx)

    features = []
    for lo, hi in _RINGS:
        ring = (radius >= lo) & (radius < hi)
        cells = []
        for s in range(_SECTORS):
            lo_a = -np.pi + s * 2 * np.pi / _SECTORS
            cell = spectrum[ring & (angle >= lo_a) & (angle < lo_a + 2 * np.pi / _SECTORS)]
            cells.append(cell.mean() if cell.size else 0.0)
        # Нормируем внутри кольца: радиальный профиль спектра одинаков у всех лиц и
        # без этого шага забивает угловую структуру, которая и различает людей.
        cells = np.asarray(cells, dtype=np.float32)
        features.extend(cells / (cells.mean() + 1e-6) - 1.0)

    vector = np.nan_to_num(np.asarray(features, dtype=np.float32))
    return vector / (np.linalg.norm(vector) + 1e-9)
