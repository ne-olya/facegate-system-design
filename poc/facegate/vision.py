"""Детекция, оценка качества кадра и эвристика liveness.

Всё в этом модуле считается по пикселям. В целевой системе детектор меняется на
SCRFD или YuNet, а PAD - на обученную модель; интерфейсы функций остаются теми же.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config

_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y : self.y + self.h, self.x : self.x + self.w]


@dataclass
class QualityReport:
    face_detected: bool
    quality_score: float
    sharpness: float
    brightness: float
    face_ratio: float
    reason: str


def detect(frame: np.ndarray) -> FaceBox | None:
    """Возвращает наибольшее найденное лицо."""
    boxes = _CASCADE.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=3, minSize=(60, 60))
    if len(boxes) == 0:
        return None
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    return FaceBox(int(x), int(y), int(w), int(h))


def assess(frame: np.ndarray, box: FaceBox, cfg: Config) -> QualityReport:
    """Гейт качества. Низкое качество - повод переснять кадр, а не отказать в проходе."""
    face = box.crop(frame)
    sharpness = float(cv2.Laplacian(face, cv2.CV_64F).var())
    brightness = float(face.mean())
    ratio = (box.w * box.h) / float(frame.shape[0] * frame.shape[1])
    lower = face[face.shape[0] // 2 :]
    occluded_ratio = float((lower > 200).mean())

    q = cfg.quality
    reason = "quality_ok"
    if ratio < q.min_face_ratio:
        reason = "face_too_small"
    elif sharpness < q.min_sharpness:
        reason = "frame_blurred"
    elif brightness < q.min_brightness:
        reason = "underexposed"
    elif brightness > q.max_brightness:
        reason = "overexposed"
    elif occluded_ratio > q.max_occlusion:
        # Маска и шарф дают крупную однородную пересвеченную область в нижней части лица.
        reason = "face_occluded"

    score = min(1.0, sharpness / 200.0) * min(1.0, ratio / 0.08)
    return QualityReport(True, round(score, 3), round(sharpness, 1), round(brightness, 1), round(ratio, 4), reason)


def liveness_score(frame: np.ndarray, box: FaceBox) -> float | None:
    """Периодичность строк кадра: экран и печатный растр дают узкий пик в спектре.

    Сравниваем пик не со средним по полосе, а с медианой соседних бинов, иначе
    гладкая пересвеченная фотография даёт ложный выброс. None означает, что на
    таком кадре проверка не считается: движок решений обязан трактовать это как
    отсутствие проверки, а не как пройденную проверку.
    """
    face = box.crop(frame).astype(np.float32)
    if face.shape[1] < 64:
        return None
    rows = face - face.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(rows * np.hanning(rows.shape[1]), axis=1)).mean(axis=0)
    band = spectrum[8:]
    if band.size < 40:
        return None
    peak_at = int(np.argmax(band))
    neighbours = np.concatenate([band[max(0, peak_at - 15) : max(0, peak_at - 4)], band[peak_at + 4 : peak_at + 15]])
    if neighbours.size == 0:
        return None
    return float(band[peak_at] / (np.median(neighbours) + 1e-6))
