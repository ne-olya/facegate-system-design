"""Генератор demo-кадров.

Реальные фотографии сотрудников использовать нельзя (см. risks-and-ops.md), поэтому
кадры рисуются процедурно. Идентичность кодируется низкочастотным паттерном на
области лица: он играет роль черт лица, которые в целевой системе читает ArcFace.
"""

import hashlib

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

WIDTH, HEIGHT = 480, 640


def _seed(employee_id: str) -> int:
    return int(hashlib.sha256(employee_id.encode()).hexdigest()[:8], 16)


def _identity_pattern(shape: tuple[int, int], employee_id: str) -> np.ndarray:
    # Средние частоты: там базовая форма лица даёт мало энергии, поэтому паттерн
    # читается дескриптором и предсказуемо слабеет при размытии.
    rng = np.random.default_rng(_seed(employee_id))
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    pattern = np.zeros(shape, dtype=np.float32)
    for _ in range(4):
        fx, fy = rng.uniform(3.0, 9.0, size=2)
        phase = rng.uniform(0, 2 * np.pi)
        pattern += np.sin(2 * np.pi * (fx * xx / shape[1] + fy * yy / shape[0]) + phase)
    return pattern / 4.0


def render(
    employee_id: str,
    illumination: str = "normal",
    occlusion: str | None = None,
    yaw: int = 0,
    blur: float = 2.0,
    replay: bool = False,
    look_alike_of: str | None = None,
) -> np.ndarray:
    """Возвращает кадр в градациях серого.

    replay имитирует съёмку экрана телефона, look_alike_of смешивает паттерн с
    чужим: так в demo получается пара похожих сотрудников без реальных лиц.
    """
    gain = {"normal": 1.0, "dim": 0.4, "backlight": 1.0}.get(illumination, 1.0)
    img = Image.new("L", (WIDTH, HEIGHT), int(70 * gain))
    draw = ImageDraw.Draw(img)

    cx, cy = WIDTH // 2 + yaw, HEIGHT // 2
    fw, fh = 150, 200
    draw.ellipse([cx - fw // 2, cy - fh // 2, cx + fw // 2, cy + fh // 2], fill=int(190 * gain))
    draw.ellipse([cx - fw // 2 - 6, cy - fh // 2 - 30, cx + fw // 2 + 6, cy - fh // 2 + 40], fill=int(60 * gain))
    for dx in (-38, 38):
        ex = cx + dx + yaw // 3
        draw.ellipse([ex - 26, cy - 46, ex + 26, cy - 14], fill=int(150 * gain))
        draw.ellipse([ex - 13, cy - 38, ex + 13, cy - 18], fill=int(35 * gain))
    draw.polygon([(cx, cy - 10), (cx - 14, cy + 30), (cx + 14, cy + 30)], fill=int(205 * gain))
    draw.ellipse([cx - 32, cy + 52, cx + 32, cy + 76], fill=int(80 * gain))

    if occlusion == "mask":
        draw.rectangle([cx - fw // 2, cy + 20, cx + fw // 2, cy + fh // 2], fill=int(225 * gain))

    frame = np.asarray(img.filter(ImageFilter.GaussianBlur(blur))).astype(np.float32)

    face = (slice(cy - fh // 2, cy + fh // 2), slice(cx - fw // 2, cx + fw // 2))
    pattern = _identity_pattern(frame[face].shape, employee_id)
    if look_alike_of is not None:
        pattern = 0.22 * pattern + 0.78 * _identity_pattern(frame[face].shape, look_alike_of)
    frame[face] += 24.0 * pattern

    if illumination == "backlight":
        # Пересвет сверху и провал в тенях: типичная утренняя проходная против окна.
        gradient = np.linspace(155, -55, HEIGHT, dtype=np.float32)[:, None]
        frame += gradient

    rng = np.random.default_rng(_seed(employee_id) % 1000)
    frame += rng.normal(0, 5, frame.shape)

    if replay:
        cols = np.arange(WIDTH, dtype=np.float32)
        frame = frame * 0.85 + 14.0 * np.sin(2 * np.pi * cols / 5.0) + 15.0

    return np.clip(frame, 0, 255).astype(np.uint8)
