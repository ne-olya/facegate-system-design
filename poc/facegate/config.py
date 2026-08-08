"""Пороги конвейера. Держим в одном месте: в проде это конфиг, который меняет
безопасник без релиза кода, а не константы, размазанные по модулям."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Quality:
    """Гейт качества кадра. Значения подобраны по demo-кадрам, см. README."""

    min_face_ratio: float = 0.03
    min_sharpness: float = 25.0
    min_brightness: float = 45.0
    max_brightness: float = 195.0
    max_occlusion: float = 0.35


@dataclass(frozen=True)
class Liveness:
    # Спектральная метрика периодичности. Выше порога deny - почти наверняка экран.
    deny_above: float = 14.0
    ok_below: float = 12.0


@dataclass(frozen=True)
class Matching:
    allow: float = 0.80
    deny_below: float = 0.55
    min_margin: float = 0.06
    # В строгом режиме (устаревший кеш) поднимаем порог, а не пускаем "как обычно".
    strict_allow_bonus: float = 0.06


@dataclass(frozen=True)
class Cache:
    fresh_minutes: int = 60
    strict_minutes: int = 1440
    dead_minutes: int = 4320


@dataclass(frozen=True)
class Config:
    quality: Quality = Quality()
    liveness: Liveness = Liveness()
    matching: Matching = Matching()
    cache: Cache = Cache()
    max_frames: int = 3
    turnstile_dedup_seconds: int = 10


DEFAULT = Config()
