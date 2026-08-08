"""Галерея шаблонов и её кеш на проходной.

На edge лежат только сотрудники, привязанные к площадке. Полный перебор здесь
дешевле ANN: 4000 векторов по 512 float32 это 8 МБ и доли миллисекунды.
Обоснование и цифры для центрального индекса - в docs/ml.md.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Candidate:
    employee_id: str
    score: float


class Gallery:
    """Несколько шаблонов на сотрудника, счёт по лучшему из них."""

    def __init__(self) -> None:
        self._templates: dict[str, list[np.ndarray]] = {}

    def enroll(self, employee_id: str, vector: np.ndarray) -> None:
        self._templates.setdefault(employee_id, []).append(vector)

    def search(self, probe: np.ndarray, k: int = 2) -> list[Candidate]:
        scored = [
            Candidate(employee_id, float(max(probe @ t for t in templates)))
            for employee_id, templates in self._templates.items()
        ]
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:k]

    def __len__(self) -> int:
        return len(self._templates)


@dataclass
class EdgeCache:
    """Локальная копия галереи с возрастом и списком отзыва.

    Список отзыва приходит отдельным приоритетным каналом, поэтому он может быть
    свежее самой галереи. Проверяем его до сравнения с базой.
    """

    gallery: Gallery
    age_minutes: int = 0
    revoked: set[str] = field(default_factory=set)

    def mode(self, cfg) -> str:
        if self.age_minutes <= cfg.cache.fresh_minutes:
            return "fresh"
        if self.age_minutes <= cfg.cache.strict_minutes:
            return "strict"
        if self.age_minutes <= cfg.cache.dead_minutes:
            return "card_only"
        return "dead"

    def is_revoked(self, employee_id: str) -> bool:
        return employee_id in self.revoked
