"""Сборка горячего пути: кадр, качество, liveness, эмбеддинг, поиск, решение, аудит."""

import time
import uuid
from dataclasses import dataclass, field

from . import embed, frames, vision
from .audit import AuditLog
from .config import Config
from .gallery import EdgeCache


@dataclass
class Turnstile:
    """Заглушка контроллера с дедупликацией импульса по decision_id."""

    opened: list[str] = field(default_factory=list)
    _seen: set[str] = field(default_factory=set)

    def open(self, decision_id: str) -> str:
        if decision_id in self._seen:
            return "duplicate_ignored"
        self._seen.add(decision_id)
        self.opened.append(decision_id)
        return "open"


class Pipeline:
    def __init__(self, cache: EdgeCache, audit: AuditLog, cfg: Config = None) -> None:
        self.cache = cache
        self.audit = audit
        self.cfg = cfg or Config()
        self.turnstile = Turnstile()
        self._decisions: dict[str, dict] = {}

    def process(self, event: dict) -> dict:
        """Возвращает решение по событию. Повтор того же event_id не пересчитывается."""
        if event["event_id"] in self._decisions:
            answer = dict(self._decisions[event["event_id"]])
            answer["reasons"] = answer["reasons"] + ["idempotent_replay"]
            # Решение не пересчитываем, а команду отправляем повторно с тем же
            # decision_id: от второго открытия защищает дедупликация на контроллере,
            # и проверить её можно только отправив дубликат.
            if answer["decision"] == "allow":
                answer["turnstile_command"] = self.turnstile.open(answer["decision_id"])
            # Повтор пишем в журнал отдельной записью, иначе попытка переиграть
            # событие не видна при разборе инцидента.
            answer["audit_id"] = self.audit.write(answer)
            return answer

        started = time.perf_counter()
        meta = event.get("metadata", {})
        self.cache.age_minutes = int(meta.get("cache_age_minutes", 0))

        quality = None
        liveness = None
        candidates = []
        attempts = 0
        while attempts < self.cfg.max_frames:
            attempts += 1
            frame = self._frame(event, attempt=attempts)
            box = vision.detect(frame)
            if box is None:
                quality = None
                continue
            quality = vision.assess(frame, box, self.cfg)
            if quality.reason != "quality_ok" and attempts < self.cfg.max_frames:
                continue
            liveness = vision.liveness_score(frame, box)
            candidates = self.cache.gallery.search(embed.encode(box.crop(frame)), k=2)
            break

        from .policy import decide

        top_id = candidates[0].employee_id if candidates else None
        verdict = decide(
            quality=quality,
            liveness=liveness,
            candidates=candidates,
            cache_mode=self.cache.mode(self.cfg),
            online=meta.get("network", "online") != "offline",
            revoked=bool(top_id and self.cache.is_revoked(top_id)),
            cfg=self.cfg,
        )

        decision_id = f"d-{uuid.uuid4().hex[:8]}"
        command = self.turnstile.open(decision_id) if verdict.decision == "allow" else "keep_closed"
        answer = {
            "event_id": event["event_id"],
            "decision_id": decision_id,
            "decision": verdict.decision,
            "employee_id": verdict.employee_id,
            "match_score": verdict.match_score,
            "margin_to_second_best": verdict.margin_to_second_best,
            "quality": {
                "face_detected": quality is not None,
                "quality_score": quality.quality_score if quality else 0.0,
                "sharpness": quality.sharpness if quality else 0.0,
                "brightness": quality.brightness if quality else 0.0,
                "liveness_score": round(liveness, 2) if liveness is not None else None,
            },
            "reasons": verdict.reasons,
            "frames_used": attempts,
            "turnstile_command": command,
            "requires_human_review": verdict.decision == "manual_review",
            "degraded_mode": self.cache.mode(self.cfg) != "fresh" or meta.get("network") == "offline",
            "cache_age_minutes": self.cache.age_minutes,
            "next_action": verdict.next_action,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        answer["audit_id"] = self.audit.write(answer)
        self._decisions[event["event_id"]] = answer
        return answer

    def _frame(self, event: dict, attempt: int) -> "object":
        """Кадр с камеры. В demo рисуется процедурно, кадры на диск не пишутся."""
        scene = dict(event["scene"])
        subject = scene.pop("subject")
        if attempt > 1 and scene.get("retry_recovers"):
            scene = {"blur": 2.0}
        scene.pop("retry_recovers", None)
        return frames.render(subject, **scene)
