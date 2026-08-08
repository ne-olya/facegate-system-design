"""Движок решений: allow, manual_review, deny.

Здесь нет моделей. Всё, что влияет на открытие турникета, описано правилами:
их проверяет безопасник, они воспроизводимы и меняются без переобучения.
"""

from dataclasses import dataclass, field

from .config import Config
from .gallery import Candidate
from .vision import QualityReport


@dataclass
class Verdict:
    decision: str
    reasons: list[str] = field(default_factory=list)
    employee_id: str | None = None
    match_score: float | None = None
    margin_to_second_best: float | None = None
    next_action: str = "use_card"


def _fail(decision: str, reason: str, action: str) -> Verdict:
    return Verdict(decision=decision, reasons=[reason], next_action=action)


def decide(
    quality: QualityReport | None,
    liveness: float,
    candidates: list[Candidate],
    cache_mode: str,
    revoked: bool,
    online: bool = True,
    cfg: Config = None,
) -> Verdict:
    """Порядок проверок задан ценой ошибки: сначала отсекаем дешёвые причины."""
    cfg = cfg or Config()

    if quality is None:
        return _fail("deny", "face_not_detected", "use_card")
    if quality.reason != "quality_ok":
        return _fail("manual_review", quality.reason, "go_to_guard")
    if liveness >= cfg.liveness.deny_above:
        return _fail("deny", "liveness_failed", "go_to_guard")
    if cache_mode == "dead":
        return _fail("manual_review", "cache_expired", "go_to_guard")

    reasons = ["quality_ok"]
    reasons.append("liveness_ok" if liveness < cfg.liveness.ok_below else "liveness_uncertain")

    if not candidates:
        return _fail("deny", "no_candidates", "go_to_guard")

    top = candidates[0]
    second = candidates[1].score if len(candidates) > 1 else 0.0
    margin = round(top.score - second, 4)
    verdict = Verdict(
        decision="manual_review",
        reasons=reasons,
        employee_id=top.employee_id,
        match_score=round(top.score, 4),
        margin_to_second_best=margin,
        next_action="go_to_guard",
    )

    if revoked:
        verdict.decision = "deny"
        verdict.reasons.append("access_revoked")
        verdict.next_action = "go_to_guard"
        return verdict

    allow_threshold = cfg.matching.allow
    if cache_mode == "strict":
        # Кеш мог не получить отзыв доступа, поэтому цена ошибки выше обычной.
        allow_threshold += cfg.matching.strict_allow_bonus
        verdict.reasons.append("degraded_stale_cache")
    if cache_mode == "card_only":
        verdict.reasons.append("degraded_card_only")
        return verdict
    if not online and cache_mode != "fresh":
        # Отзыв доступа физически не мог доехать до узла, поэтому автоматический
        # проход запрещён независимо от того, насколько уверенно совпало лицо.
        verdict.reasons.append("revocation_may_be_stale")
        return verdict

    if top.score < cfg.matching.deny_below:
        verdict.decision = "deny"
        verdict.reasons.append("match_below_deny_threshold")
        return verdict
    if margin < cfg.matching.min_margin:
        verdict.reasons.append("second_candidate_too_close")
        return verdict
    if top.score < allow_threshold:
        verdict.reasons.append("match_in_review_band")
        return verdict
    if "liveness_uncertain" in verdict.reasons:
        return verdict

    verdict.decision = "allow"
    verdict.reasons.append("match_above_allow_threshold")
    verdict.next_action = "open"
    return verdict
