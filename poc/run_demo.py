"""Demo: прогоняет пять событий с проходной и печатает решения.

Запуск: python run_demo.py
"""

import json
import sys
from pathlib import Path

from facegate import embed, frames, vision
from facegate.audit import AuditLog
from facegate.config import DEFAULT
from facegate.gallery import EdgeCache, Gallery
from facegate.pipeline import Pipeline

HERE = Path(__file__).parent
STAFF = {
    "emp-4821": {},
    "emp-1102": {},
    "emp-7730": {},
    "emp-2290": {},
    "emp-2291": {"look_alike_of": "emp-2290"},
    "emp-3140": {},
}


def build_gallery() -> Gallery:
    """Энролмент: по одному эталонному кадру на сотрудника.

    В целевой системе шаблонов 3-10 на человека (очки, ракурсы), см. docs/ml.md.
    """
    gallery = Gallery()
    for employee_id, scene in STAFF.items():
        frame = frames.render(employee_id, **scene)
        box = vision.detect(frame)
        if box is None:
            raise RuntimeError(f"энролмент не удался: лицо не найдено, {employee_id}")
        gallery.enroll(employee_id, embed.encode(box.crop(frame)))
    return gallery


def main() -> int:
    audit_path = HERE / "out" / "access_log.jsonl"
    audit_path.unlink(missing_ok=True)
    cache = EdgeCache(gallery=build_gallery(), revoked={"emp-9001"})
    pipeline = Pipeline(cache=cache, audit=AuditLog(audit_path), cfg=DEFAULT)

    events = [json.loads(line) for line in (HERE / "demo" / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]
    events.append(events[0])  # повтор первого события: проверяем идемпотентность

    print(f"галерея на edge: {len(cache.gallery)} сотрудников, вектор {embed.DIM}-D\n")
    header = f"{'event':<8}{'решение':<15}{'сотрудник':<11}{'score':>7}{'margin':>8}{'liveness':>10}{'кадров':>8}{'команда':>19}  причины"
    print(header)
    print("-" * len(header))
    for event in events:
        answer = pipeline.process(event)
        liveness = answer["quality"]["liveness_score"]
        print(
            f"{answer['event_id']:<8}{answer['decision']:<15}{answer['employee_id'] or '-':<11}"
            f"{answer['match_score'] or 0:>7.3f}{answer['margin_to_second_best'] or 0:>8.3f}"
            f"{(f'{liveness:.1f}' if liveness is not None else '-'):>10}{answer['frames_used']:>8}"
            f"{answer['turnstile_command']:>19}  {', '.join(answer['reasons'])}"
        )

    print(f"\nтурникет открывался {len(pipeline.turnstile.opened)} раз(а)")
    print(f"журнал доступа: {audit_path.relative_to(HERE)} ({len(pipeline.audit.records())} записей)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
