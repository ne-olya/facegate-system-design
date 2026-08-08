import json
from pathlib import Path

import pytest

from facegate import embed, frames, vision
from facegate.audit import AuditLog
from facegate.config import DEFAULT
from facegate.gallery import EdgeCache
from facegate.pipeline import Pipeline
from run_demo import build_gallery

EVENTS = {
    json.loads(line)["event_id"]: json.loads(line)
    for line in (Path(__file__).parents[1] / "demo" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    if line
}


@pytest.fixture
def pipeline(tmp_path):
    cache = EdgeCache(gallery=build_gallery(), revoked={"emp-9001"})
    return Pipeline(cache=cache, audit=AuditLog(tmp_path / "access_log.jsonl"), cfg=DEFAULT)


def test_happy_path_opens_turnstile(pipeline):
    answer = pipeline.process(EVENTS["e-1001"])
    assert answer["decision"] == "allow"
    assert answer["employee_id"] == "emp-4821"
    assert answer["turnstile_command"] == "open"
    assert "match_above_allow_threshold" in answer["reasons"]


@pytest.mark.parametrize(
    "event_id,reason",
    [("e-1002", "face_occluded"), ("e-1003", "liveness_failed"), ("e-1004", "second_candidate_too_close"),
     ("e-1005", "revocation_may_be_stale")],
)
def test_risky_paths_never_open(pipeline, event_id, reason):
    answer = pipeline.process(EVENTS[event_id])
    assert answer["decision"] in {"deny", "manual_review"}
    assert answer["turnstile_command"] == "keep_closed"
    assert reason in answer["reasons"]


def test_quality_gate_spends_retries_before_escalating(pipeline):
    answer = pipeline.process(EVENTS["e-1002"])
    assert answer["frames_used"] == DEFAULT.max_frames
    assert answer["requires_human_review"] is True


def test_revoked_employee_is_denied(pipeline):
    pipeline.cache.revoked.add("emp-4821")
    answer = pipeline.process(EVENTS["e-1001"])
    assert answer["decision"] == "deny"
    assert "access_revoked" in answer["reasons"]


def test_replay_of_same_event_does_not_open_twice(pipeline):
    first = pipeline.process(EVENTS["e-1001"])
    second = pipeline.process(EVENTS["e-1001"])
    assert second["decision_id"] == first["decision_id"]
    assert second["turnstile_command"] == "none"
    assert len(pipeline.turnstile.opened) == 1


def test_audit_log_keeps_reason_and_drops_biometrics(pipeline):
    pipeline.process(EVENTS["e-1003"])
    record = pipeline.audit.records()[-1]
    assert record["reasons"] and record["audit_id"]
    assert "frame" not in record and "embedding" not in record


def test_descriptor_separates_identities():
    def vector(employee_id, **scene):
        frame = frames.render(employee_id, **scene)
        return embed.encode(vision.detect(frame).crop(frame))

    own = vector("emp-4821")
    same_person_dim = vector("emp-4821", illumination="dim")
    other = vector("emp-7730")
    assert float(own @ same_person_dim) > float(own @ other) + 0.1
