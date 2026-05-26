from __future__ import annotations

from datetime import datetime, timezone

from app.models import RunAnalysisRequest
from app.task_queue import AnalysisJobPayload, RedisTaskQueueBackend


def test_analysis_job_payload_roundtrip():
    payload = AnalysisJobPayload(
        run_id="run_123",
        request=RunAnalysisRequest(dataset_id="ds_alpha", top_k_terms=12, topic_count=5, use_llm=False),
        created_at=datetime(2026, 5, 12, 8, tzinfo=timezone.utc),
    )

    encoded = payload.to_json()
    decoded = AnalysisJobPayload.from_json(encoded)

    assert decoded.run_id == payload.run_id
    assert decoded.request.dataset_id == "ds_alpha"
    assert decoded.request.top_k_terms == 12
    assert decoded.created_at == payload.created_at


def test_redis_queue_backend_enqueues_and_dequeues(monkeypatch):
    events: list[tuple[str, str, int | None]] = []

    class FakeRedis:
        def __init__(self):
            self.buffer: list[str] = []

        def lpush(self, queue_name: str, payload: str) -> None:
            events.append(("lpush", queue_name, None))
            self.buffer.insert(0, payload)

        def brpop(self, queue_name: str, timeout: int = 0):
            events.append(("brpop", queue_name, timeout))
            if not self.buffer:
                return None
            return queue_name, self.buffer.pop()

    fake_redis = FakeRedis()

    class FakeRedisFactory:
        @staticmethod
        def from_url(url: str, decode_responses: bool = False):
            assert url == "redis://cache:6379/0"
            assert decode_responses is True
            return fake_redis

    monkeypatch.setattr("app.task_queue.Redis", FakeRedisFactory)

    backend = RedisTaskQueueBackend(redis_url="redis://cache:6379/0", queue_name="analysis_jobs")
    payload = AnalysisJobPayload(
        run_id="run_456",
        request=RunAnalysisRequest(dataset_id="ds_beta", use_llm=False),
        created_at=datetime(2026, 5, 12, 9, tzinfo=timezone.utc),
    )

    backend.enqueue_analysis_job(payload=payload)
    dequeued = backend.dequeue_analysis_job(timeout_seconds=7)

    assert dequeued is not None
    assert dequeued.run_id == "run_456"
    assert dequeued.request.dataset_id == "ds_beta"
    assert events == [("lpush", "analysis_jobs", None), ("brpop", "analysis_jobs", 7)]
