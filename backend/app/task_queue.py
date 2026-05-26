from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from threading import Thread
from typing import Any, Optional

from .models import RunAnalysisRequest
from .services.analysis_jobs import execute_analysis_job
from .settings import get_settings

try:
    from redis import Redis
except ImportError:  # pragma: no cover - optional dependency for redis runtime only
    Redis = None


@dataclass(frozen=True)
class AnalysisJobPayload:
    run_id: str
    owner_key: str
    request: RunAnalysisRequest
    created_at: datetime

    def to_json(self) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "owner_key": self.owner_key,
                "request": self.request.model_dump(mode="json"),
                "created_at": self.created_at.isoformat(),
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, raw: str) -> "AnalysisJobPayload":
        payload = json.loads(raw)
        return cls(
            run_id=payload["run_id"],
            owner_key=payload["owner_key"],
            request=RunAnalysisRequest.model_validate(payload["request"]),
            created_at=datetime.fromisoformat(payload["created_at"]),
        )


class TaskQueueBackend(ABC):
    @abstractmethod
    def enqueue_analysis_job(self, *, payload: AnalysisJobPayload) -> None:
        raise NotImplementedError


class InProcessTaskQueueBackend(TaskQueueBackend):
    def enqueue_analysis_job(self, *, payload: AnalysisJobPayload) -> None:
        worker = Thread(
            target=execute_analysis_job,
            args=(payload.run_id, payload.request, payload.created_at, payload.owner_key),
            daemon=True,
        )
        worker.start()


class RedisTaskQueueBackend(TaskQueueBackend):
    def __init__(self, *, redis_url: str, queue_name: str) -> None:
        if not redis_url:
            raise RuntimeError("REDIS_URL is required for redis task queue backend")
        if Redis is None:
            raise RuntimeError("redis is not installed; cannot use redis task queue backend")
        self.redis = Redis.from_url(redis_url, decode_responses=True)
        self.queue_name = queue_name

    def enqueue_analysis_job(self, *, payload: AnalysisJobPayload) -> None:
        self.redis.lpush(self.queue_name, payload.to_json())

    def dequeue_analysis_job(self, *, timeout_seconds: int = 5) -> Optional[AnalysisJobPayload]:
        item = self.redis.brpop(self.queue_name, timeout=timeout_seconds)
        if not item:
            return None
        _, raw = item
        return AnalysisJobPayload.from_json(raw)


def get_task_queue_backend() -> TaskQueueBackend:
    settings = get_settings()
    backend = settings.task_queue_backend
    if backend == "inprocess":
        return InProcessTaskQueueBackend()
    if backend == "redis":
        return RedisTaskQueueBackend(redis_url=settings.redis_url, queue_name=settings.task_queue_name)
    raise RuntimeError(f"Unsupported task queue backend: {backend}")


def enqueue_analysis_job(
    *,
    run_id: str,
    owner_key: str,
    request: RunAnalysisRequest,
    created_at: datetime,
    handler: Any | None = None,
) -> None:
    _ = handler
    get_task_queue_backend().enqueue_analysis_job(
        payload=AnalysisJobPayload(run_id=run_id, owner_key=owner_key, request=request, created_at=created_at)
    )


def run_worker_loop(*, max_jobs: Optional[int] = None, poll_interval_seconds: float = 1.0) -> int:
    backend = get_task_queue_backend()
    if not isinstance(backend, RedisTaskQueueBackend):
        raise RuntimeError("run_worker_loop requires TASK_QUEUE_BACKEND=redis")

    completed = 0
    while max_jobs is None or completed < max_jobs:
        payload = backend.dequeue_analysis_job(timeout_seconds=max(1, int(poll_interval_seconds)))
        if payload is None:
            if max_jobs is not None:
                time.sleep(poll_interval_seconds)
            continue
        execute_analysis_job(payload.run_id, payload.request, payload.created_at, payload.owner_key)
        completed += 1
    return completed


def check_task_queue_connection() -> tuple[bool, str]:
    settings = get_settings()
    if settings.task_queue_backend == "inprocess":
        return True, "inprocess thread queue ready"

    if settings.task_queue_backend == "redis":
        try:
            backend = get_task_queue_backend()
            assert isinstance(backend, RedisTaskQueueBackend)
            backend.redis.ping()
            return True, "redis reachable"
        except Exception as exc:  # pragma: no cover - exercised in environment checks
            return False, str(exc)

    return False, f"Unsupported task queue backend: {settings.task_queue_backend}"
