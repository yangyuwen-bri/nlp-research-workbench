from __future__ import annotations

from pathlib import Path
import sys

from app.settings import Settings


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from check_platform_readiness import evaluate_platform_readiness


def _build_settings(**overrides) -> Settings:
    base = Settings(
        app_env="development",
        strict_startup_checks=False,
        storage_backend="json",
        task_queue_backend="inprocess",
        task_queue_name="analysis_jobs",
        cors_allow_origins=["http://127.0.0.1:3000"],
        database_url="",
        redis_url="",
        dashscope_api_key="",
        dashscope_base_url="https://example.com",
        dashscope_model="model",
        dashscope_timeout_seconds=120,
        bailian_enable_llm=False,
        dashscope_embedding_model="embed",
        dashscope_embedding_dimensions=1024,
        dashscope_embedding_concurrency=3,
        local_model_enable=True,
        local_transformer_enable=False,
        local_exact_match_enable=True,
        local_reference_model_enable=True,
        local_sentiment_model="sentiment",
        local_zero_shot_model="zero-shot",
        local_model_batch_size=8,
        local_model_max_documents=512,
        local_sentiment_neutral_threshold=0.6,
        local_zero_shot_hypothesis_template="{}",
        upload_max_file_bytes=10,
        upload_max_rows=10,
        upload_max_columns=10,
        upload_max_text_length=10,
    )
    values = base.__dict__ | overrides
    return Settings(**values)


def test_platform_readiness_reports_release_blockers(monkeypatch):
    monkeypatch.setattr(
        "check_platform_readiness.evaluate_runtime_readiness",
        lambda: {
            "ready": False,
            "checks": [
                {"id": "storage_backend", "status": "failed", "message": "json"},
                {"id": "task_queue_backend", "status": "failed", "message": "inprocess"},
            ],
            "summary": {"passed": 0, "failed": 2},
            "recommendation": "平台仍未达到生产环境要求，应先修复失败项。",
        },
    )

    result = evaluate_platform_readiness()

    assert result["ready"] is False
    checks = {item["id"]: item for item in result["checks"]}
    assert checks["storage_backend"]["status"] == "failed"
    assert checks["task_queue_backend"]["status"] == "failed"


def test_platform_readiness_passes_with_release_backends(monkeypatch):
    monkeypatch.setattr(
        "check_platform_readiness.evaluate_runtime_readiness",
        lambda: {
            "ready": True,
            "checks": [
                {"id": "storage_backend", "status": "passed", "message": "postgres"},
                {"id": "task_queue_backend", "status": "passed", "message": "redis"},
            ],
            "summary": {"passed": 2, "failed": 0},
            "recommendation": "平台已满足生产环境基础要求。",
        },
    )

    result = evaluate_platform_readiness()

    assert result["ready"] is True
    assert result["summary"]["failed"] == 0
