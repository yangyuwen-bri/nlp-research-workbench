from app.runtime_guard import assert_runtime_ready_for_startup, evaluate_runtime_readiness
from app.settings import Settings


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
        local_exact_match_enable=False,
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
    return Settings(**(base.__dict__ | overrides))


def test_runtime_readiness_flags_development_backends_as_not_ready():
    result = evaluate_runtime_readiness(_build_settings())
    checks = {item["id"]: item for item in result["checks"]}

    assert result["ready"] is False
    assert checks["storage_backend"]["status"] == "failed"
    assert checks["task_queue_backend"]["status"] == "failed"


def test_runtime_guard_skips_blocking_in_development():
    assert_runtime_ready_for_startup(_build_settings())


def test_runtime_guard_blocks_production_with_release_blockers():
    settings = _build_settings(app_env="production", strict_startup_checks=True)

    try:
        assert_runtime_ready_for_startup(settings)
    except RuntimeError as exc:
        message = str(exc)
        assert "storage_backend" in message
        assert "task_queue_backend" in message
    else:
        raise AssertionError("production startup should be blocked when release backends are missing")
