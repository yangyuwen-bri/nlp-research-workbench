from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_csv_list(value: str | None, *, default: list[str]) -> list[str]:
    if value is None:
        return default
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


@dataclass(frozen=True)
class Settings:
    app_env: str
    strict_startup_checks: bool
    storage_backend: str
    task_queue_backend: str
    task_queue_name: str
    cors_allow_origins: list[str]
    database_url: str
    redis_url: str
    dashscope_api_key: str
    dashscope_base_url: str
    dashscope_model: str
    dashscope_timeout_seconds: int
    bailian_enable_llm: bool
    dashscope_embedding_model: str
    dashscope_embedding_dimensions: int
    dashscope_embedding_concurrency: int
    local_model_enable: bool
    local_transformer_enable: bool
    local_exact_match_enable: bool
    local_reference_model_enable: bool
    local_sentiment_model: str
    local_zero_shot_model: str
    local_model_batch_size: int
    local_model_max_documents: int
    local_sentiment_neutral_threshold: float
    local_zero_shot_hypothesis_template: str
    upload_max_file_bytes: int
    upload_max_rows: int
    upload_max_columns: int
    upload_max_text_length: int

    @property
    def llm_ready(self) -> bool:
        return self.bailian_enable_llm and bool(self.dashscope_api_key)

    @property
    def is_production(self) -> bool:
        return self.app_env in {"production", "release"}

    @property
    def embedding_ready(self) -> bool:
        return bool(self.dashscope_api_key) and bool(self.dashscope_embedding_model)

    @property
    def local_model_ready(self) -> bool:
        return self.local_model_enable and bool(self.local_sentiment_model) and bool(self.local_zero_shot_model)


@lru_cache
def get_settings() -> Settings:
    app_env = os.getenv("APP_ENV", "development").strip().lower() or "development"
    return Settings(
        app_env=app_env,
        strict_startup_checks=_to_bool(
            os.getenv("STRICT_STARTUP_CHECKS"),
            default=app_env in {"production", "release"},
        ),
        storage_backend=os.getenv("STORAGE_BACKEND", "json").strip().lower(),
        task_queue_backend=os.getenv("TASK_QUEUE_BACKEND", "inprocess").strip().lower(),
        task_queue_name=os.getenv("TASK_QUEUE_NAME", "analysis_jobs").strip() or "analysis_jobs",
        cors_allow_origins=_to_csv_list(
            os.getenv("CORS_ALLOW_ORIGINS"),
            default=[
                "http://127.0.0.1:3000",
                "http://localhost:3000",
                "http://127.0.0.1:3001",
                "http://localhost:3001",
            ],
        ),
        database_url=os.getenv("DATABASE_URL", "").strip(),
        redis_url=os.getenv("REDIS_URL", "").strip(),
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY", "").strip(),
        dashscope_base_url=os.getenv(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).rstrip("/"),
        dashscope_model=os.getenv("DASHSCOPE_MODEL", "qwen3.6-plus-2026-04-02").strip(),
        dashscope_timeout_seconds=int(os.getenv("DASHSCOPE_TIMEOUT_SECONDS", "120")),
        bailian_enable_llm=_to_bool(os.getenv("BAILIAN_ENABLE_LLM"), default=True),
        dashscope_embedding_model=os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4").strip(),
        dashscope_embedding_dimensions=int(os.getenv("DASHSCOPE_EMBEDDING_DIMENSIONS", "1024")),
        dashscope_embedding_concurrency=max(1, int(os.getenv("DASHSCOPE_EMBEDDING_CONCURRENCY", "3"))),
        local_model_enable=_to_bool(os.getenv("LOCAL_MODEL_ENABLE"), default=True),
        local_transformer_enable=_to_bool(os.getenv("LOCAL_TRANSFORMER_ENABLE"), default=False),
        local_exact_match_enable=_to_bool(os.getenv("LOCAL_EXACT_MATCH_ENABLE"), default=False),
        local_reference_model_enable=_to_bool(os.getenv("LOCAL_REFERENCE_MODEL_ENABLE"), default=True),
        local_sentiment_model=os.getenv(
            "LOCAL_SENTIMENT_MODEL", "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
        ).strip(),
        local_zero_shot_model=os.getenv(
            "LOCAL_ZERO_SHOT_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        ).strip(),
        local_model_batch_size=int(os.getenv("LOCAL_MODEL_BATCH_SIZE", "8")),
        local_model_max_documents=int(os.getenv("LOCAL_MODEL_MAX_DOCUMENTS", "512")),
        local_sentiment_neutral_threshold=float(os.getenv("LOCAL_SENTIMENT_NEUTRAL_THRESHOLD", "0.60")),
        local_zero_shot_hypothesis_template=os.getenv(
            "LOCAL_ZERO_SHOT_HYPOTHESIS_TEMPLATE", "这段文本主要属于{}。"
        ).strip(),
        upload_max_file_bytes=int(os.getenv("UPLOAD_MAX_FILE_BYTES", str(50 * 1024 * 1024))),
        upload_max_rows=int(os.getenv("UPLOAD_MAX_ROWS", "50000")),
        upload_max_columns=int(os.getenv("UPLOAD_MAX_COLUMNS", "128")),
        upload_max_text_length=int(os.getenv("UPLOAD_MAX_TEXT_LENGTH", "20000")),
    )
