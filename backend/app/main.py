from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import analyses, datasets, exports
from .runtime_guard import assert_runtime_ready_for_startup, evaluate_runtime_readiness
from .settings import get_settings
from .storage import check_storage_connection
from .task_queue import check_task_queue_connection


settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    assert_runtime_ready_for_startup(settings)
    yield


app = FastAPI(
    title="Chinese AI Text Analysis Workbench",
    version="0.1.0",
    description="A GooSeeker-inspired Chinese research analysis platform with AI-native insight workflows.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router, prefix="/api")
app.include_router(analyses.router, prefix="/api")
app.include_router(exports.router, prefix="/api")


@app.get("/api/health")
def healthcheck():
    storage_ready, storage_message = check_storage_connection()
    task_queue_ready, task_queue_message = check_task_queue_connection()
    overall_ok = storage_ready and task_queue_ready
    return {
        "status": "ok" if overall_ok else "degraded",
        "app_env": settings.app_env,
        "strict_startup_checks": settings.strict_startup_checks,
        "llm_ready": settings.llm_ready,
        "dashscope_model": settings.dashscope_model,
        "task_queue_backend": settings.task_queue_backend,
        "storage_backend": settings.storage_backend,
        "storage_ready": storage_ready,
        "storage_message": storage_message,
        "task_queue_ready": task_queue_ready,
        "task_queue_message": task_queue_message,
    }


@app.get("/api/platform/readiness")
def platform_readiness():
    return evaluate_runtime_readiness(settings)
