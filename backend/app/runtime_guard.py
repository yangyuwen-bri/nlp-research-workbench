from __future__ import annotations

from dataclasses import asdict, dataclass

from .settings import Settings, get_settings
from .storage import check_storage_connection
from .task_queue import check_task_queue_connection


@dataclass(frozen=True)
class RuntimeCheck:
    id: str
    label: str
    status: str
    message: str


def evaluate_runtime_readiness(settings: Settings | None = None) -> dict[str, object]:
    runtime = settings or get_settings()
    checks: list[RuntimeCheck] = []

    checks.append(
        RuntimeCheck(
            id="storage_backend",
            label="持久化存储",
            status="passed" if runtime.storage_backend != "json" else "failed",
            message=(
                f"当前为 {runtime.storage_backend}。"
                if runtime.storage_backend != "json"
                else "当前仍是 json，本地文件存储不适合生产环境。"
            ),
        )
    )
    if runtime.storage_backend != "json":
        checks.append(
            RuntimeCheck(
                id="database_url",
                label="数据库连接",
                status="passed" if bool(runtime.database_url) else "failed",
                message=runtime.database_url or "未配置 DATABASE_URL。",
            )
        )
        storage_ready, storage_message = check_storage_connection()
        checks.append(
            RuntimeCheck(
                id="storage_connection",
                label="存储连通性",
                status="passed" if storage_ready else "failed",
                message=storage_message,
            )
        )

    checks.append(
        RuntimeCheck(
            id="task_queue_backend",
            label="异步任务队列",
            status="passed" if runtime.task_queue_backend != "inprocess" else "failed",
            message=(
                f"当前为 {runtime.task_queue_backend}。"
                if runtime.task_queue_backend != "inprocess"
                else "当前仍是 inprocess，进程内线程不适合生产环境。"
            ),
        )
    )
    if runtime.task_queue_backend != "inprocess":
        checks.append(
            RuntimeCheck(
                id="redis_url",
                label="队列连接",
                status="passed" if bool(runtime.redis_url) else "failed",
                message=runtime.redis_url or "未配置 REDIS_URL。",
            )
        )
        queue_ready, queue_message = check_task_queue_connection()
        checks.append(
            RuntimeCheck(
                id="queue_connection",
                label="队列连通性",
                status="passed" if queue_ready else "failed",
                message=queue_message,
            )
        )

    wildcard = any(origin == "*" for origin in runtime.cors_allow_origins)
    checks.append(
        RuntimeCheck(
            id="cors_allow_origins",
            label="跨域白名单",
            status="passed" if not wildcard else "failed",
            message=(
                ",".join(runtime.cors_allow_origins)
                if not wildcard
                else "CORS 仍包含 * ，需要改成显式白名单。"
            ),
        )
    )
    checks.append(
        RuntimeCheck(
            id="upload_guards",
            label="上传保护",
            status="passed"
            if runtime.upload_max_file_bytes > 0 and runtime.upload_max_rows > 0 and runtime.upload_max_text_length > 0
            else "failed",
            message=(
                f"文件 {runtime.upload_max_file_bytes} bytes, 行数 {runtime.upload_max_rows}, 文本 {runtime.upload_max_text_length} chars"
            ),
        )
    )

    failed = [check for check in checks if check.status != "passed"]
    return {
        "ready": len(failed) == 0,
        "checks": [asdict(check) for check in checks],
        "summary": {
            "passed": len(checks) - len(failed),
            "failed": len(failed),
        },
        "recommendation": "平台已满足生产环境基础要求。" if not failed else "平台仍未达到生产环境要求，应先修复失败项。",
    }


def assert_runtime_ready_for_startup(settings: Settings | None = None) -> None:
    runtime = settings or get_settings()
    if not runtime.is_production or not runtime.strict_startup_checks:
        return
    evaluation = evaluate_runtime_readiness(runtime)
    if evaluation["ready"]:
        return
    failed_checks = [item for item in evaluation["checks"] if item["status"] != "passed"]
    failures = "; ".join(f"{item['id']}: {item['message']}" for item in failed_checks)
    raise RuntimeError(f"Production startup blocked by runtime readiness checks: {failures}")
