from __future__ import annotations

from datetime import datetime, timezone

from ..models import AnalysisRun, Dataset, Document, RunAnalysisRequest
from ..services.analyze import run_analysis
from ..storage import load_dataset, save_analysis


def build_running_run(run_id: str, request: RunAnalysisRequest, created_at: datetime, started_at: datetime) -> AnalysisRun:
    return AnalysisRun(
        id=run_id,
        dataset_id=request.dataset_id,
        created_at=created_at,
        status="running",
        started_at=started_at,
        finished_at=None,
        generator_stack=[],
        settings={
            "analysis_stage": request.analysis_stage,
            "top_k_terms": request.top_k_terms,
            "topic_count": request.topic_count,
            "use_llm": request.use_llm,
            "smart_topic_names": request.smart_topic_names,
            "write_exports": request.write_exports,
            "export_xlsx": request.export_xlsx,
            "label_schema": request.label_schema.model_dump(mode="json") if request.label_schema else None,
        },
        outputs=None,
        error=None,
    )


def execute_analysis_job(run_id: str, request: RunAnalysisRequest, created_at: datetime) -> None:
    started_at = datetime.now(timezone.utc)
    running = build_running_run(run_id, request, created_at, started_at)
    save_analysis(running)

    try:
        payload = load_dataset(request.dataset_id)
        dataset = Dataset.model_validate(payload["dataset"])
        documents = [Document.model_validate(item) for item in payload["documents"]]
        run, _ = run_analysis(dataset, documents, request, run_id=run_id, created_at=created_at)
        save_analysis(run.model_copy(update={"started_at": started_at, "finished_at": datetime.now(timezone.utc)}))
    except Exception as exc:
        failed = running.model_copy(update={"status": "failed", "finished_at": datetime.now(timezone.utc), "error": str(exc)})
        save_analysis(failed)
