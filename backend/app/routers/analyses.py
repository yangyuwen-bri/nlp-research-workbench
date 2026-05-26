from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, HTTPException, Query

from ..models import (
    AnalysisOverviewSummary,
    AnalysisPreviewSummary,
    AnalysisRun,
    AnalysisRunOverview,
    AnalysisRunSummary,
    AnalysisSectionPage,
    LabelSchema,
    RunAnalysisRequest,
    TopicNamingResponse,
    TopicNamingSuggestion,
)
from ..services.llm import BailianClientError, name_topic_clusters
from ..services.analysis_jobs import execute_analysis_job
from ..storage import list_analyses, load_analysis, load_dataset, save_analysis
from ..task_queue import enqueue_analysis_job


router = APIRouter(prefix="/analyses", tags=["analyses"])


@router.get("", response_model=List[AnalysisRunSummary])
def get_analyses(dataset_id: Optional[str] = None):
    return list_analyses(dataset_id=dataset_id)


def _build_pending_run(request: RunAnalysisRequest) -> AnalysisRun:
    return AnalysisRun(
        id=f"run_{uuid.uuid4().hex[:10]}",
        dataset_id=request.dataset_id,
        created_at=datetime.now(timezone.utc),
        status="queued",
        started_at=None,
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

def _enqueue_analysis_run(run: AnalysisRun, request: RunAnalysisRequest) -> None:
    enqueue_analysis_job(
        run_id=run.id,
        request=request,
        created_at=run.created_at,
        handler=execute_analysis_job,
    )


@router.post("/run")
def create_analysis(request: RunAnalysisRequest):
    try:
        load_dataset(request.dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset not found") from exc

    existing_runs = list_analyses(dataset_id=request.dataset_id)
    active_run = next((item for item in existing_runs if item.status in {"queued", "running"}), None)
    if active_run:
        raise HTTPException(
            status_code=409,
            detail=f"当前数据集已有任务正在运行，请等待完成后再试。任务ID：{active_run.id}",
        )

    run = _build_pending_run(request)
    save_analysis(run)
    _enqueue_analysis_run(run, request)
    return {"run": run}


def _request_from_run(run: AnalysisRun) -> RunAnalysisRequest:
    settings = run.settings or {}
    return RunAnalysisRequest(
        dataset_id=run.dataset_id,
        analysis_stage=settings.get("analysis_stage", "discover"),
        top_k_terms=int(settings.get("top_k_terms", 25)),
        topic_count=int(settings.get("topic_count", 4)),
        label_schema=LabelSchema.model_validate(settings["label_schema"]) if settings.get("label_schema") else None,
        use_llm=bool(settings.get("use_llm", False)),
        smart_topic_names=bool(settings.get("smart_topic_names", False)),
        write_exports=bool(settings.get("write_exports", True)),
        export_xlsx=bool(settings.get("export_xlsx", True)),
    )


@router.post("/{run_id}/retry")
def retry_analysis(run_id: str):
    try:
        original = load_analysis(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Analysis not found") from exc

    try:
        load_dataset(original.dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset not found") from exc

    request = _request_from_run(original)
    run = _build_pending_run(request)
    save_analysis(run)
    _enqueue_analysis_run(run, request)
    return {"run": run}


@router.get("/{run_id}")
def get_analysis(run_id: str):
    try:
        return load_analysis(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Analysis not found") from exc


def _build_analysis_overview(run: AnalysisRun) -> AnalysisRunOverview:
    outputs = run.outputs
    dominant_classification: str | None = None
    sample_count = 0
    if outputs and outputs.classification_results:
        counts = Counter(item.label for item in outputs.classification_results)
        dominant_classification = counts.most_common(1)[0][0]
    if outputs:
        sample_count = (
            len(outputs.sentiment_results)
            or len(outputs.classification_results)
            or len(outputs.tokenized_documents)
            or sum(topic.size for topic in outputs.topics)
        )

    overview = AnalysisOverviewSummary(
        sample_count=sample_count,
        topics_count=len(outputs.topics) if outputs else 0,
        positive_count=sum(1 for item in outputs.sentiment_results if item.label == "positive") if outputs else 0,
        neutral_count=sum(1 for item in outputs.sentiment_results if item.label == "neutral") if outputs else 0,
        negative_count=sum(1 for item in outputs.sentiment_results if item.label == "negative") if outputs else 0,
        dominant_classification=dominant_classification,
        export_count=len(outputs.exports) if outputs else 0,
    )
    previews = AnalysisPreviewSummary(
        top_terms=outputs.top_terms[:10] if outputs else [],
        topics=outputs.topics if outputs else [],
        report_markdown=outputs.report_markdown if outputs else "",
        exports=outputs.exports if outputs else [],
    )
    return AnalysisRunOverview(
        id=run.id,
        dataset_id=run.dataset_id,
        status=run.status,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        generator_stack=run.generator_stack,
        settings=run.settings,
        error=run.error,
        overview=overview,
        previews=previews,
    )


@router.get("/{run_id}/summary", response_model=AnalysisRunOverview)
def get_analysis_summary(run_id: str):
    try:
        run = load_analysis(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Analysis not found") from exc
    return _build_analysis_overview(run)


@router.post("/{run_id}/topics/name", response_model=TopicNamingResponse)
def name_analysis_topics(run_id: str):
    try:
        run = load_analysis(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Analysis not found") from exc
    if not run.outputs or not run.outputs.topics:
        raise HTTPException(status_code=400, detail="当前分析没有可命名的主题。")

    try:
        dataset_payload = load_dataset(run.dataset_id)
        dataset_name = dataset_payload["dataset"]["name"]
    except (FileNotFoundError, KeyError, TypeError):
        dataset_name = run.dataset_id

    try:
        raw_suggestions = name_topic_clusters(dataset_name=dataset_name, topics=run.outputs.topics)
    except BailianClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    suggestions = [
        TopicNamingSuggestion(
            topic_id=str(item.get("topic_id", "")),
            name=str(item.get("name", "")).strip(),
            summary=str(item.get("summary", "")).strip(),
            confidence=float(item.get("confidence", 0.0) or 0.0),
        )
        for item in raw_suggestions
        if str(item.get("topic_id", "")).strip() and str(item.get("name", "")).strip()
    ]
    suggestion_map = {item.topic_id: item for item in suggestions}
    updated_topics = [
        topic.model_copy(
            update={
                "name": suggestion_map[topic.topic_id].name,
                "suggested_name": suggestion_map[topic.topic_id].name,
                "summary": suggestion_map[topic.topic_id].summary or topic.summary,
                "name_source": "llm",
            }
        )
        if topic.topic_id in suggestion_map
        else topic
        for topic in run.outputs.topics
    ]
    updated_outputs = run.outputs.model_copy(update={"topics": updated_topics})
    updated_run = run.model_copy(update={"outputs": updated_outputs})
    save_analysis(updated_run)
    return TopicNamingResponse(run=updated_run, suggestions=suggestions)


def _serialize_section_items(run: AnalysisRun, section: str) -> List[Dict[str, Any]]:
    if not run.outputs:
        return []
    outputs = run.outputs
    if section == "tokenized":
        return outputs.tokenized_documents
    if section == "terms":
        return outputs.top_terms
    if section == "selected":
        return outputs.selected_terms
    if section == "matches":
        return outputs.match_rows
    if section == "matrix":
        return outputs.binary_matrix
    if section == "frequency_matrix":
        return outputs.frequency_matrix
    if section == "cooccurrence":
        return outputs.cooccurrence_edges
    if section == "sentiment":
        return [
            {
                "document_id": item.document_id,
                "label": item.label,
                "score": round(item.score, 4),
                "snippet": item.evidence.snippet,
                "source": item.evidence.generator,
            }
            for item in outputs.sentiment_results
        ]
    if section == "classification":
        return [
            {
                "document_id": item.document_id,
                "label": item.label,
                "confidence": round(item.confidence, 4),
                "snippet": item.evidence.snippet,
                "source": item.evidence.generator,
            }
            for item in outputs.classification_results
        ]
    raise HTTPException(status_code=404, detail="Section not found")


@router.get("/{run_id}/sections/{section}", response_model=AnalysisSectionPage)
def get_analysis_section(
    run_id: str,
    section: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
):
    try:
        run = load_analysis(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Analysis not found") from exc

    items = _serialize_section_items(run, section)
    start = (page - 1) * page_size
    end = start + page_size
    return AnalysisSectionPage(
        section=section,
        page=page,
        page_size=page_size,
        total=len(items),
        items=items[start:end],
    )
