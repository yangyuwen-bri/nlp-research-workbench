from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..models import (
    AnalysisOutputs,
    AnalysisRun,
    ClassificationResult,
    Dataset,
    Document,
    Evidence,
    ExportArtifact,
    InsightCard,
    LabelProfile,
    LlmExecution,
    SemanticExecution,
    ReportJob,
    RunAnalysisRequest,
    SentimentResult,
)
from ..storage import EXPORTS_DIR
from ..storage import load_workspace
from ..settings import get_settings
from .report import build_outline, build_report_markdown
from .llm import enrich_topics_and_report
from .embeddings import embed_texts
from .local_models import LocalModelError, analyze_sentiments_with_local_models, analyze_with_local_models
from .semantic import build_label_semantic_classifications
from .topic_models import build_topics
from .workspace import build_workspace_snapshot, default_workspace

ILLEGAL_EXCEL_CHARS = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")


def _ensure_export_dir(run_id: str) -> Path:
    export_dir = EXPORTS_DIR / run_id
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def _sanitize_excel_value(value: object) -> object:
    if isinstance(value, str):
        return ILLEGAL_EXCEL_CHARS.sub("", value)
    return value


def _write_table(
    export_dir: Path,
    artifact: str,
    rows: List[Dict[str, object]],
    *,
    export_xlsx: bool,
) -> List[ExportArtifact]:
    artifacts: List[ExportArtifact] = []
    frame = pd.DataFrame(rows)
    csv_path = export_dir / f"{artifact}.csv"
    frame.to_csv(csv_path, index=False)
    artifacts.append(ExportArtifact(artifact=artifact, format="csv", path=str(csv_path), rows=len(frame)))
    if export_xlsx:
        xlsx_path = export_dir / f"{artifact}.xlsx"
        frame = frame.map(_sanitize_excel_value) if not frame.empty else frame
        frame.to_excel(xlsx_path, index=False)
        artifacts.append(ExportArtifact(artifact=artifact, format="xlsx", path=str(xlsx_path), rows=len(frame)))
    return artifacts


def _analysis_stage(request: RunAnalysisRequest) -> str:
    return getattr(request, "analysis_stage", "explore")


def _confirmed_labels(request: RunAnalysisRequest) -> List[str]:
    schema = getattr(request, "label_schema", None)
    labels = schema.labels if schema and schema.labels else []
    return [str(label).strip() for label in labels if str(label).strip()]


def _confirmed_label_profiles(request: RunAnalysisRequest) -> List[LabelProfile]:
    schema = getattr(request, "label_schema", None)
    profiles = schema.profiles if schema and schema.profiles else []
    return [profile for profile in profiles if profile.name.strip()]


def _classify_with_confirmed_labels(
    documents: List[Document],
    labels: List[str],
    *,
    settings: object,
    texts: List[str],
    profiles: List[LabelProfile] | None = None,
) -> tuple[List[ClassificationResult], str]:
    if not labels:
        raise LocalModelError("分类阶段需要用户确认后的标签名单。")

    try:
        local_outputs = analyze_with_local_models(
            documents,
            classification_labels=labels,
            settings=settings,  # type: ignore[arg-type]
        )
        return local_outputs.classification_results, local_outputs.classification_strategy
    except LocalModelError:
        if not getattr(settings, "embedding_ready", False):
            raise LocalModelError(
                "分类阶段需要可用的分类模型：请启用本地 zero-shot 模型，或配置 DASHSCOPE_API_KEY 与 embedding 模型接口。"
            )

    document_embeddings = embed_texts(texts)
    return (
        build_label_semantic_classifications(documents, labels, document_embeddings, profiles=profiles),
        "profile-embedding-classification" if profiles else "label-embedding-classification",
    )


def run_analysis(
    dataset: Dataset,
    documents: List[Document],
    request: RunAnalysisRequest,
    *,
    run_id: str | None = None,
    created_at: datetime | None = None,
) -> tuple[AnalysisRun, ReportJob]:
    settings = get_settings()
    run_id = run_id or f"run_{uuid.uuid4().hex[:10]}"
    created_at = created_at or datetime.now(timezone.utc)
    export_dir = _ensure_export_dir(run_id)
    texts = [document.content for document in documents]
    analysis_stage = _analysis_stage(request)
    should_run_topics = analysis_stage in {"topics", "discover", "full"}
    should_run_exploration = analysis_stage in {"explore", "full"}
    should_run_classification = analysis_stage in {"classify", "full"}
    should_run_text_tables = should_run_exploration
    should_run_sentiment = analysis_stage in {"sentiment", "full"}

    try:
        workspace = load_workspace(dataset.id)
    except FileNotFoundError:
        workspace = default_workspace(dataset.id, auto_top_k_terms=request.top_k_terms)

    if should_run_text_tables:
        workspace_snapshot = build_workspace_snapshot(workspace, documents, top_k_terms=request.top_k_terms)
        term_stats = workspace_snapshot.top_terms
        counter: Counter = Counter({str(item["term"]): int(item.get("term_frequency", 0) or 0) for item in term_stats})
        selected = workspace_snapshot.selected_terms
        tokenized_rows = workspace_snapshot.tokenized_documents
        match_rows = workspace_snapshot.match_rows
        binary_matrix = workspace_snapshot.binary_matrix
        frequency_matrix = workspace_snapshot.frequency_matrix
        cooccurrence_edges = workspace_snapshot.cooccurrence_edges
    else:
        term_stats = []
        counter: Counter = Counter()
        selected = []
        tokenized_rows = []
        match_rows = []
        binary_matrix = []
        frequency_matrix = []
        cooccurrence_edges = []

    sentiment_results: List[SentimentResult] = []
    classification_results: List[ClassificationResult] = []
    sentiment_counter: Counter = Counter()
    class_counter: Counter = Counter()
    local_outputs = None
    local_sentiment_outputs = None
    classification_strategy_name = "not_requested"
    sentiment_strategy_name = "not_requested"
    classification_error: str | None = None
    semantic_execution = SemanticExecution(
        requested=request.use_llm or settings.local_model_ready,
        attempted=False,
        used=False,
        status="not_requested" if not (request.use_llm or settings.local_model_ready) else "degraded",
        model=settings.local_sentiment_model if settings.local_model_ready else settings.dashscope_model,
        strategy="local-transformers -> llm-document-analysis -> embedding-prototype -> rule",
        message=(
            "本地 transformer 模型未启用，情感/分类将回退到传统路径。"
            if not (request.use_llm or settings.local_model_ready)
            else "更强语义分析路径尚未执行。"
        ),
    )
    classification_labels = _confirmed_labels(request)
    classification_profiles = _confirmed_label_profiles(request)
    if analysis_stage == "classify" and not classification_labels:
        raise ValueError("分类阶段需要用户确认后的标签名单。")

    if should_run_classification:
        try:
            classification_results, classification_strategy_name = _classify_with_confirmed_labels(
                documents,
                classification_labels,
                settings=settings,
                texts=texts,
                profiles=classification_profiles,
            )
            for result in classification_results:
                class_counter[result.label] += 1
            semantic_execution = SemanticExecution(
                requested=True,
                attempted=True,
                used=True,
                status="succeeded",
                model=(
                    settings.local_zero_shot_model
                    if classification_strategy_name == "local-zero-shot-classification"
                    else settings.dashscope_embedding_model
                ),
                strategy=classification_strategy_name,
                message="已按用户确认的标签名单重新运行分类。",
            )
        except LocalModelError as exc:
            classification_error = str(exc)
            semantic_execution = SemanticExecution(
                requested=request.use_llm or settings.local_model_ready,
                attempted=settings.local_model_ready,
                used=False,
                status="degraded" if (request.use_llm or settings.local_model_ready) else "not_requested",
                model=settings.local_sentiment_model if settings.local_model_ready else None,
                strategy="local-transformers -> llm-document-analysis -> embedding-prototype -> rule",
                message=f"本地 transformer 模型未能生效，准备回退：{exc}",
                error_type=type(exc).__name__,
            )
            if analysis_stage == "classify":
                raise

    if should_run_sentiment and not sentiment_results:
        try:
            local_sentiment_outputs = analyze_sentiments_with_local_models(documents, settings=settings)
            sentiment_results = local_sentiment_outputs.sentiment_results
            sentiment_strategy_name = local_sentiment_outputs.strategy
            for result in sentiment_results:
                sentiment_counter[result.label] += 1
            semantic_execution = SemanticExecution(
                requested=True,
                attempted=True,
                used=True,
                status="succeeded",
                model=settings.local_sentiment_model,
                strategy=sentiment_strategy_name,
                message=local_sentiment_outputs.message,
            )
        except LocalModelError as exc:
            semantic_execution = SemanticExecution(
                requested=True,
                attempted=True,
                used=False,
                status="degraded",
                model=settings.local_sentiment_model,
                strategy="strict-production-sentiment",
                message=f"情感分析主链不可用：{exc}",
                error_type=type(exc).__name__,
            )
            raise

    if should_run_sentiment and not sentiment_results:
        raise LocalModelError("情感分析未返回有效结果。")

    if should_run_classification and not classification_results:
        semantic_execution = SemanticExecution(
            requested=request.use_llm or settings.local_model_ready,
            attempted=True,
            used=bool(sentiment_results),
            status="degraded",
            model=semantic_execution.model,
            strategy=semantic_execution.strategy,
            message=(
                classification_error
                or "当前生产分类模式未返回结果。已禁用规则与语义相似度兜底，分类模块本次不输出结果。"
            ),
            error_type="ClassificationUnsupported",
        )

    topics, topic_strategy = (
        build_topics(documents, request.topic_count, allow_embeddings=settings.embedding_ready)
        if should_run_topics
        else ([], "not_requested")
    )
    insight_cards = []
    if should_run_sentiment:
        insight_cards.append(
            InsightCard(
                id=f"card_sentiment_{run_id}",
                title="情感结构",
                summary=f"正向 {sentiment_counter['positive']} 条，负向 {sentiment_counter['negative']} 条，中性 {sentiment_counter['neutral']} 条。",
                kind="sentiment",
                evidences=[item.evidence for item in sentiment_results[:3]],
                metrics=dict(sentiment_counter),
            )
        )
    if should_run_classification:
        insight_cards.append(
            InsightCard(
                id=f"card_class_{run_id}",
                title="分类结构",
                summary=f"最多的分类是 {class_counter.most_common(1)[0][0] if class_counter else '暂无分类结果'}。",
                kind="classification",
                evidences=[item.evidence for item in classification_results[:3]],
                metrics=dict(class_counter),
            )
        )
    for topic in topics[:3]:
        insight_cards.append(
            InsightCard(
                id=f"card_{topic.topic_id}",
                title=f"主题：{topic.name}",
                summary=topic.summary,
                kind="topic",
                evidences=topic.evidences,
                metrics={"size": topic.size, "keywords": topic.keywords[:5]},
            )
        )

    report_markdown = build_report_markdown(dataset.name, topics, sentiment_results, classification_results, insight_cards)
    llm_used = False
    smart_topic_names = bool(getattr(request, "smart_topic_names", False))
    llm_requested = bool(request.use_llm and smart_topic_names)
    llm_execution = LlmExecution(
        requested=llm_requested,
        attempted=False,
        used=False,
        status="not_requested" if not llm_requested else "degraded",
        model=settings.dashscope_model,
        message="未请求智能主题命名。" if not llm_requested else "智能主题命名尚未执行。",
    )
    if llm_requested:
        llm_execution = LlmExecution(
            requested=True,
            attempted=True,
            used=False,
            status="degraded",
            model=settings.dashscope_model,
                message="智能主题命名已请求，但尚未返回有效结果。",
        )
        try:
            llm_result = enrich_topics_and_report(
                dataset_name=dataset.name,
                top_terms=term_stats,
                topics=topics,
                insight_cards=insight_cards,
                sentiment_summary=dict(sentiment_counter),
                classification_summary=dict(class_counter),
            )
        except (BailianClientError, httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            llm_result = None
            llm_execution = LlmExecution(
                requested=True,
                attempted=True,
                used=False,
                status="degraded",
                model=settings.dashscope_model,
                message=f"智能主题命名失败，保留算法主题结果：{exc}",
                error_type=type(exc).__name__,
            )
        if llm_result:
            override_map = {item["topic_id"]: item for item in llm_result.get("topic_overrides", []) if item.get("topic_id")}
            topics = [
                topic.model_copy(
                    update={
                        "name": override_map.get(topic.topic_id, {}).get("name", topic.name),
                        "summary": override_map.get(topic.topic_id, {}).get("summary", topic.summary),
                    }
                )
                for topic in topics
            ]
            topic_map = {topic.topic_id: topic for topic in topics}
            for index, item in enumerate(llm_result.get("insight_cards", [])[:3], start=1):
                topic = topic_map.get(item.get("topic_id", ""))
                if not topic:
                    continue
                insight_cards.append(
                    InsightCard(
                        id=f"card_llm_{run_id}_{index}",
                        title=item.get("title", "LLM 洞察"),
                        summary=item.get("summary", topic.summary),
                        kind="report",
                        evidences=topic.evidences,
                        metrics={"topic_id": topic.topic_id, "generator": "llm"},
                    )
                )
            report_markdown = llm_result.get("report_markdown", report_markdown)
            llm_used = True
            llm_execution = LlmExecution(
                requested=True,
                attempted=True,
                used=True,
                status="succeeded",
                model=settings.dashscope_model,
                message="智能主题命名已完成。",
            )
    report_id = f"report_{uuid.uuid4().hex[:10]}"
    report_job = ReportJob(
        id=report_id,
        analysis_run_id=run_id,
        created_at=datetime.now(timezone.utc),
        markdown=report_markdown,
        outline=build_outline(insight_cards),
    )

    exports: List[ExportArtifact] = []
    write_exports = getattr(request, "write_exports", True)
    export_xlsx = getattr(request, "export_xlsx", True)
    if write_exports:
        if should_run_text_tables:
            exports += _write_table(export_dir, "term_frequency", term_stats, export_xlsx=export_xlsx)
            exports += _write_table(export_dir, "tokenized_documents", tokenized_rows, export_xlsx=export_xlsx)
            exports += _write_table(export_dir, "selected_terms", selected, export_xlsx=export_xlsx)
            exports += _write_table(export_dir, "match_rows", match_rows, export_xlsx=export_xlsx)
            exports += _write_table(export_dir, "binary_matrix", binary_matrix, export_xlsx=export_xlsx)
            exports += _write_table(export_dir, "frequency_matrix", frequency_matrix, export_xlsx=export_xlsx)
            exports += _write_table(export_dir, "cooccurrence_edges", cooccurrence_edges, export_xlsx=export_xlsx)
        if should_run_sentiment:
            exports += _write_table(
                export_dir,
                "sentiment_results",
                [
                    {
                        "document_id": result.document_id,
                        "label": result.label,
                        "score": result.score,
                        "evidence": result.evidence.snippet,
                        **result.aspect_hits,
                    }
                    for result in sentiment_results
                ],
                export_xlsx=export_xlsx,
            )
        if should_run_classification:
            exports += _write_table(
                export_dir,
                "classification_results",
                [
                    {
                        "document_id": result.document_id,
                        "label": result.label,
                        "confidence": result.confidence,
                        "evidence": result.evidence.snippet,
                    }
                    for result in classification_results
                ],
                export_xlsx=export_xlsx,
            )
        if should_run_topics:
            exports += _write_table(
                export_dir,
                "topic_clusters",
                [
                    {
                        "topic_id": topic.topic_id,
                        "name": topic.name,
                        "size": topic.size,
                        "keywords": ", ".join(topic.keywords),
                        "summary": topic.summary,
                    }
                    for topic in topics
                ],
                export_xlsx=export_xlsx,
            )
        report_path = export_dir / "report.md"
        report_path.write_text(report_markdown, encoding="utf-8")
        exports.append(ExportArtifact(artifact="report", format="md", path=str(report_path), rows=1))

    outputs = AnalysisOutputs(
        top_terms=term_stats,
        tokenized_documents=tokenized_rows,
        selected_terms=selected,
        match_rows=match_rows,
        binary_matrix=binary_matrix,
        frequency_matrix=frequency_matrix,
        cooccurrence_edges=cooccurrence_edges,
        sentiment_results=sentiment_results,
        classification_results=classification_results,
        topics=topics,
        insight_cards=insight_cards,
        report_markdown=report_markdown,
        exports=exports,
        semantic_execution=semantic_execution,
        llm_execution=llm_execution,
    )
    run = AnalysisRun(
        id=run_id,
        owner_key=dataset.owner_key,
        dataset_id=dataset.id,
        created_at=created_at,
        status="completed",
        generator_stack=["rule", "model", *(["llm"] if llm_used else [])],
        settings={
            "analysis_stage": analysis_stage,
            "top_k_terms": request.top_k_terms,
            "topic_count": request.topic_count,
            "raw_unique_terms": len(counter),
            "workspace": workspace.model_dump(mode="json"),
            "workspace_curated_terms": len(workspace.curated_terms),
            "workspace_custom_terms": len(workspace.custom_terms),
            "workspace_synonym_groups": len(workspace.synonym_groups),
            "use_llm": request.use_llm,
            "smart_topic_names": smart_topic_names,
            "write_exports": write_exports,
            "export_xlsx": export_xlsx,
            "topic_strategy": topic_strategy,
            "sentiment_strategy": (
                "not_requested"
                if not should_run_sentiment
                else local_outputs.sentiment_strategy
                if semantic_execution.used and local_outputs is not None
                else sentiment_strategy_name
            ),
            "classification_strategy": (
                classification_strategy_name
                if should_run_classification and classification_results
                else "unsupported" if should_run_classification
                else "not_requested"
            ),
        },
        outputs=outputs,
    )
    return run, report_job
