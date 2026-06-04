from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


GeneratorKind = Literal["rule", "model", "llm"]
AnalysisStage = Literal["explore", "topics", "discover", "sentiment", "classify", "full"]


class Document(BaseModel):
    id: str
    dataset_id: str
    source_row: int
    title: Optional[str] = None
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Dataset(BaseModel):
    id: str
    owner_key: str = "local"
    name: str
    source_filename: str
    source_type: Literal["upload", "library", "connector"] = "upload"
    source_label: Optional[str] = None
    source_reference: Optional[str] = None
    source_meta: Dict[str, Any] = Field(default_factory=dict)
    language: str = "zh-CN"
    created_at: datetime
    document_count: int
    text_column: str
    labels: List[str] = Field(default_factory=list)
    fingerprint: Optional[str] = None


class SynonymGroup(BaseModel):
    canonical_term: str
    aliases: List[str] = Field(default_factory=list)


class DatasetWorkspace(BaseModel):
    dataset_id: str
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    auto_top_k_terms: int = 25
    custom_terms: List[str] = Field(default_factory=list)
    excluded_terms: List[str] = Field(default_factory=list)
    synonym_groups: List[SynonymGroup] = Field(default_factory=list)
    curated_terms: List[str] = Field(default_factory=list)
    notes: str = ""


class DatasetWorkspacePatch(BaseModel):
    auto_top_k_terms: Optional[int] = Field(default=None, ge=1, le=500)
    custom_terms: Optional[List[str]] = None
    excluded_terms: Optional[List[str]] = None
    synonym_groups: Optional[List[SynonymGroup]] = None
    curated_terms: Optional[List[str]] = None
    notes: Optional[str] = None


class WorkspaceSummary(BaseModel):
    document_count: int = 0
    custom_term_count: int = 0
    excluded_term_count: int = 0
    synonym_group_count: int = 0
    curated_term_count: int = 0
    filtered_unique_terms: int = 0
    selected_term_count: int = 0


class DatasetWorkspaceSnapshot(BaseModel):
    workspace: DatasetWorkspace
    summary: WorkspaceSummary
    top_terms: List[Dict[str, Any]] = Field(default_factory=list)
    tokenized_documents: List[Dict[str, Any]] = Field(default_factory=list)
    selected_terms: List[Dict[str, Any]] = Field(default_factory=list)
    match_rows: List[Dict[str, Any]] = Field(default_factory=list)
    binary_matrix: List[Dict[str, Any]] = Field(default_factory=list)
    frequency_matrix: List[Dict[str, Any]] = Field(default_factory=list)
    cooccurrence_edges: List[Dict[str, Any]] = Field(default_factory=list)


class DatasetWorkspaceOverview(BaseModel):
    workspace: DatasetWorkspace
    summary: WorkspaceSummary


class Span(BaseModel):
    document_id: str
    text: str
    start: int
    end: int
    label: str
    confidence: float
    generator: GeneratorKind


class LexiconEntry(BaseModel):
    term: str
    weight: float = 1.0
    category: Optional[str] = None


class Lexicon(BaseModel):
    id: str
    name: str
    kind: Literal["stopwords", "sentiment", "classification", "synonym"]
    entries: List[LexiconEntry]


class LabelProfile(BaseModel):
    name: str
    description: str = ""
    keywords: List[str] = Field(default_factory=list)
    positive_examples: List[str] = Field(default_factory=list)
    negative_examples: List[str] = Field(default_factory=list)
    source_topic_ids: List[str] = Field(default_factory=list)


class LabelSchema(BaseModel):
    id: str
    name: str
    description: str
    labels: List[str]
    profiles: List[LabelProfile] = Field(default_factory=list)


class Evidence(BaseModel):
    value: str
    confidence: float
    snippet: str
    document_id: str
    generator: GeneratorKind


class InsightCard(BaseModel):
    id: str
    title: str
    summary: str
    kind: Literal["topic", "sentiment", "classification", "entity", "report"]
    evidences: List[Evidence]
    metrics: Dict[str, Any] = Field(default_factory=dict)


class TopicCluster(BaseModel):
    topic_id: str
    name: str
    suggested_name: Optional[str] = None
    name_source: Literal["algorithm", "llm", "user"] = "algorithm"
    size: int
    keywords: List[str]
    summary: str
    evidences: List[Evidence]


class SentimentResult(BaseModel):
    document_id: str
    label: Literal["positive", "neutral", "negative"]
    score: float
    aspect_hits: Dict[str, float] = Field(default_factory=dict)
    evidence: Evidence


class ClassificationResult(BaseModel):
    document_id: str
    label: str
    confidence: float
    evidence: Evidence


class ExportArtifact(BaseModel):
    artifact: str
    format: Literal["csv", "xlsx", "json", "md"]
    path: str
    rows: int


class ExportArtifactSummary(BaseModel):
    run_id: str
    dataset_id: str
    created_at: datetime
    artifact: str
    format: Literal["csv", "xlsx", "json", "md"]
    path: str
    rows: int


class LlmExecution(BaseModel):
    requested: bool
    attempted: bool
    used: bool
    status: Literal["not_requested", "succeeded", "degraded"]
    provider: str = "dashscope"
    model: Optional[str] = None
    message: str
    error_type: Optional[str] = None


class SemanticExecution(BaseModel):
    requested: bool
    attempted: bool
    used: bool
    status: Literal["not_requested", "succeeded", "degraded"]
    provider: str = "dashscope"
    model: Optional[str] = None
    strategy: Optional[str] = None
    message: str
    error_type: Optional[str] = None


class AnalysisOutputs(BaseModel):
    top_terms: List[Dict[str, Any]]
    tokenized_documents: List[Dict[str, Any]]
    selected_terms: List[Dict[str, Any]]
    match_rows: List[Dict[str, Any]]
    binary_matrix: List[Dict[str, Any]]
    frequency_matrix: List[Dict[str, Any]] = Field(default_factory=list)
    cooccurrence_edges: List[Dict[str, Any]]
    sentiment_results: List[SentimentResult]
    classification_results: List[ClassificationResult]
    topics: List[TopicCluster]
    insight_cards: List[InsightCard]
    report_markdown: str
    exports: List[ExportArtifact]
    semantic_execution: SemanticExecution
    llm_execution: LlmExecution


class AnalysisRun(BaseModel):
    id: str
    owner_key: str = "local"
    dataset_id: str
    created_at: datetime
    status: Literal["queued", "running", "completed", "failed"]
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    generator_stack: List[GeneratorKind]
    settings: Dict[str, Any]
    outputs: Optional[AnalysisOutputs] = None
    error: Optional[str] = None


class AnalysisRunSummary(BaseModel):
    id: str
    owner_key: str = "local"
    dataset_id: str
    created_at: datetime
    status: Literal["queued", "running", "completed", "failed"]
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    generator_stack: List[GeneratorKind]
    settings: Dict[str, Any]
    has_outputs: bool
    export_count: int = 0
    error: Optional[str] = None


class ReportJob(BaseModel):
    id: str
    analysis_run_id: str
    created_at: datetime
    markdown: str
    outline: List[str]


class DatasetDetail(BaseModel):
    dataset: Dataset
    documents: List[Document]
    workspace: Optional[DatasetWorkspace] = None


class DatasetLibraryItem(BaseModel):
    id: str
    name: str
    source_filename: str
    source_type: Literal["upload", "library", "connector"] = "upload"
    source_label: Optional[str] = None
    source_reference: Optional[str] = None
    source_meta: Dict[str, Any] = Field(default_factory=dict)
    language: str = "zh-CN"
    created_at: datetime
    document_count: int
    text_column: str
    analysis_count: int = 0
    completed_analysis_count: int = 0
    failed_analysis_count: int = 0
    export_count: int = 0
    last_run_at: Optional[datetime] = None
    last_run_status: Optional[str] = None


class RunAnalysisRequest(BaseModel):
    dataset_id: str
    analysis_stage: AnalysisStage = "explore"
    top_k_terms: int = 25
    topic_count: int = 4
    label_schema: Optional[LabelSchema] = None
    use_llm: bool = False
    smart_topic_names: bool = False
    write_exports: bool = True
    export_xlsx: bool = True


class TopicNamingSuggestion(BaseModel):
    topic_id: str
    name: str
    summary: str = ""
    confidence: float = 0.0


class TopicNamingResponse(BaseModel):
    run: AnalysisRun
    suggestions: List[TopicNamingSuggestion]


class AnalysisOverviewSummary(BaseModel):
    sample_count: int = 0
    topics_count: int = 0
    positive_count: int = 0
    neutral_count: int = 0
    negative_count: int = 0
    dominant_classification: Optional[str] = None
    export_count: int = 0


class AnalysisPreviewSummary(BaseModel):
    top_terms: List[Dict[str, Any]] = Field(default_factory=list)
    topics: List[TopicCluster] = Field(default_factory=list)
    report_markdown: str = ""
    exports: List[ExportArtifact] = Field(default_factory=list)


class AnalysisRunOverview(BaseModel):
    id: str
    dataset_id: str
    status: Literal["queued", "running", "completed", "failed"]
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    generator_stack: List[GeneratorKind]
    settings: Dict[str, Any]
    error: Optional[str] = None
    overview: AnalysisOverviewSummary
    previews: AnalysisPreviewSummary


class AnalysisSectionPage(BaseModel):
    section: str
    page: int
    page_size: int
    total: int
    items: List[Dict[str, Any]] = Field(default_factory=list)


class WorkspaceSectionPage(BaseModel):
    section: str
    page: int
    page_size: int
    total: int
    items: List[Dict[str, Any]] = Field(default_factory=list)


class DataSourceAsset(BaseModel):
    id: str
    name: str
    category: str
    kind: Literal["builtin"]
    description: str
    source_name: str
    source_url: str
    license: str = "public-demo"
    language: str = "zh-CN"
    document_count: int
    recommended_modules: List[str] = Field(default_factory=list)
    fields: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    notes: str = ""


class DataSourceImportRecord(BaseModel):
    dataset_id: str
    dataset_name: str
    source_type: Literal["library", "connector"]
    source_label: Optional[str] = None
    source_reference: Optional[str] = None
    created_at: datetime
    document_count: int


class SourceConnectorItem(BaseModel):
    id: str
    name: str
    description: str
    input_hint: str
    status: Literal["ready", "beta"] = "ready"
    supports_preview: bool = True
    supports_import: bool = True


class OpenAlexPreviewRequest(BaseModel):
    query: str = Field(min_length=2, max_length=120)
    limit: int = Field(default=10, ge=1, le=20)


class OpenAlexPreviewRow(BaseModel):
    external_id: str
    title: str
    abstract: str
    publication_year: Optional[int] = None
    authors: List[str] = Field(default_factory=list)
    source_name: Optional[str] = None
    source_url: Optional[str] = None


class OpenAlexPreviewResponse(BaseModel):
    connector: Literal["openalex"] = "openalex"
    query: str
    total_returned: int
    items: List[OpenAlexPreviewRow] = Field(default_factory=list)


class OpenAlexImportRequest(BaseModel):
    query: str = Field(min_length=2, max_length=120)
    limit: int = Field(default=50, ge=5, le=200)


class HuggingFaceSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=120)
    limit: int = Field(default=8, ge=1, le=20)


class HuggingFaceDatasetCandidate(BaseModel):
    dataset_id: str
    author: Optional[str] = None
    description: str = ""
    likes: int = 0
    downloads: int = 0
    last_modified: Optional[str] = None


class HuggingFaceSearchResponse(BaseModel):
    connector: Literal["huggingface"] = "huggingface"
    query: str
    total_returned: int
    items: List[HuggingFaceDatasetCandidate] = Field(default_factory=list)


class HuggingFacePreviewRequest(BaseModel):
    dataset_id: str = Field(min_length=3, max_length=200)
    limit: int = Field(default=8, ge=1, le=20)


class HuggingFacePreviewRow(BaseModel):
    row_idx: int
    text: str
    title: Optional[str] = None
    label: Optional[str] = None
    raw: Dict[str, Any] = Field(default_factory=dict)


class HuggingFacePreviewResponse(BaseModel):
    connector: Literal["huggingface"] = "huggingface"
    dataset_id: str
    config: str
    split: str
    total_returned: int
    items: List[HuggingFacePreviewRow] = Field(default_factory=list)


class HuggingFaceImportRequest(BaseModel):
    dataset_id: str = Field(min_length=3, max_length=200)
    limit: int = Field(default=100, ge=5, le=300)
