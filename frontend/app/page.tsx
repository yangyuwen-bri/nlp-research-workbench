"use client";

import { useDeferredValue, useEffect, useMemo, useRef, useState, useTransition } from "react";

type Dataset = {
  id: string;
  name: string;
  document_count: number;
  source_filename: string;
  created_at: string;
  text_column?: string;
  analysis_count?: number;
  completed_analysis_count?: number;
  failed_analysis_count?: number;
  export_count?: number;
  last_run_at?: string | null;
  last_run_status?: string | null;
};

type DatasetDocument = {
  id: string;
  source_row: number;
  title?: string | null;
  content: string;
};

type DatasetWorkspace = {
  dataset_id: string;
  updated_at: string;
  auto_top_k_terms: number;
  custom_terms: string[];
  excluded_terms: string[];
  synonym_groups: SynonymGroup[];
  curated_terms: string[];
  notes: string;
};

type SynonymGroup = {
  canonical_term: string;
  aliases: string[];
};

type WorkspaceSummary = {
  document_count: number;
  custom_term_count: number;
  excluded_term_count: number;
  synonym_group_count: number;
  curated_term_count: number;
  filtered_unique_terms: number;
  selected_term_count: number;
};

type WorkspaceOverview = {
  workspace: DatasetWorkspace;
  summary: WorkspaceSummary;
};

type DatasetWorkspaceSnapshot = WorkspaceOverview & {
  top_terms: TableRow[];
  tokenized_documents: TableRow[];
  selected_terms: TableRow[];
  match_rows: TableRow[];
  binary_matrix: TableRow[];
  frequency_matrix: TableRow[];
  cooccurrence_edges: TableRow[];
};

type DatasetDetail = {
  dataset: Dataset;
  documents: DatasetDocument[];
  workspace?: DatasetWorkspace | null;
};

type Evidence = {
  snippet: string;
  document_id: string;
  confidence: number;
  generator: "rule" | "model" | "llm";
  value: string;
};

type Topic = {
  topic_id: string;
  name: string;
  suggested_name?: string | null;
  name_source?: "algorithm" | "llm" | "user";
  size: number;
  keywords: string[];
  summary: string;
  evidences: Evidence[];
};

type ExportArtifact = {
  artifact: string;
  format: "csv" | "xlsx" | "json" | "md";
  path: string;
  rows: number;
};

type ExportArtifactSummary = ExportArtifact & {
  run_id: string;
  dataset_id: string;
  created_at: string;
};

type TableRow = Record<string, string | number | boolean | null>;

type AnalysisOverviewSummary = {
  sample_count: number;
  topics_count: number;
  positive_count: number;
  neutral_count: number;
  negative_count: number;
  dominant_classification?: string | null;
  export_count: number;
};

type AnalysisRun = {
  id: string;
  dataset_id: string;
  status: string;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  error?: string | null;
};

type AnalysisRunSummary = {
  id: string;
  dataset_id: string;
  created_at: string;
  status: string;
  started_at?: string | null;
  finished_at?: string | null;
  generator_stack: string[];
  has_outputs: boolean;
  export_count: number;
  error?: string | null;
};

type AnalysisPreviewSummary = {
  top_terms: TableRow[];
  topics: Topic[];
  report_markdown: string;
  exports: ExportArtifact[];
};

type AnalysisRunOverview = AnalysisRun & {
  generator_stack: string[];
  settings: Record<string, unknown>;
  overview: AnalysisOverviewSummary;
  previews: AnalysisPreviewSummary;
};

type AnalysisSectionPage = {
  section: string;
  page: number;
  page_size: number;
  total: number;
  items: TableRow[];
};

type WorkspaceSectionPage = {
  section: string;
  page: number;
  page_size: number;
  total: number;
  items: TableRow[];
};

type TabId =
  | "dataset"
  | "overview"
  | "workspace"
  | "tokenized"
  | "terms"
  | "selected"
  | "matches"
  | "matrix"
  | "frequency"
  | "cooccurrence"
  | "topics"
  | "sentiment"
  | "classification"
  | "report"
  | "exports";

type AnalysisStage = "explore" | "topics" | "discover" | "sentiment" | "classify" | "full";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
const PAGE_SIZE = 50;

const moduleGroups: Array<{ label: string; ids: TabId[] }> = [
  { label: "语料", ids: ["dataset", "overview"] },
  { label: "工作台", ids: ["workspace", "tokenized", "terms", "selected", "matches", "matrix", "frequency", "cooccurrence"] },
  { label: "高级分析", ids: ["topics", "sentiment", "classification", "report", "exports"] },
];

const moduleLabels: Record<TabId, string> = {
  dataset: "数据预览",
  overview: "总览",
  workspace: "词项工作台",
  tokenized: "分词结果",
  terms: "词频",
  selected: "选词结果",
  matches: "匹配表",
  matrix: "二值矩阵",
  frequency: "频次矩阵",
  cooccurrence: "共词关系",
  topics: "主题分析",
  sentiment: "情感分析",
  classification: "分类结果",
  report: "研究报告",
  exports: "导出文件",
};

const tableModuleKeys: TabId[] = ["tokenized", "terms", "selected", "matches", "matrix", "frequency", "cooccurrence"];
const workspaceSectionMap: Record<TabId, string> = {
  tokenized: "tokenized_documents",
  terms: "top_terms",
  selected: "selected_terms",
  matches: "match_rows",
  matrix: "binary_matrix",
  frequency: "frequency_matrix",
  cooccurrence: "cooccurrence_edges",
  dataset: "tokenized_documents",
  overview: "selected_terms",
  workspace: "top_terms",
  topics: "top_terms",
  sentiment: "top_terms",
  classification: "top_terms",
  report: "top_terms",
  exports: "top_terms",
};
const analysisSectionMap: Partial<Record<TabId, string>> = {
  sentiment: "sentiment",
  classification: "classification",
};

const artifactLabels: Record<string, string> = {
  term_frequency: "词频",
  tokenized_documents: "分词",
  selected_terms: "选词",
  match_rows: "匹配",
  binary_matrix: "二值矩阵",
  frequency_matrix: "频次矩阵",
  cooccurrence_edges: "共词",
  sentiment_results: "情感",
  classification_results: "分类",
  topic_clusters: "主题",
  report: "报告",
};

const columnLabels: Record<string, string> = {
  row_id: "序号",
  source_row: "原始行号",
  document_id: "文档ID",
  content: "原文",
  tokens: "分词结果",
  term: "词语",
  term_frequency: "词频",
  document_frequency: "文档数",
  selection_source: "来源",
  selection_rank: "顺序",
  matched_terms: "命中词",
  source: "来源",
  target: "目标",
  weight: "强度",
  label: "结果",
  score: "得分",
  confidence: "置信度",
  snippet: "证据文本",
  artifact: "内容",
  format: "格式",
  rows: "行数",
  path: "文件路径",
};

function stringifyValue(value: unknown) {
  if (value === null || value === undefined) return "";
  if (Array.isArray(value)) return value.join(" / ");
  if (typeof value === "boolean") return value ? "1" : "0";
  return String(value);
}

function filterRows<T extends Record<string, unknown>>(rows: T[], query: string) {
  const normalized = query.trim().toLowerCase();
  if (!normalized) return rows;
  return rows.filter((row) =>
    Object.values(row).some((value) => stringifyValue(value).toLowerCase().includes(normalized)),
  );
}

function columnLabel(column: string) {
  return columnLabels[column] ?? column;
}

function artifactLabel(artifact: string) {
  return artifactLabels[artifact] ?? artifact;
}

function formatDateTime(value?: string | null) {
  if (!value) return "-";
  const normalized = /(?:Z|[+-]\d{2}:\d{2})$/.test(value) ? value : `${value}Z`;
  return new Date(normalized).toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatRunStatus(status?: string | null) {
  if (status === "queued") return "排队中";
  if (status === "running") return "分析中";
  if (status === "completed") return "已完成";
  if (status === "failed") return "失败";
  return status || "-";
}

function exportUrl(runId: string, artifact: ExportArtifact) {
  return `${API_BASE_URL}/api/exports/${runId}/${artifact.artifact}/${artifact.format}`;
}

function sortExports(exports: ExportArtifact[]) {
  return [...exports].sort((left, right) => {
    if (left.artifact === right.artifact) return left.format.localeCompare(right.format);
    return artifactLabel(left.artifact).localeCompare(artifactLabel(right.artifact), "zh-Hans-CN");
  });
}

function renderGenericTable(
  rows: TableRow[],
  emptyLabel: string,
  page: number,
  totalRows: number,
  loading: boolean,
  onPageChange: (page: number) => void,
) {
  if (loading) return <p className="empty-state">正在加载...</p>;
  const columns = rows[0] ? Object.keys(rows[0]) : [];
  if (!rows.length) return <p className="empty-state">{emptyLabel}</p>;
  const totalPages = Math.max(1, Math.ceil(totalRows / PAGE_SIZE));
  const currentPage = Math.min(page, totalPages);
  return (
    <div className="table-stack">
      <div className="table-shell">
        <table className="data-table">
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column}>{columnLabel(column)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={`${rowIndex}-${columns[0] ?? "row"}`}>
                {columns.map((column) => (
                  <td key={`${rowIndex}-${column}`}>{stringifyValue(row[column])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="table-footer">
        <span>
          第 {currentPage} / {totalPages} 页
        </span>
        <span>{totalRows} 条</span>
        <div className="table-pager">
          <button className="mini-button" disabled={currentPage <= 1} onClick={() => onPageChange(currentPage - 1)} type="button">
            上一页
          </button>
          <button
            className="mini-button"
            disabled={currentPage >= totalPages}
            onClick={() => onPageChange(currentPage + 1)}
            type="button"
          >
            下一页
          </button>
        </div>
      </div>
    </div>
  );
}

async function parseJson<T>(response: Response): Promise<T> {
  return (await response.json()) as T;
}

export default function Home() {
  const uploadFormRef = useRef<HTMLFormElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [datasetDetail, setDatasetDetail] = useState<DatasetDetail | null>(null);
  const [workspaceOverview, setWorkspaceOverview] = useState<WorkspaceOverview | null>(null);
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisRunSummary[]>([]);
  const [exportHistory, setExportHistory] = useState<ExportArtifactSummary[]>([]);
  const [analysis, setAnalysis] = useState<AnalysisRunOverview | null>(null);
  const [workspaceSectionRows, setWorkspaceSectionRows] = useState<Partial<Record<TabId, TableRow[]>>>({});
  const [workspaceSectionTotals, setWorkspaceSectionTotals] = useState<Partial<Record<TabId, number>>>({});
  const [workspaceSectionLoading, setWorkspaceSectionLoading] = useState<Partial<Record<TabId, boolean>>>({});
  const [sectionRows, setSectionRows] = useState<Partial<Record<TabId, TableRow[]>>>({});
  const [sectionTotals, setSectionTotals] = useState<Partial<Record<TabId, number>>>({});
  const [sectionLoading, setSectionLoading] = useState<Partial<Record<TabId, boolean>>>({});
  const [workspaceLoading, setWorkspaceLoading] = useState(false);
  const [workspaceSaving, setWorkspaceSaving] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [activeTab, setActiveTab] = useState<TabId>("dataset");
  const [searchQuery, setSearchQuery] = useState("");
  const [tablePage, setTablePage] = useState(1);
  const [selectedUploadName, setSelectedUploadName] = useState("");
  const [reportDraft, setReportDraft] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [customTermInput, setCustomTermInput] = useState("");
  const [excludedTermInput, setExcludedTermInput] = useState("");
  const [synonymCanonicalInput, setSynonymCanonicalInput] = useState("");
  const [synonymAliasesInput, setSynonymAliasesInput] = useState("");
  const [curatedSelection, setCuratedSelection] = useState<Record<string, boolean>>({});
  const [topicNameEdits, setTopicNameEdits] = useState<Record<string, string>>({});
  const [selectedTopicIds, setSelectedTopicIds] = useState<Record<string, boolean>>({});
  const deferredQuery = useDeferredValue(searchQuery);
  const [isPending, startTransition] = useTransition();

  const selectedDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === selectedDatasetId) ?? null,
    [datasets, selectedDatasetId],
  );

  const activeRun = analysis && analysis.dataset_id === selectedDatasetId ? analysis : null;
  const sortedExports = useMemo(() => sortExports(activeRun?.previews.exports ?? []), [activeRun?.previews.exports]);

  const workspaceFilteredRows = useMemo(() => {
    const next: Partial<Record<TabId, TableRow[]>> = {};
    for (const tabId of tableModuleKeys) {
      next[tabId] = filterRows(workspaceSectionRows[tabId] ?? [], deferredQuery);
    }
    return next;
  }, [deferredQuery, workspaceSectionRows]);

  const filteredAnalysisRows = useMemo(() => {
    const sentiments = filterRows(
      (sectionRows.sentiment ?? []).map((item) => ({ ...item, source: stringifyValue(item.source) })),
      deferredQuery,
    );
    const classifications = filterRows(
      (sectionRows.classification ?? []).map((item) => ({ ...item, source: stringifyValue(item.source) })),
      deferredQuery,
    );
    return { sentiment: sentiments, classification: classifications };
  }, [deferredQuery, sectionRows.classification, sectionRows.sentiment]);

  const visibleTopics = useMemo(() => {
    const topics = activeRun?.previews.topics ?? [];
    if (!deferredQuery.trim()) return topics;
    return topics.filter((topic) =>
      filterRows([{ name: topic.name, summary: topic.summary, keywords: topic.keywords.join(" / ") }], deferredQuery).length > 0,
    );
  }, [activeRun?.previews.topics, deferredQuery]);

  const candidateCuratedTerms = useMemo(() => {
    const set = new Set<string>();
    for (const row of workspaceSectionRows.terms ?? []) {
      const term = String(row.term ?? "").trim();
      if (term) set.add(term);
    }
    for (const term of workspaceOverview?.workspace.custom_terms ?? []) set.add(term);
    for (const term of workspaceOverview?.workspace.curated_terms ?? []) set.add(term);
    return Array.from(set).slice(0, 80);
  }, [workspaceOverview?.workspace.curated_terms, workspaceOverview?.workspace.custom_terms, workspaceSectionRows.terms]);

  const moduleCounts: Partial<Record<TabId, number>> = {
    dataset: selectedDataset?.document_count ?? 0,
    overview: activeRun?.overview.sample_count ?? workspaceOverview?.summary.document_count ?? selectedDataset?.document_count ?? 0,
    workspace: workspaceOverview?.summary.selected_term_count ?? 0,
    tokenized: workspaceOverview?.summary.document_count ?? 0,
    terms: workspaceSectionTotals.terms ?? workspaceOverview?.summary.filtered_unique_terms ?? 0,
    selected: workspaceSectionTotals.selected ?? workspaceOverview?.summary.selected_term_count ?? 0,
    matches: workspaceSectionTotals.matches ?? workspaceOverview?.summary.document_count ?? 0,
    matrix: workspaceSectionTotals.matrix ?? workspaceOverview?.summary.document_count ?? 0,
    frequency: workspaceSectionTotals.frequency ?? workspaceOverview?.summary.document_count ?? 0,
    cooccurrence: workspaceSectionTotals.cooccurrence ?? 0,
    topics: activeRun?.previews.topics.length ?? 0,
    sentiment: sectionTotals.sentiment ?? activeRun?.overview.positive_count ?? 0,
    classification: sectionTotals.classification ?? 0,
    report: reportDraft ? 1 : 0,
    exports: sortedExports.length,
  };

  const sidebarStatus = useMemo(() => {
    if (isUploading) return "正在上传语料...";
    if (workspaceSaving) return "正在保存词项...";
    if (workspaceLoading) return "正在加载工作台...";
    if (isPending) return "处理中...";
    if (!statusMessage) return "";
    if (["请先上传语料。", "请选择已上传数据集。", "已加载数据集。", "已加载分析结果。", "词项工作台已刷新。"].includes(statusMessage)) {
      return "";
    }
    return statusMessage;
  }, [isPending, isUploading, statusMessage, workspaceLoading, workspaceSaving]);

  useEffect(() => {
    void refreshDatasets();
  }, []);

  useEffect(() => {
    setReportDraft(activeRun?.previews.report_markdown ?? "");
  }, [activeRun?.previews.report_markdown]);

  useEffect(() => {
    const selected = workspaceOverview?.workspace.curated_terms ?? [];
    if (!candidateCuratedTerms.length && !selected.length) {
      setCuratedSelection({});
      return;
    }
    setCuratedSelection((current) => {
      const next: Record<string, boolean> = {};
      for (const term of candidateCuratedTerms) {
        next[term] = selected.includes(term) ? true : current[term] ?? false;
      }
      return next;
    });
  }, [candidateCuratedTerms, workspaceOverview?.workspace.curated_terms]);

  useEffect(() => {
    const topics = activeRun?.previews.topics ?? [];
    setTopicNameEdits((current) => {
      const next: Record<string, string> = {};
      for (const topic of topics) next[topic.topic_id] = current[topic.topic_id] ?? topic.suggested_name ?? topic.name;
      return next;
    });
    setSelectedTopicIds((current) => {
      const next: Record<string, boolean> = {};
      for (const topic of topics) next[topic.topic_id] = current[topic.topic_id] ?? true;
      return next;
    });
  }, [activeRun?.previews.topics]);

  useEffect(() => {
    setTablePage(1);
  }, [activeTab, deferredQuery, activeRun?.id, workspaceOverview?.workspace.updated_at]);

  useEffect(() => {
    if (!selectedDatasetId) {
      setDatasetDetail(null);
      setWorkspaceOverview(null);
      setWorkspaceSectionRows({});
      setWorkspaceSectionTotals({});
      setWorkspaceSectionLoading({});
      setAnalysisHistory([]);
      setExportHistory([]);
      setAnalysis(null);
      setSectionRows({});
      setSectionTotals({});
      setSectionLoading({});
      return;
    }
    setWorkspaceSectionRows({});
    setWorkspaceSectionTotals({});
    setWorkspaceSectionLoading({});
    void refreshDatasetDetail(selectedDatasetId);
    void refreshWorkspace(selectedDatasetId, { silent: true });
    void refreshAnalysisHistory(selectedDatasetId, true);
    void refreshExportHistory(selectedDatasetId);
  }, [selectedDatasetId]);

  useEffect(() => {
    if (!selectedDatasetId) return;
    if (activeTab === "overview" && !workspaceSectionRows.selected?.length) {
      void loadWorkspaceSection(selectedDatasetId, "selected", 1, { silent: true });
      return;
    }
    if (activeTab === "workspace" && !workspaceSectionRows.terms?.length) {
      void loadWorkspaceSection(selectedDatasetId, "terms", 1, { silent: true });
      return;
    }
    if (!tableModuleKeys.includes(activeTab)) return;
    void loadWorkspaceSection(selectedDatasetId, activeTab, tablePage, { silent: true });
  }, [activeTab, selectedDatasetId, tablePage, workspaceSectionRows.selected?.length, workspaceSectionRows.terms?.length]);

  useEffect(() => {
    if (!activeRun || !analysisSectionMap[activeTab] || activeRun.status !== "completed") return;
    void loadAnalysisSection(activeRun.id, activeTab, tablePage);
  }, [activeRun?.id, activeRun?.status, activeTab, tablePage]);

  useEffect(() => {
    if (!activeRun || (activeRun.status !== "queued" && activeRun.status !== "running")) return;
    const timer = window.setTimeout(() => {
      void loadAnalysisSummary(activeRun.id, { silent: true, refreshHistory: true });
    }, 1500);
    return () => window.clearTimeout(timer);
  }, [activeRun]);

  async function refreshDatasets() {
    const response = await fetch(`${API_BASE_URL}/api/datasets`, { cache: "no-store" });
    const payload = await parseJson<Dataset[]>(response);
    setDatasets(payload);
    setSelectedDatasetId((current) => (current && payload.some((item) => item.id === current) ? current : ""));
  }

  async function refreshDatasetDetail(datasetId: string) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/datasets/${datasetId}?limit=20`, { cache: "no-store" });
      if (!response.ok) {
        setDatasetDetail(null);
        return;
      }
      setDatasetDetail(await parseJson<DatasetDetail>(response));
    } catch {
      setDatasetDetail(null);
    }
  }

  async function refreshWorkspace(datasetId: string, options?: { silent?: boolean }) {
    setWorkspaceLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/datasets/${datasetId}/workspace/summary`, { cache: "no-store" });
      if (!response.ok) {
        if (!options?.silent) setStatusMessage("加载词项工作台失败。");
        return;
      }
      const payload = await parseJson<WorkspaceOverview>(response);
      setWorkspaceOverview(payload);
      if (!options?.silent) setStatusMessage("词项工作台已刷新。");
    } catch {
      if (!options?.silent) setStatusMessage("加载词项工作台失败。");
    } finally {
      setWorkspaceLoading(false);
    }
  }

  async function loadWorkspaceSection(datasetId: string, tabId: TabId, page: number, options?: { silent?: boolean }) {
    if (!tableModuleKeys.includes(tabId)) return;
    const section = workspaceSectionMap[tabId];
    setWorkspaceSectionLoading((current) => ({ ...current, [tabId]: true }));
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/datasets/${datasetId}/workspace/sections/${section}?page=${page}&page_size=${PAGE_SIZE}`,
        { cache: "no-store" },
      );
      if (!response.ok) {
        if (!options?.silent) setStatusMessage("加载工作台内容失败。");
        return;
      }
      const payload = await parseJson<WorkspaceSectionPage>(response);
      setWorkspaceSectionRows((current) => ({ ...current, [tabId]: payload.items }));
      setWorkspaceSectionTotals((current) => ({ ...current, [tabId]: payload.total }));
    } catch {
      if (!options?.silent) setStatusMessage("加载工作台内容失败。");
    } finally {
      setWorkspaceSectionLoading((current) => ({ ...current, [tabId]: false }));
    }
  }

  async function refreshAnalysisHistory(datasetId: string, autoLoadLatest: boolean) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/analyses?dataset_id=${encodeURIComponent(datasetId)}`, { cache: "no-store" });
      if (!response.ok) {
        setAnalysisHistory([]);
        return;
      }
      const payload = await parseJson<AnalysisRunSummary[]>(response);
      setAnalysisHistory(payload);
      if (autoLoadLatest && payload[0]) await loadAnalysisSummary(payload[0].id, { silent: true });
    } catch {
      setAnalysisHistory([]);
    }
  }

  async function refreshExportHistory(datasetId: string) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/exports?dataset_id=${encodeURIComponent(datasetId)}`, { cache: "no-store" });
      if (!response.ok) {
        setExportHistory([]);
        return;
      }
      setExportHistory(await parseJson<ExportArtifactSummary[]>(response));
    } catch {
      setExportHistory([]);
    }
  }

  async function loadAnalysisSummary(runId: string, options?: { silent?: boolean; refreshHistory?: boolean }) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/analyses/${runId}/summary`, { cache: "no-store" });
      if (!response.ok) {
        if (!options?.silent) setStatusMessage("加载分析结果失败。");
        return;
      }
      const payload = await parseJson<AnalysisRunOverview>(response);
      setAnalysis(payload);
      if (payload.status === "completed") {
        setSectionRows({});
        setSectionTotals({});
        setSectionLoading({});
      }
      if (options?.refreshHistory && payload.dataset_id) {
        void refreshAnalysisHistory(payload.dataset_id, false);
        void refreshExportHistory(payload.dataset_id);
        void refreshDatasets();
      }
      if (!options?.silent) setStatusMessage(payload.status === "completed" ? "已加载分析结果。" : formatRunStatus(payload.status));
    } catch {
      if (!options?.silent) setStatusMessage("加载分析结果失败。");
    }
  }

  async function loadAnalysisSection(runId: string, tabId: TabId, page: number) {
    const section = analysisSectionMap[tabId];
    if (!section) return;
    setSectionLoading((current) => ({ ...current, [tabId]: true }));
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/analyses/${runId}/sections/${section}?page=${page}&page_size=${PAGE_SIZE}`,
        { cache: "no-store" },
      );
      if (!response.ok) return;
      const payload = await parseJson<AnalysisSectionPage>(response);
      setSectionRows((current) => ({ ...current, [tabId]: payload.items }));
      setSectionTotals((current) => ({ ...current, [tabId]: payload.total }));
    } finally {
      setSectionLoading((current) => ({ ...current, [tabId]: false }));
    }
  }

  async function saveWorkspaceRequest(url: string, init: RequestInit, successMessage: string) {
    if (!selectedDatasetId) return;
    setWorkspaceSaving(true);
    try {
      const response = await fetch(url, init);
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        setStatusMessage(payload?.detail ?? "词项工作台保存失败。");
        return;
      }
      const payload = await parseJson<DatasetWorkspaceSnapshot>(response);
      setWorkspaceOverview({ workspace: payload.workspace, summary: payload.summary });
      setWorkspaceSectionRows({});
      setWorkspaceSectionTotals({});
      setStatusMessage(successMessage);
    } catch {
      setStatusMessage("词项工作台保存失败。");
    } finally {
      setWorkspaceSaving(false);
    }
  }

  async function handleUpload(formData: FormData) {
    setIsUploading(true);
    setStatusMessage("正在上传语料...");
    try {
      const response = await fetch(`${API_BASE_URL}/api/datasets/upload`, { method: "POST", body: formData });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        setStatusMessage(payload?.detail ?? "上传失败。");
        return;
      }
      const payload = await response.json();
      const nextDataset = payload.dataset as Dataset;
      setDatasets((current) => [nextDataset, ...current.filter((item) => item.id !== nextDataset.id)]);
      setDatasetDetail({
        dataset: nextDataset,
        documents: (payload.documents as DatasetDocument[]) ?? [],
        workspace: null,
      });
      setSelectedDatasetId(nextDataset.id);
      setWorkspaceOverview(null);
      setWorkspaceSectionRows({});
      setWorkspaceSectionTotals({});
      setWorkspaceSectionLoading({});
      setAnalysis(null);
      setSectionRows({});
      setSectionTotals({});
      setSectionLoading({});
      setActiveTab("dataset");
      setSearchQuery("");
      setSelectedUploadName("");
      setCustomTermInput("");
      setExcludedTermInput("");
      setSynonymCanonicalInput("");
      setSynonymAliasesInput("");
      uploadFormRef.current?.reset();
      if (fileInputRef.current) fileInputRef.current.value = "";
      setStatusMessage("上传完成。");
      void refreshWorkspace(nextDataset.id, { silent: true });
      void refreshAnalysisHistory(nextDataset.id, true);
      void refreshExportHistory(nextDataset.id);
    } catch {
      setStatusMessage("上传失败，请检查服务是否已启动。");
    } finally {
      setIsUploading(false);
    }
  }

  async function runAnalysisStage(stage: AnalysisStage, targetTab: TabId, statusText: string) {
    if (!selectedDatasetId) {
      setStatusMessage("请先选择数据集。");
      return;
    }
    startTransition(async () => {
      setStatusMessage(statusText);
      const response = await fetch(`${API_BASE_URL}/api/analyses/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: selectedDatasetId,
          analysis_stage: stage,
          topic_count: 8,
          use_llm: false,
          smart_topic_names: false,
          write_exports: true,
          export_xlsx: false,
        }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        setStatusMessage(payload?.detail ?? "分析失败。");
        return;
      }
      const payload = await response.json();
      const nextRun = payload.run as AnalysisRun;
      setAnalysis({
        ...nextRun,
        generator_stack: [],
        settings: {},
        overview: {
          sample_count: 0,
          topics_count: 0,
          positive_count: 0,
          neutral_count: 0,
          negative_count: 0,
          dominant_classification: null,
          export_count: 0,
        },
        previews: { top_terms: [], topics: [], report_markdown: "", exports: [] },
      });
      setSectionRows({});
      setSectionTotals({});
      setSectionLoading({});
      setActiveTab(targetTab);
      setSearchQuery("");
      setStatusMessage(formatRunStatus(nextRun.status));
      await refreshAnalysisHistory(selectedDatasetId, false);
    });
  }

  async function retryAnalysis(runId: string) {
    startTransition(async () => {
      setStatusMessage("正在重新提交...");
      const response = await fetch(`${API_BASE_URL}/api/analyses/${runId}/retry`, { method: "POST" });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        setStatusMessage(payload?.detail ?? "重试失败。");
        return;
      }
      const payload = await response.json();
      const nextRun = payload.run as AnalysisRun;
      setAnalysis({
        ...nextRun,
        generator_stack: [],
        settings: {},
        overview: {
          sample_count: 0,
          topics_count: 0,
          positive_count: 0,
          neutral_count: 0,
          negative_count: 0,
          dominant_classification: null,
          export_count: 0,
        },
        previews: { top_terms: [], topics: [], report_markdown: "", exports: [] },
      });
      setActiveTab("overview");
      setStatusMessage(formatRunStatus(nextRun.status));
      await refreshAnalysisHistory(nextRun.dataset_id, false);
    });
  }

  async function addCustomTerms() {
    if (!selectedDatasetId || !customTermInput.trim()) return;
    const terms = customTermInput
      .split(/[\n,，、]+/)
      .map((item) => item.trim())
      .filter(Boolean);
    if (!terms.length) return;
    setCustomTermInput("");
    await saveWorkspaceRequest(
      `${API_BASE_URL}/api/datasets/${selectedDatasetId}/workspace/custom-terms`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(terms) },
      "补词已更新。",
    );
  }

  async function addExcludedTerms() {
    if (!selectedDatasetId || !excludedTermInput.trim()) return;
    const terms = excludedTermInput
      .split(/[\n,，、]+/)
      .map((item) => item.trim())
      .filter(Boolean);
    if (!terms.length) return;
    setExcludedTermInput("");
    await saveWorkspaceRequest(
      `${API_BASE_URL}/api/datasets/${selectedDatasetId}/workspace/excluded-terms`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(terms) },
      "排除词已更新。",
    );
  }

  async function saveSynonymGroup() {
    if (!selectedDatasetId || !synonymCanonicalInput.trim()) return;
    const aliases = synonymAliasesInput
      .split(/[\n,，、]+/)
      .map((item) => item.trim())
      .filter(Boolean);
    await saveWorkspaceRequest(
      `${API_BASE_URL}/api/datasets/${selectedDatasetId}/workspace/synonym-groups`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ canonical_term: synonymCanonicalInput.trim(), aliases }),
      },
      "同义词组已更新。",
    );
    setSynonymCanonicalInput("");
    setSynonymAliasesInput("");
  }

  async function saveCuratedTerms() {
    if (!selectedDatasetId) return;
    const curatedTerms = candidateCuratedTerms.filter((term) => curatedSelection[term]);
    await saveWorkspaceRequest(
      `${API_BASE_URL}/api/datasets/${selectedDatasetId}/workspace/curated-terms`,
      { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(curatedTerms) },
      curatedTerms.length ? "人工选词已保存。" : "已切回自动选词。",
    );
  }

  async function copyReport() {
    if (!reportDraft.trim()) {
      setStatusMessage("暂无报告内容。");
      return;
    }
    try {
      await navigator.clipboard.writeText(reportDraft);
      setStatusMessage("报告已复制。");
    } catch {
      setStatusMessage("复制失败。");
    }
  }

  async function runClassificationFromTopics() {
    if (!selectedDatasetId || !activeRun || activeRun.status !== "completed") {
      setStatusMessage("请先完成主题分析。");
      return;
    }
    const topics = activeRun.previews.topics.filter((topic) => selectedTopicIds[topic.topic_id]);
    const labels = topics
      .map((topic) => (topicNameEdits[topic.topic_id] ?? topic.suggested_name ?? topic.name).trim())
      .filter(Boolean);
    if (!labels.length) {
      setStatusMessage("请选择分类名称。");
      return;
    }
    const profiles = topics.map((topic) => ({
      name: (topicNameEdits[topic.topic_id] ?? topic.suggested_name ?? topic.name).trim(),
      description: topic.summary || `${topic.name}相关文本`,
      keywords: topic.keywords.slice(0, 12),
      positive_examples: topic.evidences.map((item) => item.snippet).filter(Boolean).slice(0, 3),
      negative_examples: [],
      source_topic_ids: [topic.topic_id],
    }));
    startTransition(async () => {
      setStatusMessage("正在运行分类...");
      const response = await fetch(`${API_BASE_URL}/api/analyses/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: selectedDatasetId,
          analysis_stage: "classify",
          label_schema: {
            id: "confirmed_topics",
            name: "确认主题分类",
            description: "用户确认后的主题分类",
            labels: Array.from(new Set(labels)),
            profiles,
          },
          use_llm: false,
          write_exports: true,
          export_xlsx: false,
        }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        setStatusMessage(payload?.detail ?? "分类失败。");
        return;
      }
      const payload = await response.json();
      const nextRun = payload.run as AnalysisRun;
      setAnalysis({
        ...nextRun,
        generator_stack: [],
        settings: {},
        overview: {
          sample_count: 0,
          topics_count: 0,
          positive_count: 0,
          neutral_count: 0,
          negative_count: 0,
          dominant_classification: null,
          export_count: 0,
        },
        previews: { top_terms: [], topics: [], report_markdown: "", exports: [] },
      });
      setActiveTab("classification");
      setStatusMessage(formatRunStatus(nextRun.status));
      await refreshAnalysisHistory(selectedDatasetId, false);
    });
  }

  const latestFailedRun = analysisHistory.find((item) => item.status === "failed");
  const latestCompletedRun = analysisHistory.find((item) => item.status === "completed");

  function renderDatasetPreview() {
    const emptyHint = selectedDataset
      ? "当前数据集暂无预览内容。"
      : "先在左侧上传语料并选择数据集，这里会显示样本预览。";
    return (
      <section className="panel">
        <div className="panel-header">
          <h2>数据预览</h2>
        </div>
        <dl className="dataset-meta dataset-meta-grid">
          <div>
            <dt>名称</dt>
            <dd>{selectedDataset?.name ?? "-"}</dd>
          </div>
          <div>
            <dt>样本量</dt>
            <dd>{selectedDataset?.document_count ?? 0}</dd>
          </div>
          <div>
            <dt>文件</dt>
            <dd>{selectedDataset?.source_filename ?? "-"}</dd>
          </div>
          <div>
            <dt>文本列</dt>
            <dd>{datasetDetail?.dataset.text_column ?? "-"}</dd>
          </div>
        </dl>
        <div className="preview-table">
          <div className="table-shell">
            <table className="data-table">
              <thead>
                <tr>
                  <th>行号</th>
                  <th>标题</th>
                  <th>正文</th>
                </tr>
              </thead>
              <tbody>
                {(datasetDetail?.documents ?? []).slice(0, 20).map((document) => (
                  <tr key={document.id}>
                    <td>{document.source_row}</td>
                    <td>{document.title ?? "-"}</td>
                    <td>{document.content}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {!datasetDetail?.documents.length ? <p className="empty-state">{emptyHint}</p> : null}
        </div>
      </section>
    );
  }

  function renderOverview() {
    return (
      <div className="workspace-stack">
        <section className="panel">
          <div className="panel-header">
            <h2>工作总览</h2>
          </div>
          <div className="metric-grid">
            <div className="metric-box">
              <span>语料样本</span>
              <strong>{workspaceOverview?.summary.document_count ?? selectedDataset?.document_count ?? 0}</strong>
            </div>
            <div className="metric-box">
              <span>工作词表</span>
              <strong>{workspaceOverview?.summary.selected_term_count ?? 0}</strong>
            </div>
            <div className="metric-box">
              <span>补词数</span>
              <strong>{workspaceOverview?.summary.custom_term_count ?? 0}</strong>
            </div>
            <div className="metric-box">
              <span>同义词组</span>
              <strong>{workspaceOverview?.summary.synonym_group_count ?? 0}</strong>
            </div>
            <div className="metric-box">
              <span>最新高级分析</span>
              <strong>{activeRun ? formatRunStatus(activeRun.status) : "未运行"}</strong>
            </div>
          </div>
        </section>

        <section className="panel split-panel">
          <div>
            <div className="panel-header">
              <h2>当前词项工作台</h2>
            </div>
            <div className="simple-list">
              {(workspaceSectionRows.selected ?? []).slice(0, 10).map((item, index) => (
                <div className="simple-row" key={`${item.term}-${index}`}>
                  <span>{stringifyValue(item.term)}</span>
                  <strong>{stringifyValue(item.term_frequency)}</strong>
                </div>
              ))}
              {!workspaceSectionRows.selected?.length ? <p className="empty-state">暂无工作词项。</p> : null}
            </div>
          </div>
          <div>
            <div className="panel-header">
              <h2>最近分析记录</h2>
            </div>
            <div className="simple-list">
              {analysisHistory.slice(0, 6).map((run) => (
                <button
                  className={`history-item ${activeRun?.id === run.id ? "history-item-active" : ""}`}
                  key={run.id}
                  type="button"
                  onClick={() => void loadAnalysisSummary(run.id)}
                >
                  <div className="history-row">
                    <strong>{formatRunStatus(run.status)}</strong>
                    <span>{formatDateTime(run.finished_at ?? run.created_at)}</span>
                  </div>
                  <div className="history-subrow">
                    {run.id} · 导出 {run.export_count} 项
                  </div>
                  {run.error ? <div className="history-error">{run.error}</div> : null}
                </button>
              ))}
              {!analysisHistory.length ? <p className="empty-state">暂无分析历史。</p> : null}
            </div>
          </div>
        </section>
      </div>
    );
  }

  function renderWorkspaceManager() {
    const snapshot = workspaceOverview;
    if (!snapshot) {
      return (
        <section className="panel">
          <div className="panel-header">
            <h2>词项工作台</h2>
          </div>
          <p className="empty-state">请选择数据集。</p>
        </section>
      );
    }

    return (
      <div className="workspace-stack">
        <section className="panel">
          <div className="panel-header panel-actions">
            <h2>词项工作台</h2>
            <div className="action-row">
              <button className="ghost-button" disabled={workspaceLoading || workspaceSaving} onClick={() => void refreshWorkspace(selectedDatasetId)} type="button">
                {workspaceLoading ? "刷新中" : "刷新工作台"}
              </button>
              <button className="primary-button" disabled={workspaceSaving} onClick={() => void saveCuratedTerms()} type="button">
                保存人工选词
              </button>
            </div>
          </div>
          <div className="metric-grid workspace-metric-grid">
            <div className="metric-box">
              <span>过滤后词表</span>
              <strong>{snapshot.summary.filtered_unique_terms}</strong>
            </div>
            <div className="metric-box">
              <span>当前选词</span>
              <strong>{snapshot.summary.selected_term_count}</strong>
            </div>
            <div className="metric-box">
              <span>补词</span>
              <strong>{snapshot.summary.custom_term_count}</strong>
            </div>
            <div className="metric-box">
              <span>排除词</span>
              <strong>{snapshot.summary.excluded_term_count}</strong>
            </div>
            <div className="metric-box">
              <span>同义词组</span>
              <strong>{snapshot.summary.synonym_group_count}</strong>
            </div>
          </div>
        </section>

        <section className="panel split-panel workspace-ops-grid">
          <div className="panel-soft">
            <div className="panel-subhead">
              <h3>补词</h3>
            </div>
            <div className="inline-input-row">
              <input
                className="search-input"
                placeholder="输入词语，逗号分隔"
                value={customTermInput}
                onChange={(event) => setCustomTermInput(event.target.value)}
              />
              <button className="primary-button" disabled={workspaceSaving} onClick={() => void addCustomTerms()} type="button">
                添加
              </button>
            </div>
            <div className="tag-list">
              {snapshot.workspace.custom_terms.map((term) => (
                <button
                  className="tag-chip"
                  key={term}
                  onClick={() =>
                    void saveWorkspaceRequest(
                      `${API_BASE_URL}/api/datasets/${selectedDatasetId}/workspace/custom-terms/${encodeURIComponent(term)}`,
                      { method: "DELETE" },
                      "补词已移除。",
                    )
                  }
                  type="button"
                >
                  {term}
                  <span>移除</span>
                </button>
              ))}
              {!snapshot.workspace.custom_terms.length ? <p className="empty-state">暂无补词。</p> : null}
            </div>
          </div>

          <div className="panel-soft">
            <div className="panel-subhead">
              <h3>排除词</h3>
            </div>
            <div className="inline-input-row">
              <input
                className="search-input"
                placeholder="输入要过滤的词"
                value={excludedTermInput}
                onChange={(event) => setExcludedTermInput(event.target.value)}
              />
              <button className="primary-button" disabled={workspaceSaving} onClick={() => void addExcludedTerms()} type="button">
                添加
              </button>
            </div>
            <div className="tag-list">
              {snapshot.workspace.excluded_terms.map((term) => (
                <button
                  className="tag-chip tag-chip-muted"
                  key={term}
                  onClick={() =>
                    void saveWorkspaceRequest(
                      `${API_BASE_URL}/api/datasets/${selectedDatasetId}/workspace/excluded-terms/${encodeURIComponent(term)}`,
                      { method: "DELETE" },
                      "排除词已移除。",
                    )
                  }
                  type="button"
                >
                  {term}
                  <span>恢复</span>
                </button>
              ))}
              {!snapshot.workspace.excluded_terms.length ? <p className="empty-state">暂无排除词。</p> : null}
            </div>
          </div>
        </section>

        <section className="panel split-panel workspace-ops-grid">
          <div className="panel-soft">
            <div className="panel-subhead">
              <h3>同义词归并</h3>
            </div>
            <div className="inline-input-stack">
              <input
                className="search-input"
                placeholder="规范词，例如：配送员"
                value={synonymCanonicalInput}
                onChange={(event) => setSynonymCanonicalInput(event.target.value)}
              />
              <input
                className="search-input"
                placeholder="别名，逗号分隔，例如：送餐员, 外卖员"
                value={synonymAliasesInput}
                onChange={(event) => setSynonymAliasesInput(event.target.value)}
              />
              <button className="primary-button" disabled={workspaceSaving} onClick={() => void saveSynonymGroup()} type="button">
                保存同义词组
              </button>
            </div>
            <div className="synonym-list">
              {snapshot.workspace.synonym_groups.map((group) => (
                <div className="synonym-card" key={group.canonical_term}>
                  <div className="history-row">
                    <strong>{group.canonical_term}</strong>
                    <button
                      className="mini-button"
                      onClick={() =>
                        void saveWorkspaceRequest(
                          `${API_BASE_URL}/api/datasets/${selectedDatasetId}/workspace/synonym-groups/${encodeURIComponent(group.canonical_term)}`,
                          { method: "DELETE" },
                          "同义词组已移除。",
                        )
                      }
                      type="button"
                    >
                      删除
                    </button>
                  </div>
                  <div className="topic-keywords">{group.aliases.join(" / ") || "无别名"}</div>
                </div>
              ))}
              {!snapshot.workspace.synonym_groups.length ? <p className="empty-state">暂无同义词组。</p> : null}
            </div>
          </div>

          <div className="panel-soft">
            <div className="panel-subhead">
              <h3>人工选词</h3>
            </div>
            <p className="empty-state helper-text">
              勾选后保存，即可把当前工作台切换成“人工选词模式”；如果全部取消并保存，会回退成自动选词。
            </p>
            <div className="curated-grid">
              {candidateCuratedTerms.map((term) => (
                <label className="curated-item" key={term}>
                  <input
                    checked={curatedSelection[term] ?? false}
                    onChange={(event) => setCuratedSelection((current) => ({ ...current, [term]: event.target.checked }))}
                    type="checkbox"
                  />
                  <span>{term}</span>
                </label>
              ))}
              {!candidateCuratedTerms.length ? <p className="empty-state">暂无候选词。</p> : null}
            </div>
          </div>
        </section>
      </div>
    );
  }

  function renderTopics() {
    if (!visibleTopics.length) {
      return (
        <section className="panel">
          <div className="panel-header">
            <h2>主题分析</h2>
          </div>
          <p className="empty-state">暂无主题结果。先运行“发现主题”。</p>
        </section>
      );
    }

    return (
      <section className="panel">
        <div className="panel-header panel-actions">
          <h2>主题分析</h2>
          <div className="action-row">
            <button className="ghost-button" disabled={!activeRun || activeRun.status !== "completed"} onClick={() => void runClassificationFromTopics()} type="button">
              基于主题做分类
            </button>
          </div>
        </div>
        <div className="topic-list">
          {visibleTopics.map((topic) => (
            <article className="topic-card" key={topic.topic_id}>
              <div className="topic-row-head">
                <label className="topic-check">
                  <input
                    checked={selectedTopicIds[topic.topic_id] ?? true}
                    onChange={(event) => setSelectedTopicIds((current) => ({ ...current, [topic.topic_id]: event.target.checked }))}
                    type="checkbox"
                  />
                  <strong>{topic.name}</strong>
                </label>
                <span>{topic.size}</span>
              </div>
              <input
                className="topic-name-input"
                value={topicNameEdits[topic.topic_id] ?? topic.suggested_name ?? topic.name}
                onChange={(event) => setTopicNameEdits((current) => ({ ...current, [topic.topic_id]: event.target.value }))}
              />
              <div className="topic-keywords">{topic.keywords.slice(0, 10).join(" / ")}</div>
              <p>{topic.evidences[0]?.snippet ?? topic.summary}</p>
            </article>
          ))}
        </div>
      </section>
    );
  }

  function renderReport() {
    const reportArtifact = sortedExports.find((item) => item.artifact === "report" && item.format === "md");
    return (
      <section className="panel">
        <div className="panel-header panel-actions">
          <h2>研究报告</h2>
          <div className="action-row">
            <button className="ghost-button" onClick={() => void copyReport()} type="button">
              复制
            </button>
            {reportArtifact && activeRun ? (
              <a className="primary-button" href={exportUrl(activeRun.id, reportArtifact)} rel="noreferrer" target="_blank">
                下载
              </a>
            ) : null}
          </div>
        </div>
        <textarea className="report-editor" onChange={(event) => setReportDraft(event.target.value)} placeholder="暂无报告内容。" value={reportDraft} />
      </section>
    );
  }

  function renderExports() {
    return (
      <div className="workspace-stack">
        <section className="panel">
          <div className="panel-header">
            <h2>当前分析导出</h2>
          </div>
          {!activeRun || !sortedExports.length ? (
            <p className="empty-state">暂无当前分析导出文件。</p>
          ) : (
            <div className="table-shell">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>内容</th>
                    <th>格式</th>
                    <th>行数</th>
                    <th>下载</th>
                  </tr>
                </thead>
                <tbody>
                  {sortExports(activeRun.previews.exports).map((item) => (
                    <tr key={`${item.artifact}-${item.format}`}>
                      <td>{artifactLabel(item.artifact)}</td>
                      <td>{item.format.toUpperCase()}</td>
                      <td>{item.rows}</td>
                      <td>
                        <a className="table-link" href={exportUrl(activeRun.id, item)} rel="noreferrer" target="_blank">
                          下载
                        </a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>历史导出</h2>
          </div>
          {!exportHistory.length ? (
            <p className="empty-state">暂无历史导出。</p>
          ) : (
            <div className="table-shell">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>时间</th>
                    <th>内容</th>
                    <th>格式</th>
                    <th>行数</th>
                    <th>任务</th>
                  </tr>
                </thead>
                <tbody>
                  {exportHistory.slice(0, 40).map((item) => (
                    <tr key={`${item.run_id}-${item.artifact}-${item.format}`}>
                      <td>{formatDateTime(item.created_at)}</td>
                      <td>{artifactLabel(item.artifact)}</td>
                      <td>{item.format.toUpperCase()}</td>
                      <td>{item.rows}</td>
                      <td>{item.run_id}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    );
  }

  function renderModuleTable(tabId: TabId, emptyLabel: string) {
    const rows = workspaceFilteredRows[tabId] ?? [];
    const totalRows = workspaceSectionTotals[tabId] ?? 0;
    return (
      <section className="panel">
        <div className="panel-header">
          <h2>{moduleLabels[tabId]}</h2>
        </div>
        {renderGenericTable(rows, emptyLabel, tablePage, totalRows, workspaceSectionLoading[tabId] ?? workspaceLoading, setTablePage)}
      </section>
    );
  }

  function renderAnalysisTable(tabId: TabId, emptyLabel: string) {
    const rows = tabId === "sentiment" ? filteredAnalysisRows.sentiment : filteredAnalysisRows.classification;
    return (
      <section className="panel">
        <div className="panel-header">
          <h2>{moduleLabels[tabId]}</h2>
        </div>
        {renderGenericTable(
          rows,
          emptyLabel,
          tablePage,
          sectionTotals[tabId] ?? 0,
          sectionLoading[tabId] ?? false,
          setTablePage,
        )}
      </section>
    );
  }

  function renderActiveTab() {
    if (activeTab === "dataset") return renderDatasetPreview();
    if (activeTab === "overview") return renderOverview();
    if (activeTab === "workspace") return renderWorkspaceManager();
    if (activeTab === "tokenized") return renderModuleTable("tokenized", "暂无分词结果。");
    if (activeTab === "terms") return renderModuleTable("terms", "暂无词频结果。");
    if (activeTab === "selected") return renderModuleTable("selected", "暂无选词结果。");
    if (activeTab === "matches") return renderModuleTable("matches", "暂无匹配结果。");
    if (activeTab === "matrix") return renderModuleTable("matrix", "暂无二值矩阵。");
    if (activeTab === "frequency") return renderModuleTable("frequency", "暂无频次矩阵。");
    if (activeTab === "cooccurrence") return renderModuleTable("cooccurrence", "暂无共词结果。");
    if (activeTab === "topics") return renderTopics();
    if (activeTab === "sentiment") return renderAnalysisTable("sentiment", "暂无情感结果。");
    if (activeTab === "classification") return renderAnalysisTable("classification", "暂无分类结果。");
    if (activeTab === "report") return renderReport();
    return renderExports();
  }

  const needsSearch = !["dataset", "overview", "workspace", "report"].includes(activeTab);

  return (
    <main className="app-shell">
      <div className="app-layout">
        <aside className="sidebar">
          <section className="sidebar-panel sidebar-primary">
            <div className="brand-block">
              <h1>文本分析工作台</h1>
              {sidebarStatus ? <span className="status-badge">{sidebarStatus}</span> : null}
            </div>

            <div className="sidebar-stack">
              <label className="field-label" htmlFor="upload-file">
                上传语料
              </label>
              <form
                ref={uploadFormRef}
                className="sidebar-upload"
                onSubmit={(event) => {
                  event.preventDefault();
                  const formData = new FormData(event.currentTarget);
                  const file = formData.get("file");
                  if (!(file instanceof File) || file.size <= 0) {
                    setStatusMessage("请选择文件。");
                    return;
                  }
                  void handleUpload(formData);
                }}
              >
                <input
                  ref={fileInputRef}
                  accept=".csv,.xlsx,.xls,.jsonl"
                  className="file-input"
                  id="upload-file"
                  name="file"
                  onChange={(event) => setSelectedUploadName(event.currentTarget.files?.[0]?.name ?? "")}
                  type="file"
                />
                <span className="file-meta">{selectedUploadName || "CSV / XLSX / JSONL"}</span>
                <div className="sidebar-upload-actions">
                  <button className="primary-button" disabled={isUploading} type="submit">
                    {isUploading ? "上传中" : "上传语料"}
                  </button>
                  <button
                    className="ghost-button"
                    onClick={() => {
                      setSelectedDatasetId("");
                      setDatasetDetail(null);
                      setWorkspaceOverview(null);
                      setWorkspaceSectionRows({});
                      setWorkspaceSectionTotals({});
                      setWorkspaceSectionLoading({});
                      setAnalysis(null);
                      setAnalysisHistory([]);
                      setExportHistory([]);
                      setSectionRows({});
                      setSectionTotals({});
                      setSectionLoading({});
                      setActiveTab("dataset");
                      setSearchQuery("");
                      setSelectedUploadName("");
                      if (uploadFormRef.current) uploadFormRef.current.reset();
                      if (fileInputRef.current) fileInputRef.current.value = "";
                      setStatusMessage("");
                    }}
                    type="button"
                  >
                    清空
                  </button>
                </div>
              </form>
            </div>

            <div className="sidebar-stack">
              <label className="field-label" htmlFor="dataset-select">
                当前数据集
              </label>
              <select
                className="dataset-select"
                id="dataset-select"
                value={selectedDatasetId}
                onChange={(event) => {
                  setSelectedDatasetId(event.target.value);
                  setActiveTab("dataset");
                  setSearchQuery("");
                  setStatusMessage(event.target.value ? "已加载数据集。" : "请选择已上传数据集。");
                }}
              >
                <option value="">请选择已上传数据集</option>
                {datasets.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.name} · {dataset.document_count} 条
                  </option>
                ))}
              </select>
            </div>

            <div className="module-nav" aria-label="分析模块">
              {moduleGroups.map((group) => (
                <div key={group.label}>
                  <span className="module-group-label">{group.label}</span>
                  {group.ids.map((moduleId) => (
                    <button
                      className={`module-button ${activeTab === moduleId ? "module-button-active" : ""}`}
                      key={moduleId}
                      onClick={() => setActiveTab(moduleId)}
                      type="button"
                    >
                      <span>{moduleLabels[moduleId]}</span>
                      {moduleCounts[moduleId] !== undefined ? <em>{moduleCounts[moduleId]}</em> : null}
                    </button>
                  ))}
                </div>
              ))}
            </div>
          </section>
        </aside>

        <section className="workspace">
          <section className="workspace-toolbar">
            <div className="workspace-head">
              <div>
                <span className="eyebrow">{moduleLabels[activeTab]}</span>
                <h2>{selectedDataset?.name ?? "未选择数据集"}</h2>
              </div>
              <div className="workspace-head-side">
                <div className="workspace-meta">
                  <span>{selectedDataset?.document_count ?? 0} 条</span>
                  {datasetDetail?.dataset.text_column ? <span>文本列：{datasetDetail.dataset.text_column}</span> : null}
                  {workspaceOverview?.workspace.updated_at ? <span>工作台：{formatDateTime(workspaceOverview.workspace.updated_at)}</span> : null}
                  {activeRun ? <span>分析：{formatRunStatus(activeRun.status)}</span> : null}
                </div>
              </div>
            </div>

            <div className="analysis-actions">
              <button
                className="primary-button action-card"
                disabled={!selectedDatasetId || isPending || workspaceSaving}
                onClick={() => setActiveTab("workspace")}
                type="button"
              >
                <strong>整理词项</strong>
                <span>补词、排除、同义词、人工选词</span>
              </button>
              <button
                className="ghost-button action-card"
                disabled={!selectedDatasetId || isPending || activeRun?.status === "queued" || activeRun?.status === "running"}
                onClick={() => void runAnalysisStage("explore", "selected", "正在生成探索结果...")}
                type="button"
              >
                <strong>生成探索结果</strong>
                <span>选词、匹配、矩阵、共词导出</span>
              </button>
              <button
                className="ghost-button action-card"
                disabled={!selectedDatasetId || isPending || activeRun?.status === "queued" || activeRun?.status === "running"}
                onClick={() => void runAnalysisStage("discover", "topics", "正在发现主题结构...")}
                type="button"
              >
                <strong>发现主题</strong>
                <span>主题聚类、关键词、代表文本</span>
              </button>
              <button
                className="ghost-button action-card"
                disabled={!selectedDatasetId || isPending || activeRun?.status === "queued" || activeRun?.status === "running"}
                onClick={() => void runAnalysisStage("sentiment", "sentiment", "正在生成情感结果...")}
                type="button"
              >
                <strong>情感分析</strong>
                <span>正向、中性、负向</span>
              </button>
              {latestFailedRun ? (
                <button className="ghost-button action-card" onClick={() => void retryAnalysis(latestFailedRun.id)} type="button">
                  <strong>重试失败任务</strong>
                  <span>{latestFailedRun.id}</span>
                </button>
              ) : null}
              {latestCompletedRun ? (
                <button className="ghost-button action-card" onClick={() => void loadAnalysisSummary(latestCompletedRun.id)} type="button">
                  <strong>打开最新结果</strong>
                  <span>{latestCompletedRun.id}</span>
                </button>
              ) : null}
            </div>

            {needsSearch ? (
              <input
                className="search-input"
                onChange={(event) => setSearchQuery(event.target.value)}
                placeholder={`搜索${moduleLabels[activeTab]}`}
                value={searchQuery}
              />
            ) : null}
          </section>

          {renderActiveTab()}
        </section>
      </div>
    </main>
  );
}
