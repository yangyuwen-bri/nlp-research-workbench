from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import re
from typing import Dict, Iterable, List, Sequence, Tuple

import jieba
import jieba.posseg as pseg

from ..models import (
    DatasetWorkspace,
    DatasetWorkspaceOverview,
    DatasetWorkspacePatch,
    DatasetWorkspaceSnapshot,
    Document,
    SynonymGroup,
    WorkspaceSectionPage,
    WorkspaceSummary,
)
from ..utils.text import build_binary_matrix, build_cooccurrence_edges, build_match_rows, build_term_stats, select_terms


DEFAULT_STOPWORDS = {
    "我们",
    "你们",
    "他们",
    "这个",
    "那个",
    "一些",
    "已经",
    "因为",
    "所以",
    "还是",
    "就是",
    "可以",
    "没有",
    "非常",
    "觉得",
    "真的",
    "一个",
    "进行",
    "以及",
    "自己",
}


def default_workspace(dataset_id: str, *, auto_top_k_terms: int = 25) -> DatasetWorkspace:
    return DatasetWorkspace(
        dataset_id=dataset_id,
        updated_at=datetime.now(timezone.utc),
        auto_top_k_terms=auto_top_k_terms,
    )


def _normalize_terms(values: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized


def normalize_workspace(workspace: DatasetWorkspace) -> DatasetWorkspace:
    custom_terms = _normalize_terms(workspace.custom_terms)
    excluded_terms = _normalize_terms(workspace.excluded_terms)
    curated_terms = _normalize_terms(workspace.curated_terms)
    synonym_groups: List[SynonymGroup] = []
    seen_canonicals: set[str] = set()
    for group in workspace.synonym_groups:
        canonical = group.canonical_term.strip()
        if not canonical or canonical in seen_canonicals:
            continue
        aliases = [alias for alias in _normalize_terms(group.aliases) if alias != canonical]
        seen_canonicals.add(canonical)
        synonym_groups.append(SynonymGroup(canonical_term=canonical, aliases=aliases))
    return workspace.model_copy(
        update={
            "custom_terms": custom_terms,
            "excluded_terms": excluded_terms,
            "synonym_groups": synonym_groups,
            "curated_terms": curated_terms,
        }
    )


def _touch(workspace: DatasetWorkspace) -> DatasetWorkspace:
    return workspace.model_copy(update={"updated_at": datetime.now(timezone.utc)})


def patch_workspace(workspace: DatasetWorkspace, patch: DatasetWorkspacePatch) -> DatasetWorkspace:
    update: Dict[str, object] = {}
    for field in ("auto_top_k_terms", "custom_terms", "excluded_terms", "synonym_groups", "curated_terms", "notes"):
        value = getattr(patch, field)
        if value is not None:
            update[field] = value
    return _touch(normalize_workspace(workspace.model_copy(update=update)))


def add_custom_terms(workspace: DatasetWorkspace, terms: Sequence[str]) -> DatasetWorkspace:
    return _touch(normalize_workspace(workspace.model_copy(update={"custom_terms": [*workspace.custom_terms, *terms]})))


def remove_custom_term(workspace: DatasetWorkspace, term: str) -> DatasetWorkspace:
    remaining = [item for item in workspace.custom_terms if item != term]
    return _touch(normalize_workspace(workspace.model_copy(update={"custom_terms": remaining})))


def add_excluded_terms(workspace: DatasetWorkspace, terms: Sequence[str]) -> DatasetWorkspace:
    return _touch(normalize_workspace(workspace.model_copy(update={"excluded_terms": [*workspace.excluded_terms, *terms]})))


def remove_excluded_term(workspace: DatasetWorkspace, term: str) -> DatasetWorkspace:
    remaining = [item for item in workspace.excluded_terms if item != term]
    return _touch(normalize_workspace(workspace.model_copy(update={"excluded_terms": remaining})))


def upsert_synonym_group(workspace: DatasetWorkspace, group: SynonymGroup) -> DatasetWorkspace:
    groups = [item for item in workspace.synonym_groups if item.canonical_term != group.canonical_term]
    groups.append(group)
    return _touch(normalize_workspace(workspace.model_copy(update={"synonym_groups": groups})))


def remove_synonym_group(workspace: DatasetWorkspace, canonical_term: str) -> DatasetWorkspace:
    groups = [item for item in workspace.synonym_groups if item.canonical_term != canonical_term]
    return _touch(normalize_workspace(workspace.model_copy(update={"synonym_groups": groups})))


def set_curated_terms(workspace: DatasetWorkspace, terms: Sequence[str]) -> DatasetWorkspace:
    return _touch(normalize_workspace(workspace.model_copy(update={"curated_terms": list(terms)})))


def _keep_token(word: str) -> bool:
    word = word.strip()
    if len(word) < 2:
        return False
    if word in DEFAULT_STOPWORDS:
        return False
    if re.fullmatch(r"[\W_]+", word):
        return False
    return True


def _alias_map(workspace: DatasetWorkspace) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for group in workspace.synonym_groups:
        mapping[group.canonical_term] = group.canonical_term
        for alias in group.aliases:
            mapping[alias] = group.canonical_term
    return mapping


def _build_pos_tokenizer(workspace: DatasetWorkspace) -> pseg.POSTokenizer:
    tokenizer = jieba.Tokenizer()
    for term in workspace.custom_terms:
        tokenizer.add_word(term)
    return pseg.POSTokenizer(tokenizer)


def tokenize_with_workspace(
    text: str,
    workspace: DatasetWorkspace,
    *,
    pos_tokenizer: pseg.POSTokenizer | None = None,
    alias_map: Dict[str, str] | None = None,
    excluded_terms: set[str] | None = None,
) -> List[Tuple[str, str]]:
    tokenizer = pos_tokenizer or _build_pos_tokenizer(workspace)
    alias_mapping = alias_map or _alias_map(workspace)
    excluded = excluded_terms or set(workspace.excluded_terms)
    tokens: List[Tuple[str, str]] = []
    for word, flag in tokenizer.cut(text):
        word = word.strip()
        if not _keep_token(word):
            continue
        word = alias_mapping.get(word, word)
        if word in excluded or not _keep_token(word):
            continue
        tokens.append((word, flag))
    return tokens


def build_frequency_matrix(
    texts: Sequence[str],
    tokenized_documents: Sequence[List[Tuple[str, str]]],
    selected_terms: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for index, (text, tokens) in enumerate(zip(texts, tokenized_documents), start=1):
        counter = Counter(word for word, _ in tokens)
        row: Dict[str, object] = {"row_id": index, "content": text}
        for term in selected_terms:
            row[term] = int(counter.get(term, 0))
        rows.append(row)
    return rows


def _workspace_core(
    workspace: DatasetWorkspace,
    documents: Sequence[Document],
    *,
    top_k_terms: int | None = None,
):
    normalized_workspace = normalize_workspace(workspace)
    texts = [document.content for document in documents]
    pos_tokenizer = _build_pos_tokenizer(normalized_workspace)
    alias_mapping = _alias_map(normalized_workspace)
    excluded = set(normalized_workspace.excluded_terms)
    tokenized = [
        tokenize_with_workspace(
            text,
            normalized_workspace,
            pos_tokenizer=pos_tokenizer,
            alias_map=alias_mapping,
            excluded_terms=excluded,
        )
        for text in texts
    ]
    term_stats, counter = build_term_stats(tokenized)
    top_k = top_k_terms or normalized_workspace.auto_top_k_terms
    selected_rows = _selected_term_rows(normalized_workspace, term_stats, top_k)
    selected_terms = [str(item["term"]) for item in selected_rows]
    summary = WorkspaceSummary(
        document_count=len(documents),
        custom_term_count=len(normalized_workspace.custom_terms),
        excluded_term_count=len(normalized_workspace.excluded_terms),
        synonym_group_count=len(normalized_workspace.synonym_groups),
        curated_term_count=len(normalized_workspace.curated_terms),
        filtered_unique_terms=len(counter),
        selected_term_count=len(selected_terms),
    )
    return normalized_workspace, texts, tokenized, term_stats, selected_rows, selected_terms, summary


def _selected_term_rows(
    workspace: DatasetWorkspace,
    term_stats: Sequence[Dict[str, object]],
    top_k: int,
) -> List[Dict[str, object]]:
    stats_by_term = {str(item["term"]): item for item in term_stats}
    if workspace.curated_terms:
        rows: List[Dict[str, object]] = []
        for index, term in enumerate(workspace.curated_terms, start=1):
            base = stats_by_term.get(
                term,
                {"term": term, "term_frequency": 0, "document_frequency": 0, "pos": "", "cooccurrence_rank": index},
            )
            row = dict(base)
            row["selection_source"] = "curated"
            row["selection_rank"] = index
            rows.append(row)
        return rows

    rows = []
    for index, item in enumerate(select_terms(term_stats, top_k), start=1):
        row = dict(item)
        row["selection_source"] = "auto"
        row["selection_rank"] = index
        rows.append(row)
    return rows


def build_workspace_snapshot(
    workspace: DatasetWorkspace,
    documents: Sequence[Document],
    *,
    top_k_terms: int | None = None,
) -> DatasetWorkspaceSnapshot:
    normalized_workspace, texts, tokenized, term_stats, selected_rows, selected_terms, summary = _workspace_core(
        workspace,
        documents,
        top_k_terms=top_k_terms,
    )
    tokenized_rows = [
        {
            "row_id": index + 1,
            "document_id": document.id,
            "content": document.content,
            "tokens": " ".join(word for word, _ in tokens),
        }
        for index, (document, tokens) in enumerate(zip(documents, tokenized))
    ]
    match_rows = build_match_rows(texts, tokenized, selected_terms)
    binary_matrix = build_binary_matrix(texts, tokenized, selected_terms)
    frequency_matrix = build_frequency_matrix(texts, tokenized, selected_terms)
    cooccurrence_edges = build_cooccurrence_edges(binary_matrix, selected_terms)
    return DatasetWorkspaceSnapshot(
        workspace=normalized_workspace,
        summary=summary,
        top_terms=term_stats,
        tokenized_documents=tokenized_rows,
        selected_terms=selected_rows,
        match_rows=match_rows,
        binary_matrix=binary_matrix,
        frequency_matrix=frequency_matrix,
        cooccurrence_edges=cooccurrence_edges,
    )


def build_workspace_overview(
    workspace: DatasetWorkspace,
    documents: Sequence[Document],
    *,
    top_k_terms: int | None = None,
) -> DatasetWorkspaceOverview:
    normalized_workspace, _, _, _, _, _, summary = _workspace_core(
        workspace,
        documents,
        top_k_terms=top_k_terms,
    )
    return DatasetWorkspaceOverview(workspace=normalized_workspace, summary=summary)


def build_workspace_section_page(
    workspace: DatasetWorkspace,
    documents: Sequence[Document],
    *,
    section: str,
    page: int,
    page_size: int,
    top_k_terms: int | None = None,
) -> WorkspaceSectionPage:
    _, texts, tokenized, term_stats, selected_rows, selected_terms, _ = _workspace_core(
        workspace,
        documents,
        top_k_terms=top_k_terms,
    )
    section_rows: List[Dict[str, object]]
    if section == "tokenized_documents":
        section_rows = [
            {
                "row_id": index + 1,
                "document_id": document.id,
                "content": document.content,
                "tokens": " ".join(word for word, _ in tokens),
            }
            for index, (document, tokens) in enumerate(zip(documents, tokenized))
        ]
    elif section == "top_terms":
        section_rows = list(term_stats)
    elif section == "selected_terms":
        section_rows = list(selected_rows)
    elif section == "match_rows":
        section_rows = build_match_rows(texts, tokenized, selected_terms)
    elif section == "binary_matrix":
        section_rows = build_binary_matrix(texts, tokenized, selected_terms)
    elif section == "frequency_matrix":
        section_rows = build_frequency_matrix(texts, tokenized, selected_terms)
    elif section == "cooccurrence_edges":
        binary_matrix = build_binary_matrix(texts, tokenized, selected_terms)
        section_rows = build_cooccurrence_edges(binary_matrix, selected_terms)
    else:
        raise ValueError(section)

    safe_page = max(1, page)
    safe_page_size = max(1, page_size)
    start = (safe_page - 1) * safe_page_size
    end = start + safe_page_size
    return WorkspaceSectionPage(
        section=section,
        page=safe_page,
        page_size=safe_page_size,
        total=len(section_rows),
        items=section_rows[start:end],
    )
