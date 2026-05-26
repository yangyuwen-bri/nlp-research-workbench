from __future__ import annotations

from importlib import import_module
import re
from typing import Any, List, Sequence

import httpx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..models import Document, Evidence, TopicCluster
from ..utils.text import summarize_keywords, tokenize
from .embeddings import embed_texts
from .llm import BailianClientError

TOPIC_KEYWORD_COUNT = 8
TOPIC_MAX_FEATURES = 1500
TOPIC_BAD_TOKEN = re.compile(r"^(?:https?://|www\.|http|cn|com|\d+[\dA-Za-z_-]*)", re.IGNORECASE)
TOPIC_VALID_CHAR = re.compile(r"[\u4e00-\u9fffA-Za-z]")
TOPIC_CHINESE_CHAR = re.compile(r"[\u4e00-\u9fff]")
TOPIC_GENERIC_TERMS = {
    "一个",
    "一些",
    "一般",
    "不错",
    "东西",
    "个人",
    "为了",
    "但是",
    "位置",
    "体验",
    "入住",
    "其实",
    "其它",
    "其他",
    "再来",
    "准备",
    "出差",
    "前往",
    "可以",
    "可能",
    "因为",
    "地方",
    "基本",
    "太差",
    "如果",
    "客人",
    "宾馆",
    "就是",
    "已经",
    "希望",
    "感觉",
    "携程",
    "整体",
    "时候",
    "有些",
    "有点",
    "服务",
    "比较",
    "没有",
    "环境",
    "现在",
    "真的",
    "知道",
    "觉得",
    "还是",
    "这个",
    "这里",
    "酒店",
    "里面",
    "非常",
    "预定",
    "预订",
    "房间",
}


def _is_topic_token(token: str) -> bool:
    stripped = token.strip()
    if len(stripped) < 2:
        return False
    if stripped in TOPIC_GENERIC_TERMS:
        return False
    if TOPIC_BAD_TOKEN.match(stripped):
        return False
    if not TOPIC_CHINESE_CHAR.search(stripped):
        return False
    if sum(char.isdigit() for char in stripped) >= max(2, len(stripped) // 2):
        return False
    return bool(TOPIC_VALID_CHAR.search(stripped))


def _tokenize_for_vectorizer(text: str) -> List[str]:
    return [word for word, _ in tokenize(text) if _is_topic_token(word)]


def _vectorizer_thresholds(document_count: int | None) -> tuple[float, int]:
    if document_count is not None and document_count < 20:
        return 1.0, 1
    return 0.55, 2


def _make_tfidf_vectorizer(document_count: int | None = None) -> TfidfVectorizer:
    max_df, min_df = _vectorizer_thresholds(document_count)
    return TfidfVectorizer(
        max_features=TOPIC_MAX_FEATURES,
        max_df=max_df,
        min_df=min_df,
        ngram_range=(1, 2),
        tokenizer=_tokenize_for_vectorizer,
        token_pattern=None,
        lowercase=False,
    )


def _make_count_vectorizer(document_count: int | None = None) -> CountVectorizer:
    max_df, min_df = _vectorizer_thresholds(document_count)
    if document_count is None or document_count >= 20:
        max_df = 0.65
    return CountVectorizer(
        max_features=TOPIC_MAX_FEATURES,
        max_df=max_df,
        min_df=min_df,
        ngram_range=(1, 2),
        tokenizer=_tokenize_for_vectorizer,
        token_pattern=None,
        lowercase=False,
    )


def _dense(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def _extract_topic_keywords(texts: Sequence[str], top_n: int = TOPIC_KEYWORD_COUNT) -> List[str]:
    if not texts:
        return []
    vectorizer = _make_tfidf_vectorizer(len(texts))
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return []
    centroid = matrix.mean(axis=0)
    feature_names = vectorizer.get_feature_names_out()
    sorted_indices = centroid.A1.argsort()[::-1][:top_n]
    return [feature_names[index] for index in sorted_indices if feature_names[index]]


def _extract_contrastive_keywords(
    documents: Sequence[Document],
    labels: Sequence[int],
    *,
    top_n: int = TOPIC_KEYWORD_COUNT,
) -> dict[int, List[str]]:
    texts = [document.content for document in documents]
    if not texts:
        return {}
    vectorizer = _make_count_vectorizer(len(texts))
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return {}

    feature_names = vectorizer.get_feature_names_out()
    dense_matrix = _dense(matrix).astype(float)
    unique_labels = sorted(set(int(label) for label in labels))
    cluster_term_totals = []
    cluster_sizes = []
    for label in unique_labels:
        member_indices = [index for index, topic_label in enumerate(labels) if int(topic_label) == label]
        cluster_counts = dense_matrix[member_indices].sum(axis=0)
        cluster_term_totals.append(cluster_counts)
        cluster_sizes.append(max(float(len(member_indices)), 1.0))

    cluster_matrix = np.vstack(cluster_term_totals)
    cluster_presence = (cluster_matrix > 0).sum(axis=0)
    global_presence = (dense_matrix > 0).sum(axis=0)
    cluster_idf = np.log((1 + len(unique_labels)) / (1 + cluster_presence)) + 1.0
    document_idf = np.log((1 + len(texts)) / (1 + global_presence)) + 1.0

    keyword_map: dict[int, List[str]] = {}
    for row_index, label in enumerate(unique_labels):
        normalized_tf = cluster_matrix[row_index] / cluster_sizes[row_index]
        scores = normalized_tf * cluster_idf * np.sqrt(document_idf)
        ordered_indices = np.argsort(scores)[::-1]
        keywords: List[str] = []
        seen_parts: set[str] = set()
        for feature_index in ordered_indices:
            term = feature_names[feature_index]
            if not term or scores[feature_index] <= 0:
                continue
            parts = set(term.split())
            if term in seen_parts:
                continue
            if len(parts) == 1 and term in seen_parts:
                continue
            keywords.append(term)
            seen_parts.update(parts)
            seen_parts.add(term)
            if len(keywords) >= top_n:
                break
        keyword_map[label] = keywords
    return keyword_map


def _cluster_topics_from_labels(
    documents: Sequence[Document],
    labels: Sequence[int],
    vectors: np.ndarray,
    *,
    keyword_map: dict[int, List[str]] | None = None,
) -> List[TopicCluster]:
    topics: List[TopicCluster] = []
    unique_labels = sorted(set(int(label) for label in labels))
    label_positions = {label: index + 1 for index, label in enumerate(unique_labels)}
    resolved_keyword_map = keyword_map or _extract_contrastive_keywords(documents, labels)
    for label in unique_labels:
        member_indices = [index for index, topic_label in enumerate(labels) if int(topic_label) == label]
        cluster_texts = [documents[index].content for index in member_indices]
        keywords = resolved_keyword_map.get(label) or _extract_topic_keywords(cluster_texts)
        cluster_vectors = vectors[member_indices]
        centroid = cluster_vectors.mean(axis=0, keepdims=True)
        similarities = cosine_similarity(cluster_vectors, centroid).reshape(-1)
        ordered_members = [member_indices[index] for index in np.argsort(similarities)[::-1]]
        evidences = [
            Evidence(
                value="主题证据",
                confidence=0.72,
                snippet=documents[index].content[:160],
                document_id=documents[index].id,
                generator="model",
            )
            for index in ordered_members[:3]
        ]
        position = label_positions[label]
        topics.append(
            TopicCluster(
                topic_id=f"topic_{position}",
                name=f"主题 {position}",
                size=len(member_indices),
                keywords=keywords,
                summary=summarize_keywords(keywords),
                evidences=evidences,
            )
        )
    return topics


def _load_bertopic() -> tuple[type[Any] | None, type[Any] | None]:
    try:
        bertopic_module = import_module("bertopic")
        BERTopic = getattr(bertopic_module, "BERTopic", None)
    except Exception:
        return None, None
    try:
        dimensionality_module = import_module("bertopic.dimensionality")
        base_dimensionality = getattr(dimensionality_module, "BaseDimensionalityReduction", None)
    except Exception:
        base_dimensionality = None
    return BERTopic, base_dimensionality


def _build_bertopic_topics(
    documents: Sequence[Document],
    topic_count: int,
    *,
    embeddings: np.ndarray,
    strategy: str,
) -> tuple[List[TopicCluster], str]:
    texts = [document.content for document in documents]
    cluster_count = max(1, min(topic_count, len(texts)))
    if cluster_count == 1:
        labels = np.zeros(len(texts), dtype=int)
        return _cluster_topics_from_labels(documents, labels, embeddings), strategy

    BERTopic, base_dimensionality = _load_bertopic()
    if BERTopic is None:
        raise ImportError("BERTopic is not installed")

    bertopic_kwargs: dict[str, Any] = {
        "embedding_model": None,
        "hdbscan_model": KMeans(n_clusters=cluster_count, random_state=42, n_init=10),
        "vectorizer_model": _make_count_vectorizer(len(texts)),
        "calculate_probabilities": False,
        "top_n_words": TOPIC_KEYWORD_COUNT,
        "verbose": False,
    }
    if base_dimensionality is not None:
        bertopic_kwargs["umap_model"] = base_dimensionality()
    topic_model = BERTopic(**bertopic_kwargs)
    labels, _ = topic_model.fit_transform(texts, embeddings=embeddings)
    normalized_labels = [int(label) for label in labels]
    keyword_map: dict[int, List[str]] = {}
    for label in sorted(set(normalized_labels)):
        if label < 0:
            continue
        try:
            topic_terms = topic_model.get_topic(label) or []
        except Exception:
            topic_terms = []
        keywords = [term for term, _ in topic_terms[:TOPIC_KEYWORD_COUNT] if term]
        if keywords:
            keyword_map[label] = keywords
    topics = _cluster_topics_from_labels(documents, normalized_labels, embeddings, keyword_map=keyword_map)
    return topics, strategy


def _build_kmeans_topics(
    documents: Sequence[Document],
    topic_count: int,
    *,
    vectors: np.ndarray,
    strategy: str,
) -> tuple[List[TopicCluster], str]:
    cluster_count = max(1, min(topic_count, len(documents)))
    if cluster_count == 1:
        labels = np.zeros(len(documents), dtype=int)
    else:
        labels = KMeans(n_clusters=cluster_count, random_state=42, n_init=10).fit_predict(vectors)
    topics = _cluster_topics_from_labels(documents, labels, vectors)
    return topics, strategy


def build_topics(documents: Sequence[Document], topic_count: int, allow_embeddings: bool) -> tuple[List[TopicCluster], str]:
    texts = [document.content for document in documents]
    if not texts:
        return [], "none"

    tfidf_vectors: np.ndarray | None = None
    semantic_vectors: np.ndarray | None = None
    semantic_error: Exception | None = None

    if allow_embeddings:
        try:
            semantic_vectors = _dense(embed_texts(texts))
        except (BailianClientError, httpx.HTTPError, ValueError) as exc:
            semantic_error = exc

    try:
        if semantic_vectors is not None:
            return _build_bertopic_topics(
                documents,
                topic_count,
                embeddings=semantic_vectors,
                strategy="bertopic_semantic_embeddings",
            )
        tfidf_vectors = _dense(_make_tfidf_vectorizer(len(texts)).fit_transform(texts))
        return _build_kmeans_topics(
            documents,
            topic_count,
            vectors=tfidf_vectors,
            strategy="tfidf_kmeans",
        )
    except Exception:
        if tfidf_vectors is None:
            tfidf_vectors = _dense(_make_tfidf_vectorizer(len(texts)).fit_transform(texts))
        if semantic_vectors is not None:
            return _build_kmeans_topics(
                documents,
                topic_count,
                vectors=semantic_vectors,
                strategy="dashscope_embedding_kmeans",
            )
        if semantic_error is not None:
            return _build_kmeans_topics(
                documents,
                topic_count,
                vectors=tfidf_vectors,
                strategy="tfidf_kmeans",
            )
        return _build_kmeans_topics(
            documents,
            topic_count,
            vectors=tfidf_vectors,
            strategy="tfidf_kmeans",
        )
