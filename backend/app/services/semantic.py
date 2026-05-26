from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from ..models import ClassificationResult, Document, Evidence, LabelProfile, SentimentResult
from ..utils.text import split_sentences
from .embeddings import embed_texts


SENTIMENT_PROTOTYPES = {
    "positive": [
        "配送很快，味道很好，下次还会再买。",
        "房间很干净，服务态度很好，整体很满意。",
        "质量不错，做工精细，性价比很高，值得推荐。",
    ],
    "neutral": [
        "已经收到，包装完整，整体还可以，没有特别明显的问题。",
        "就是普通体验，和预期差不多，先用一段时间再看。",
        "送到了，能正常使用，暂时没有更多感受。",
    ],
    "negative": [
        "太难吃了，送得也慢，再也不会点了。",
        "水果收到已经坏了，物流很慢，体验很差。",
        "做工粗糙，功能有问题，客服也没有及时处理。",
    ],
}

CLASSIFICATION_PROTOTYPES = {
    "产品体验": [
        "这条评论主要在讨论产品功能、质量、性能、口味、房间设施或商品本身的体验。",
        "用户关注的是做工、稳定性、使用效果、味道和核心功能。",
    ],
    "价格感知": [
        "这条评论主要在讨论价格、性价比、贵不贵、优惠力度和促销活动。",
        "用户重点提到价格偏高、便宜、折扣、值不值得买。",
    ],
    "服务反馈": [
        "这条评论主要在讨论客服、物流、配送、安装、售后和服务响应速度。",
        "用户重点提到服务态度、履约体验、送货速度和处理问题的过程。",
    ],
}


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _profile_entries(profile: LabelProfile, label: str) -> List[str]:
    parts = [
        f"这段文本主要属于{label}。",
        f"该文本讨论的是{label}相关内容。",
    ]
    if profile.description.strip():
        parts.append(f"{label}的定义：{profile.description.strip()}")
    if profile.keywords:
        parts.append(f"{label}的关键词：{'、'.join(profile.keywords[:12])}")
    for example in profile.positive_examples[:3]:
        clean_example = str(example).strip()
        if clean_example:
            parts.append(f"{label}的典型文本：{clean_example[:120]}")
    return [" ".join(parts)[:900]]


def _profile_predict(
    texts: Sequence[str],
    document_vectors: np.ndarray,
    labels: Sequence[str],
    profiles: Sequence[LabelProfile],
) -> List[Tuple[str, float, str]]:
    profile_map = {profile.name.strip(): profile for profile in profiles if profile.name.strip()}
    positive_entries: List[str] = []
    positive_ranges: Dict[str, tuple[int, int]] = {}
    cursor = 0
    for label in labels:
        profile = profile_map.get(label) or LabelProfile(name=label)
        entries = _profile_entries(profile, label)
        positive_entries.extend(entries)
        positive_ranges[label] = (cursor, cursor + len(entries))
        cursor += len(entries)

    positive_vectors_raw = _normalize(embed_texts(positive_entries))
    positive_vectors = []
    for label in labels:
        start, end = positive_ranges[label]
        positive_vectors.append(positive_vectors_raw[start:end].mean(axis=0))
    positive_vectors = _normalize(np.asarray(positive_vectors))
    profile_keywords = {
        label: [
            str(keyword).strip()
            for keyword in (profile_map.get(label).keywords if profile_map.get(label) else [])
            if str(keyword).strip()
        ]
        for label in labels
    }

    negative_entries_by_label: Dict[str, List[str]] = {}
    for label in labels:
        profile = profile_map.get(label)
        entries = [str(item).strip()[:180] for item in (profile.negative_examples if profile else []) if str(item).strip()]
        contrast_parts = []
        for other_label in labels:
            if other_label == label:
                continue
            other_profile = profile_map.get(other_label)
            if other_profile:
                contrast_parts.append(
                    " ".join(
                        [
                            f"{other_label}：",
                            other_profile.description[:120],
                            "、".join(other_profile.keywords[:8]),
                            " ".join(str(item).strip()[:80] for item in other_profile.positive_examples[:2] if str(item).strip()),
                        ]
                    )
                )
        if contrast_parts:
            entries.append("不属于该类的对照主题：" + " ".join(contrast_parts)[:900])
        negative_entries_by_label[label] = entries

    negative_entries: List[str] = []
    negative_ranges: Dict[str, tuple[int, int]] = {}
    cursor = 0
    for label, entries in negative_entries_by_label.items():
        if not entries:
            continue
        negative_entries.extend(entries)
        negative_ranges[label] = (cursor, cursor + len(entries))
        cursor += len(entries)
    negative_vectors_raw = _normalize(embed_texts(negative_entries)) if negative_entries else np.zeros((0, positive_vectors.shape[1]))

    normalized_documents = _normalize(document_vectors)
    predictions: List[Tuple[str, float, str]] = []
    for text, vector in zip(texts, normalized_documents):
        positive_scores = positive_vectors @ vector
        scores = []
        for index, label in enumerate(labels):
            score = float(positive_scores[index])
            keyword_hits = sum(1 for keyword in profile_keywords.get(label, []) if keyword and keyword in text)
            if keyword_hits:
                score += min(0.35, 0.08 * keyword_hits)
            negative_range = negative_ranges.get(label)
            if negative_range is not None:
                start, end = negative_range
                negative_vectors = negative_vectors_raw[start:end]
                score -= 0.25 * float(np.max(negative_vectors @ vector))
            scores.append(score)
        probabilities = _softmax(np.asarray(scores, dtype=float) * 10.0)
        best_index = int(np.argmax(probabilities))
        snippet = split_sentences(text)[0] if split_sentences(text) else text[:120]
        predictions.append((labels[best_index], float(probabilities[best_index]), snippet))
    return predictions


def _prototype_predict(
    texts: Sequence[str], document_vectors: np.ndarray, prototype_map: Dict[str, Union[str, List[str]]]
) -> List[Tuple[str, float, str]]:
    prototype_labels = list(prototype_map.keys())
    prototype_texts: List[str] = []
    prototype_ranges: Dict[str, tuple[int, int]] = {}
    cursor = 0
    for label in prototype_labels:
        entries = prototype_map[label]
        if isinstance(entries, str):
            entries = [entries]
        prototype_texts.extend(entries)
        prototype_ranges[label] = (cursor, cursor + len(entries))
        cursor += len(entries)
    raw_vectors = _normalize(embed_texts(prototype_texts))
    prototype_vectors = []
    for label in prototype_labels:
        start, end = prototype_ranges[label]
        prototype_vectors.append(raw_vectors[start:end].mean(axis=0))
    prototype_vectors = _normalize(np.asarray(prototype_vectors))
    normalized_documents = _normalize(document_vectors)
    predictions: List[Tuple[str, float, str]] = []
    for text, vector in zip(texts, normalized_documents):
        similarities = prototype_vectors @ vector
        probabilities = _softmax(similarities)
        best_index = int(np.argmax(probabilities))
        snippet = split_sentences(text)[0] if split_sentences(text) else text[:120]
        predictions.append((prototype_labels[best_index], float(probabilities[best_index]), snippet))
    return predictions


def build_semantic_sentiments(documents: Sequence[Document], document_vectors: np.ndarray) -> List[SentimentResult]:
    predictions = _prototype_predict([document.content for document in documents], document_vectors, SENTIMENT_PROTOTYPES)
    return [
        SentimentResult(
            document_id=document.id,
            label=label,  # type: ignore[arg-type]
            score=confidence,
            aspect_hits={},
            evidence=Evidence(
                value=label,
                confidence=confidence,
                snippet=snippet,
                document_id=document.id,
                generator="model",
            ),
        )
        for document, (label, confidence, snippet) in zip(documents, predictions)
    ]


def build_semantic_classifications(
    documents: Sequence[Document], document_vectors: np.ndarray
) -> List[ClassificationResult]:
    predictions = _prototype_predict(
        [document.content for document in documents], document_vectors, CLASSIFICATION_PROTOTYPES
    )
    return [
        ClassificationResult(
            document_id=document.id,
            label=label,
            confidence=confidence,
            evidence=Evidence(
                value=label,
                confidence=confidence,
                snippet=snippet,
                document_id=document.id,
                generator="model",
            ),
        )
        for document, (label, confidence, snippet) in zip(documents, predictions)
    ]


def build_label_semantic_classifications(
    documents: Sequence[Document],
    labels: Sequence[str],
    document_vectors: np.ndarray,
    profiles: Sequence[LabelProfile] | None = None,
) -> List[ClassificationResult]:
    clean_labels = [str(label).strip() for label in labels if str(label).strip()]
    if not clean_labels:
        return []
    clean_profiles = [profile for profile in (profiles or []) if profile.name.strip()]
    if clean_profiles:
        predictions = _profile_predict([document.content for document in documents], document_vectors, clean_labels, clean_profiles)
        return [
            ClassificationResult(
                document_id=document.id,
                label=label,
                confidence=confidence,
                evidence=Evidence(
                    value=label,
                    confidence=confidence,
                    snippet=snippet,
                    document_id=document.id,
                    generator="model",
                ),
            )
            for document, (label, confidence, snippet) in zip(documents, predictions)
        ]
    profile_map = {profile.name.strip(): profile for profile in (profiles or []) if profile.name.strip()}
    prototype_map: Dict[str, List[str]] = {}
    for label in clean_labels:
        profile = profile_map.get(label)
        entries = [
            f"这段文本主要属于{label}。",
            f"该文本讨论的是{label}相关内容。",
        ]
        if profile:
            if profile.description.strip():
                entries.append(f"{label}的定义：{profile.description.strip()}")
            if profile.keywords:
                entries.append(f"{label}的关键词：{'、'.join(profile.keywords[:12])}")
            for example in profile.positive_examples[:3]:
                clean_example = str(example).strip()
                if clean_example:
                    entries.append(f"{label}的典型文本：{clean_example[:160]}")
            for example in profile.negative_examples[:2]:
                clean_example = str(example).strip()
                if clean_example:
                    entries.append(f"不属于{label}的文本：{clean_example[:160]}")
        prototype_map[label] = entries
    predictions = _prototype_predict([document.content for document in documents], document_vectors, prototype_map)
    return [
        ClassificationResult(
            document_id=document.id,
            label=label,
            confidence=confidence,
            evidence=Evidence(
                value=label,
                confidence=confidence,
                snippet=snippet,
                document_id=document.id,
                generator="model",
            ),
        )
        for document, (label, confidence, snippet) in zip(documents, predictions)
    ]
