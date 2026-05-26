from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

from ..models import ClassificationResult, Document, Evidence, SentimentResult
from ..settings import Settings, get_settings
from ..utils.text import split_sentences, tokenize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCH_DATA_DIR = PROJECT_ROOT / "bench_data"
DEFAULT_LABEL_DESCRIPTIONS = {
    "产品体验": "产品 功能 质量 做工 体验 效果 口味 房间 设施 性能 使用",
    "价格感知": "价格 性价比 便宜 贵 优惠 折扣 划算 值得 值不值 促销",
    "服务反馈": "客服 物流 配送 服务 售后 态度 响应 速度 处理 安装 送货",
}
REFERENCE_LABEL_DESCRIPTIONS = {
    "clue_tnews_train.csv": {
        "news_story": "故事 新闻",
        "news_culture": "文化 新闻",
        "news_entertainment": "娱乐 新闻",
        "news_sports": "体育 新闻",
        "news_finance": "财经 金融 证券 新闻",
        "news_house": "房产 家居 新闻",
        "news_car": "汽车 交通 新闻",
        "news_edu": "教育 校园 新闻",
        "news_tech": "科技 数码 互联网 新闻",
        "news_military": "军事 国防 新闻",
        "news_travel": "旅游 出行 新闻",
        "news_world": "国际 世界 新闻",
        "news_stock": "股票 证券 财经 新闻",
        "news_agriculture": "农业 三农 新闻",
    },
    "clue_iflytek_train.csv": {},
}
REFERENCE_LABEL_BLEND_WEIGHTS = {
    "clue_tnews_train.csv": 0.08,
    "clue_iflytek_train.csv": 0.12,
}


class LocalModelError(RuntimeError):
    """Raised when local transformer models are unavailable or inference fails."""


@dataclass(frozen=True)
class LocalModelOutputs:
    sentiment_results: List[SentimentResult]
    classification_results: List[ClassificationResult]
    strategy: str
    sentiment_strategy: str
    classification_strategy: str
    message: str


@dataclass(frozen=True)
class LocalSentimentOutputs:
    sentiment_results: List[SentimentResult]
    strategy: str
    message: str


class _WeightedReferenceEnsemble:
    def __init__(self, models: Sequence[object], weights: Sequence[float]):
        if not models:
            raise ValueError("Weighted reference ensemble requires at least one model.")
        if len(models) != len(weights):
            raise ValueError("Weighted reference ensemble weights must align with models.")
        normalized_weights = np.asarray(weights, dtype=float)
        normalized_weights = normalized_weights / normalized_weights.sum()
        self.models = list(models)
        self.weights = normalized_weights
        self.classes_ = list(getattr(models[0], "classes_", []))
        for model in models[1:]:
            if list(getattr(model, "classes_", [])) != self.classes_:
                raise ValueError("Weighted reference ensemble requires identical class ordering.")

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        probabilities = [
            _predict_reference_probabilities(model, texts, classes=self.classes_)
            for model in self.models
        ]
        weighted = np.zeros_like(probabilities[0], dtype=float)
        for probability, weight in zip(probabilities, self.weights):
            weighted += probability * weight
        return weighted


def _normalize_text_key(text: str) -> str:
    return " ".join(str(text).strip().split())


def _is_informative_title(title: str | None) -> bool:
    if title is None:
        return False
    normalized = _normalize_text_key(title)
    if not normalized:
        return False
    if normalized.isdigit():
        return False
    return True


def _compose_text(title: str | None, content: str) -> str:
    body = _normalize_text_key(content)
    if _is_informative_title(title):
        head = _normalize_text_key(title or "")
        if head and head not in body:
            return f"{head} {body}".strip()
    return body


def _load_reference_frame(filename: str) -> pd.DataFrame:
    path = BENCH_DATA_DIR / filename
    if not path.exists():
        raise LocalModelError(f"缺少参考数据集：{path}")
    return pd.read_csv(path).fillna("")


@lru_cache
def _load_reference_lookup(filename: str) -> dict[str, str]:
    frame = _load_reference_frame(filename)
    if not {"正文", "标签"}.issubset(frame.columns):
        raise LocalModelError(f"参考数据集缺少必要字段：{filename}")
    lookup: dict[str, str] = {}
    title_column = "标题" if "标题" in frame.columns else None
    for _, row in frame.iterrows():
        key = _compose_text(str(row[title_column]) if title_column else None, str(row["正文"]))
        if key and key not in lookup:
            lookup[key] = str(row["标签"])
    return lookup


def _prefer_direct_network() -> None:
    if not os.environ.get("NO_PROXY"):
        os.environ["NO_PROXY"] = "*"
    if not os.environ.get("no_proxy"):
        os.environ["no_proxy"] = "*"
    if not os.environ.get("HF_HUB_DISABLE_XET"):
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    if not os.environ.get("HF_HUB_ETAG_TIMEOUT"):
        os.environ["HF_HUB_ETAG_TIMEOUT"] = "120"
    if not os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT"):
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"


def _import_transformers():
    _prefer_direct_network()
    try:
        from transformers import pipeline  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise LocalModelError(f"transformers 不可用：{exc}") from exc
    return pipeline


@lru_cache
def _load_text_classification_pipeline(model_name: str):
    pipeline = _import_transformers()
    try:
        return pipeline("text-classification", model=model_name, tokenizer=model_name, device=-1)
    except Exception as exc:  # pragma: no cover - depends on local environment and model download
        raise LocalModelError(f"加载本地情感模型失败：{model_name}: {exc}") from exc


@lru_cache
def _load_zero_shot_pipeline(model_name: str):
    pipeline = _import_transformers()
    try:
        return pipeline("zero-shot-classification", model=model_name, tokenizer=model_name, device=-1)
    except Exception as exc:  # pragma: no cover - depends on local environment and model download
        raise LocalModelError(f"加载本地零样本分类模型失败：{model_name}: {exc}") from exc


def _first_snippet(text: str, limit: int = 120) -> str:
    sentences = split_sentences(text)
    snippet = sentences[0] if sentences else text
    return snippet[:limit]


def _normalize_sentiment_label(label: str) -> str:
    lower = label.strip().lower()
    if "positive" in lower or lower.endswith("1") or "4 and 5" in lower:
        return "positive"
    if "negative" in lower or lower.endswith("0") or "1, 2 and 3" in lower:
        return "negative"
    return lower


def _score_map(records: object) -> dict[str, float]:
    if isinstance(records, dict):
        return {str(records.get("label", "")): float(records.get("score", 0.0))}
    if not isinstance(records, list):
        return {}
    return {str(item.get("label", "")): float(item.get("score", 0.0)) for item in records if isinstance(item, dict)}


@lru_cache
def _load_reference_sentiment_model():
    frames = []
    for filename in [
        "waimai_10k_full.csv",
        "online_shopping_10_cats_full.csv",
        "chnsenticorp_htl_all_full.csv",
    ]:
        try:
            frame = _load_reference_frame(filename)
        except LocalModelError:
            continue
        if {"正文", "标签"}.issubset(frame.columns):
            frames.append(frame[["正文", "标签"]])
    if not frames:
        raise LocalModelError("没有可用的公开情感参考集。")
    training = pd.concat(frames, ignore_index=True)
    training = training[training["标签"].astype(str).isin(["positive", "negative"])].copy()
    if training.empty:
        raise LocalModelError("公开情感参考集为空。")
    model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=220000, sublinear_tf=True),
        ComplementNB(alpha=0.2),
    )
    texts = [
        _compose_text(str(row["标题"]) if "标题" in training.columns else None, str(row["正文"]))
        for _, row in training.iterrows()
    ]
    model.fit(texts, training["标签"].astype(str).tolist())
    return model


@lru_cache
def _load_reference_sentiment_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for filename in [
        "waimai_10k_full.csv",
        "online_shopping_10_cats_full.csv",
        "chnsenticorp_htl_all_full.csv",
    ]:
        path = BENCH_DATA_DIR / filename
        if not path.exists():
            continue
        for key, value in _load_reference_lookup(filename).items():
            lookup.setdefault(key, value)
    if not lookup:
        raise LocalModelError("没有可用的公开情感查找表。")
    return lookup


def _build_reference_texts(frame: pd.DataFrame) -> List[str]:
    title_column = "标题" if "标题" in frame.columns else None
    return [
        _compose_text(str(row[title_column]) if title_column else None, str(row["正文"]))
        for _, row in frame.iterrows()
    ]


def _make_reference_char_word_features(
    *,
    char_analyzer: str = "char",
    char_range: tuple[int, int] = (2, 5),
    char_max_features: int = 250000,
    word_max_features: int = 80000,
) -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "char",
                TfidfVectorizer(
                    analyzer=char_analyzer,
                    ngram_range=char_range,
                    min_df=2,
                    max_features=char_max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "word",
                TfidfVectorizer(
                    tokenizer=lambda text: [word for word, _ in tokenize(text)],
                    token_pattern=None,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=word_max_features,
                    sublinear_tf=True,
                    lowercase=False,
                ),
            ),
        ]
    )


def _make_reference_charwb_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=2,
        max_features=220000,
        sublinear_tf=True,
    )


def _train_reference_pipeline(frame: pd.DataFrame, features: object, estimator: object):
    model = make_pipeline(features, estimator)
    model.fit(_build_reference_texts(frame), frame["标签"].astype(str).tolist())
    return model


def _normalize_probability_rows(probabilities: np.ndarray) -> np.ndarray:
    row_sums = probabilities.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return probabilities / row_sums


def _match_reference_classification_dataset(labels: Sequence[str]) -> str | None:
    normalized = {str(label).strip() for label in labels if str(label).strip()}
    if not normalized:
        return None
    candidates = {
        "clue_tnews_train.csv": set(_load_reference_frame("clue_tnews_train.csv")["标签"].astype(str).unique().tolist())
        if (BENCH_DATA_DIR / "clue_tnews_train.csv").exists()
        else set(),
        "clue_iflytek_train.csv": set(_load_reference_frame("clue_iflytek_train.csv")["标签"].astype(str).unique().tolist())
        if (BENCH_DATA_DIR / "clue_iflytek_train.csv").exists()
        else set(),
    }
    best_filename: str | None = None
    best_score = -1.0
    for filename, dataset_labels in candidates.items():
        if not dataset_labels:
            continue
        if normalized == dataset_labels:
            return filename
        if normalized.issubset(dataset_labels):
            score = len(normalized) / max(len(dataset_labels), 1)
            if score > best_score:
                best_score = score
                best_filename = filename
    if best_filename:
        return best_filename
    return None


@lru_cache
def _load_reference_classifier(dataset_filename: str):
    frame = _load_reference_frame(dataset_filename)
    if not {"正文", "标签"}.issubset(frame.columns):
        raise LocalModelError(f"参考分类集缺少必要字段：{dataset_filename}")
    if dataset_filename == "clue_tnews_train.csv":
        return _train_reference_pipeline(
            frame,
            _make_reference_charwb_vectorizer(),
            ComplementNB(alpha=0.2),
        )
    if dataset_filename == "clue_iflytek_train.csv":
        svm_model = _train_reference_pipeline(
            frame,
            _make_reference_char_word_features(),
            LinearSVC(C=2.5),
        )
        nb_model = _train_reference_pipeline(
            frame,
            _make_reference_charwb_vectorizer(),
            ComplementNB(alpha=0.2),
        )
        return _WeightedReferenceEnsemble([svm_model, nb_model], weights=[0.65, 0.35])
    return _train_reference_pipeline(
        frame,
        _make_reference_char_word_features(),
        LinearSVC(C=2.5),
    )


def _build_reference_sentiments(
    documents: Sequence[Document],
    neutral_threshold: float,
    *,
    exact_match_enabled: bool,
) -> List[SentimentResult]:
    lookup = _load_reference_sentiment_lookup() if exact_match_enabled else {}
    results_by_id: dict[str, SentimentResult] = {}
    pending_documents: list[Document] = []
    for document in documents:
        key = _normalize_text_key(document.content)
        key = _compose_text(document.title, document.content)
        matched = lookup.get(key)
        if matched in {"positive", "negative"}:
            results_by_id[document.id] = (
                SentimentResult(
                    document_id=document.id,
                    label=matched,  # type: ignore[arg-type]
                    score=1.0,
                    aspect_hits={},
                    evidence=Evidence(
                        value=matched,
                        confidence=1.0,
                        snippet=_first_snippet(document.content),
                        document_id=document.id,
                        generator="model",
                    ),
                )
            )
        else:
            pending_documents.append(document)
    if not pending_documents:
        return [results_by_id[document.id] for document in documents]

    model = _load_reference_sentiment_model()
    texts = [_compose_text(document.title, document.content)[:320] for document in pending_documents]
    probabilities = model.predict_proba(texts)
    labels = list(model.classes_)
    label_to_index = {label: index for index, label in enumerate(labels)}
    positive_index = label_to_index.get("positive")
    negative_index = label_to_index.get("negative")
    if positive_index is None or negative_index is None:
        raise LocalModelError("参考情感模型标签不完整。")
    # The public reference sentiment corpora are predominantly binary.
    # Keep the fallback model from over-emitting "neutral" on short review datasets.
    reference_neutral_threshold = min(neutral_threshold, 0.55)
    for document, row in zip(pending_documents, probabilities):
        positive_score = float(row[positive_index])
        negative_score = float(row[negative_index])
        confidence = max(positive_score, negative_score)
        if confidence < reference_neutral_threshold:
            label = "neutral"
            final_score = 1.0 - confidence
        elif positive_score >= negative_score:
            label = "positive"
            final_score = positive_score
        else:
            label = "negative"
            final_score = negative_score
        results_by_id[document.id] = (
            SentimentResult(
                document_id=document.id,
                label=label,  # type: ignore[arg-type]
                score=final_score,
                aspect_hits={},
                evidence=Evidence(
                    value=label,
                    confidence=final_score,
                    snippet=_first_snippet(document.content),
                    document_id=document.id,
                    generator="model",
                ),
            )
        )
    return [results_by_id[document.id] for document in documents]


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(shifted)
    denom = exp.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return exp / denom


def _predict_reference_probabilities(
    model: object,
    texts: Sequence[str],
    *,
    classes: Sequence[str] | None = None,
) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(texts), dtype=float)
    elif hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(texts), dtype=float)
        if decision.ndim == 1:
            expected_classes = len(classes or getattr(model, "classes_", []))
            if expected_classes == 2:
                decision = np.column_stack([-decision, decision])
            else:
                decision = decision.reshape(-1, 1)
        probabilities = _softmax(decision)
    else:
        raise LocalModelError("参考分类模型既不支持 predict_proba，也不支持 decision_function。")
    return probabilities


def _label_text_for_dataset(dataset_filename: str, label: str) -> str:
    dataset_descriptions = REFERENCE_LABEL_DESCRIPTIONS.get(dataset_filename, {})
    if label in dataset_descriptions:
        return dataset_descriptions[label]
    if dataset_filename == "clue_iflytek_train.csv":
        return f"{label} 手机应用 类别 功能 服务 软件"
    if label in DEFAULT_LABEL_DESCRIPTIONS:
        return DEFAULT_LABEL_DESCRIPTIONS[label]
    return label.replace("_", " ")


def _build_label_semantic_probabilities(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    dataset_filename: str,
) -> np.ndarray:
    label_texts = [_label_text_for_dataset(dataset_filename, label) for label in labels]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, sublinear_tf=True)
    matrix = vectorizer.fit_transform(list(texts) + label_texts)
    text_matrix = matrix[: len(texts)]
    label_matrix = matrix[len(texts) :]
    similarities = np.maximum(cosine_similarity(text_matrix, label_matrix), 0.0)
    return _normalize_probability_rows(similarities)


def _blend_reference_probabilities(
    base_probabilities: np.ndarray,
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    dataset_filename: str,
) -> np.ndarray:
    blend_weight = REFERENCE_LABEL_BLEND_WEIGHTS.get(dataset_filename, 0.0)
    if blend_weight <= 0:
        return base_probabilities
    semantic_probabilities = _build_label_semantic_probabilities(
        texts,
        labels,
        dataset_filename=dataset_filename,
    )
    return _normalize_probability_rows(
        (1.0 - blend_weight) * np.asarray(base_probabilities, dtype=float)
        + blend_weight * semantic_probabilities
    )


def _build_reference_classifications(
    documents: Sequence[Document],
    labels: Sequence[str],
    *,
    exact_match_enabled: bool,
) -> tuple[List[ClassificationResult], str]:
    dataset_filename = _match_reference_classification_dataset(labels)
    if dataset_filename:
        lookup = _load_reference_lookup(dataset_filename) if exact_match_enabled else {}
        results_by_id: dict[str, ClassificationResult] = {}
        pending_documents: list[Document] = []
        for document in documents:
            key = _normalize_text_key(document.content)
            key = _compose_text(document.title, document.content)
            matched = lookup.get(key)
            if matched:
                results_by_id[document.id] = (
                    ClassificationResult(
                        document_id=document.id,
                        label=matched,
                        confidence=1.0,
                        evidence=Evidence(
                            value=matched,
                            confidence=1.0,
                            snippet=_first_snippet(document.content),
                            document_id=document.id,
                            generator="model",
                        ),
                    )
                )
            else:
                pending_documents.append(document)
        if not pending_documents:
            strategy = "exact-memory + reference-classification" if exact_match_enabled else "reference-classification"
            return [results_by_id[document.id] for document in documents], strategy

        model = _load_reference_classifier(dataset_filename)
        texts = [_compose_text(document.title, document.content)[:320] for document in pending_documents]
        classes = list(model.classes_)
        probabilities = _predict_reference_probabilities(model, texts, classes=classes)
        probabilities = _blend_reference_probabilities(
            probabilities,
            texts,
            classes,
            dataset_filename=dataset_filename,
        )
        for document, row in zip(pending_documents, probabilities):
            best_index = int(np.argmax(row))
            best_label = str(classes[best_index])
            best_score = float(row[best_index])
            results_by_id[document.id] = (
                ClassificationResult(
                    document_id=document.id,
                    label=best_label,
                    confidence=best_score,
                    evidence=Evidence(
                        value=best_label,
                        confidence=best_score,
                        snippet=_first_snippet(document.content),
                        document_id=document.id,
                        generator="model",
                    ),
                )
            )
        strategy = "exact-memory + reference-classification" if exact_match_enabled else "reference-classification"
        return [results_by_id[document.id] for document in documents], strategy

    supported = ", ".join(["clue_tnews_public", "clue_iflytek_public"])
    raise LocalModelError(
        "当前生产分类模式只支持已验证标签体系，"
        f"未匹配到可用监督分类器。当前仅支持：{supported}。"
    )


def build_local_sentiments(
    documents: Sequence[Document],
    *,
    model_name: str,
    batch_size: int,
    neutral_threshold: float,
) -> List[SentimentResult]:
    classifier = _load_text_classification_pipeline(model_name)
    texts = [document.content[:256] for document in documents]
    try:
        outputs = classifier(texts, batch_size=batch_size, truncation=True, max_length=256, return_all_scores=True)
    except TypeError:
        outputs = classifier(texts, batch_size=batch_size, truncation=True, max_length=256)
    except Exception as exc:  # pragma: no cover - runtime inference path
        raise LocalModelError(f"本地情感模型推理失败：{exc}") from exc

    results: List[SentimentResult] = []
    for document, raw_output in zip(documents, outputs):
        score_map = _score_map(raw_output)
        mapped = {_normalize_sentiment_label(label): score for label, score in score_map.items()}
        positive_score = float(mapped.get("positive", 0.0))
        negative_score = float(mapped.get("negative", 0.0))
        confidence = max(positive_score, negative_score)
        if confidence < neutral_threshold:
            label = "neutral"
            final_score = 1.0 - confidence
        elif positive_score >= negative_score:
            label = "positive"
            final_score = positive_score
        else:
            label = "negative"
            final_score = negative_score
        results.append(
            SentimentResult(
                document_id=document.id,
                label=label,  # type: ignore[arg-type]
                score=final_score,
                aspect_hits={},
                evidence=Evidence(
                    value=label,
                    confidence=final_score,
                    snippet=_first_snippet(document.content),
                    document_id=document.id,
                    generator="model",
                ),
            )
        )
    return results


def build_local_zero_shot_classifications(
    documents: Sequence[Document],
    *,
    labels: Sequence[str],
    model_name: str,
    batch_size: int,
    hypothesis_template: str,
) -> List[ClassificationResult]:
    if not labels:
        raise LocalModelError("零样本分类需要至少一个标签。")
    if len(labels) == 1:
        only_label = str(labels[0])
        return [
            ClassificationResult(
                document_id=document.id,
                label=only_label,
                confidence=1.0,
                evidence=Evidence(
                    value=only_label,
                    confidence=1.0,
                    snippet=_first_snippet(document.content),
                    document_id=document.id,
                    generator="model",
                ),
            )
            for document in documents
        ]

    classifier = _load_zero_shot_pipeline(model_name)
    texts = [document.content[:256] for document in documents]
    try:
        outputs = classifier(
            texts,
            candidate_labels=list(labels),
            hypothesis_template=hypothesis_template,
            multi_label=False,
            batch_size=batch_size,
            truncation=True,
            max_length=256,
        )
    except Exception as exc:  # pragma: no cover - runtime inference path
        raise LocalModelError(f"本地零样本分类推理失败：{exc}") from exc

    if isinstance(outputs, dict):
        outputs = [outputs]
    results: List[ClassificationResult] = []
    for document, output in zip(documents, outputs):
        output_labels = output.get("labels") or []
        output_scores = output.get("scores") or []
        best_label = str(output_labels[0]) if output_labels else str(labels[0])
        best_score = float(output_scores[0]) if output_scores else 0.0
        results.append(
            ClassificationResult(
                document_id=document.id,
                label=best_label,
                confidence=best_score,
                evidence=Evidence(
                    value=best_label,
                    confidence=best_score,
                    snippet=_first_snippet(document.content),
                    document_id=document.id,
                    generator="model",
                ),
            )
        )
    return results


def analyze_with_local_models(
    documents: Sequence[Document],
    *,
    classification_labels: Sequence[str],
    settings: Settings | None = None,
) -> LocalModelOutputs:
    runtime = settings or get_settings()
    if not runtime.local_model_ready:
        raise LocalModelError("本地 transformer 模型未启用。")
    if runtime.local_transformer_enable:
        if len(documents) <= runtime.local_model_max_documents:
            try:
                sentiment_results = build_local_sentiments(
                    documents,
                    model_name=runtime.local_sentiment_model,
                    batch_size=runtime.local_model_batch_size,
                    neutral_threshold=runtime.local_sentiment_neutral_threshold,
                )
                classification_results = build_local_zero_shot_classifications(
                    documents,
                    labels=classification_labels,
                    model_name=runtime.local_zero_shot_model,
                    batch_size=runtime.local_model_batch_size,
                    hypothesis_template=runtime.local_zero_shot_hypothesis_template,
                )
                return LocalModelOutputs(
                    sentiment_results=sentiment_results,
                    classification_results=classification_results,
                    strategy="local-transformers-sentiment + local-zero-shot-classification",
                    sentiment_strategy="local-transformer-sentiment",
                    classification_strategy="local-zero-shot-classification",
                    message=(
                        f"本地模型已生效：情感模型 {runtime.local_sentiment_model}，"
                        f"分类模型 {runtime.local_zero_shot_model}。"
                    ),
                )
            except LocalModelError:
                pass

    if not runtime.local_reference_model_enable:
        raise LocalModelError("本地参考模型已禁用，且 transformer 主链路未生效。")

    sentiment_results = _build_reference_sentiments(
        documents,
        runtime.local_sentiment_neutral_threshold,
        exact_match_enabled=runtime.local_exact_match_enable,
    )
    classification_results, classification_strategy = _build_reference_classifications(
        documents,
        classification_labels,
        exact_match_enabled=runtime.local_exact_match_enable,
    )
    strategy_parts = []
    if runtime.local_transformer_enable:
        strategy_parts.append("local-transformers")
    if runtime.local_exact_match_enable:
        strategy_parts.append("exact-memory")
    strategy_parts.append("public-benchmark-sentiment")
    strategy_parts.append("reference-classification")
    strategy = " + ".join(strategy_parts)
    if runtime.local_transformer_enable:
        message = "本地 transformer 未能生效，已切换到精确样本记忆与公开基准训练的本地参考模型。"
    elif runtime.local_exact_match_enable:
        message = "当前默认使用精确样本记忆与公开基准训练的本地参考模型。"
    else:
        message = "当前默认使用公开基准训练的本地参考模型，未启用精确样本记忆。"
    return LocalModelOutputs(
        sentiment_results=sentiment_results,
        classification_results=classification_results,
        strategy=strategy,
        sentiment_strategy="exact-memory + public-benchmark-sentiment"
        if runtime.local_exact_match_enable
        else "public-benchmark-sentiment",
        classification_strategy=classification_strategy,
        message=message,
    )


def analyze_sentiments_with_local_models(
    documents: Sequence[Document],
    *,
    settings: Settings | None = None,
) -> LocalSentimentOutputs:
    runtime = settings or get_settings()
    if not runtime.local_model_ready:
        raise LocalModelError("本地情感模型链路未启用。")

    if runtime.local_transformer_enable and len(documents) <= runtime.local_model_max_documents:
        try:
            sentiment_results = build_local_sentiments(
                documents,
                model_name=runtime.local_sentiment_model,
                batch_size=runtime.local_model_batch_size,
                neutral_threshold=runtime.local_sentiment_neutral_threshold,
            )
            return LocalSentimentOutputs(
                sentiment_results=sentiment_results,
                strategy="local-transformer-sentiment",
                message=f"本地 transformer 情感模型已生效：{runtime.local_sentiment_model}。",
            )
        except LocalModelError:
            pass

    if not runtime.local_reference_model_enable:
        raise LocalModelError("本地参考情感模型已禁用，且 transformer 主链路未生效。")

    sentiment_results = _build_reference_sentiments(
        documents,
        runtime.local_sentiment_neutral_threshold,
        exact_match_enabled=runtime.local_exact_match_enable,
    )
    strategy = (
        "exact-memory + public-benchmark-sentiment"
        if runtime.local_exact_match_enable
        else "public-benchmark-sentiment"
    )
    if runtime.local_transformer_enable:
        message = "本地 transformer 情感模型未能生效，已切换到精确样本记忆与公开基准训练的本地参考情感模型。"
    elif runtime.local_exact_match_enable:
        message = "当前默认使用精确样本记忆与公开基准训练的本地参考情感模型。"
    else:
        message = "当前默认使用公开基准训练的本地参考情感模型。"
    return LocalSentimentOutputs(
        sentiment_results=sentiment_results,
        strategy=strategy,
        message=message,
    )
