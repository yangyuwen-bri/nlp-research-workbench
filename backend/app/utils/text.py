from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import jieba.posseg as pseg


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

POSITIVE_WORDS = {
    "喜欢": 1.6,
    "满意": 1.8,
    "优秀": 1.7,
    "值得": 1.2,
    "方便": 1.1,
    "稳定": 1.2,
    "专业": 1.2,
    "推荐": 1.6,
}

NEGATIVE_WORDS = {
    "失望": -1.8,
    "糟糕": -1.7,
    "麻烦": -1.1,
    "卡顿": -1.5,
    "崩溃": -2.0,
    "困难": -1.1,
    "昂贵": -1.0,
    "垃圾": -2.2,
}

NEGATIONS = {"不", "没", "无", "非", "别"}
BOOSTERS = {"很": 1.2, "非常": 1.4, "太": 1.3, "特别": 1.35}

DEFAULT_LABEL_KEYWORDS = {
    "产品体验": ["体验", "功能", "界面", "稳定", "卡顿", "bug"],
    "价格感知": ["价格", "优惠", "性价比", "昂贵", "便宜", "活动"],
    "服务反馈": ["客服", "服务", "售后", "响应", "退货", "物流"],
}

ASPECT_KEYWORDS = {
    "产品": ["产品", "功能", "性能", "界面", "系统"],
    "服务": ["客服", "售后", "响应", "服务", "沟通"],
    "价格": ["价格", "性价比", "便宜", "优惠", "折扣"],
}


def split_sentences(text: str) -> List[str]:
    return [segment.strip() for segment in re.split(r"[。！？!\n]+", text) if segment.strip()]


def tokenize(text: str) -> List[Tuple[str, str]]:
    tokens: List[Tuple[str, str]] = []
    for word, flag in pseg.cut(text):
        word = word.strip()
        if len(word) < 2:
            continue
        if word in DEFAULT_STOPWORDS:
            continue
        if re.fullmatch(r"[\W_]+", word):
            continue
        tokens.append((word, flag))
    return tokens


def build_term_stats(tokenized_documents: Sequence[List[Tuple[str, str]]]) -> Tuple[List[Dict[str, object]], Counter]:
    total_counter: Counter = Counter()
    document_counter: Counter = Counter()
    pos_map: Dict[str, str] = {}
    for tokens in tokenized_documents:
        unique_terms = set()
        for word, flag in tokens:
            total_counter[word] += 1
            pos_map.setdefault(word, flag)
            unique_terms.add(word)
        for term in unique_terms:
            document_counter[term] += 1
    stats = [
        {
            "term": term,
            "term_frequency": count,
            "document_frequency": document_counter[term],
            "pos": pos_map.get(term, "n"),
        }
        for term, count in total_counter.most_common()
    ]
    return stats, total_counter


def select_terms(term_stats: Sequence[Dict[str, object]], top_k: int) -> List[Dict[str, object]]:
    shortlist = [item for item in term_stats if str(item["pos"]).startswith(("n", "v", "a"))]
    return list(shortlist[:top_k])


def build_binary_matrix(
    texts: Sequence[str], tokenized_documents: Sequence[List[Tuple[str, str]]], selected_terms: Sequence[str]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for index, (text, tokens) in enumerate(zip(texts, tokenized_documents), start=1):
        token_set = {word for word, _ in tokens}
        row = {"row_id": index, "content": text}
        for term in selected_terms:
            row[term] = 1 if term in token_set else 0
        rows.append(row)
    return rows


def build_match_rows(texts: Sequence[str], tokenized_documents: Sequence[List[Tuple[str, str]]], selected_terms: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for index, (text, tokens) in enumerate(zip(texts, tokenized_documents), start=1):
        token_list = [word for word, _ in tokens]
        matched = [term for term in selected_terms if term in token_list]
        rows.append({"row_id": index, "content": text, "matched_terms": ", ".join(matched)})
    return rows


def build_cooccurrence_edges(binary_matrix: Sequence[Dict[str, object]], selected_terms: Sequence[str]) -> List[Dict[str, object]]:
    edge_counter: Counter = Counter()
    for row in binary_matrix:
        active = [term for term in selected_terms if row.get(term) == 1]
        for index, source in enumerate(active):
            for target in active[index + 1 :]:
                edge_counter[(source, target)] += 1
    return [
        {"source": source, "target": target, "weight": weight}
        for (source, target), weight in edge_counter.most_common(120)
    ]


def score_sentiment(text: str) -> Tuple[float, Dict[str, float], str]:
    score = 0.0
    aspect_hits = {key: 0.0 for key in ASPECT_KEYWORDS}
    chosen_sentence = split_sentences(text)[0] if split_sentences(text) else text[:80]
    sentences = split_sentences(text) or [text]
    best_abs = 0.0
    for sentence in sentences:
        tokens = [word for word, _ in tokenize(sentence)]
        sentence_score = 0.0
        for idx, token in enumerate(tokens):
            multiplier = 1.0
            if idx > 0 and tokens[idx - 1] in NEGATIONS:
                multiplier *= -1
            if idx > 0 and tokens[idx - 1] in BOOSTERS:
                multiplier *= BOOSTERS[tokens[idx - 1]]
            sentence_score += POSITIVE_WORDS.get(token, 0.0) * multiplier
            sentence_score += NEGATIVE_WORDS.get(token, 0.0) * multiplier
        for aspect, keywords in ASPECT_KEYWORDS.items():
            if any(keyword in sentence for keyword in keywords):
                aspect_hits[aspect] += sentence_score
        score += sentence_score
        if abs(sentence_score) >= best_abs:
            best_abs = abs(sentence_score)
            chosen_sentence = sentence
    return score, aspect_hits, chosen_sentence


def normalize_score(score: float) -> float:
    return 1 / (1 + math.exp(-score)) if score else 0.5


def classify_text(text: str, labels: Dict[str, List[str]] | None = None) -> Tuple[str, float, str]:
    catalog = labels or DEFAULT_LABEL_KEYWORDS
    scores: Dict[str, float] = {}
    for label, keywords in catalog.items():
        scores[label] = sum(1.0 for keyword in keywords if keyword in text)
    label = max(scores, key=scores.get)
    confidence = 0.4 + min(scores[label] * 0.15, 0.55)
    evidence = next((keyword for keyword in catalog[label] if keyword in text), text[:40])
    return label, confidence, evidence


def summarize_keywords(keywords: Iterable[str]) -> str:
    keyword_list = list(keywords)
    if not keyword_list:
        return "未识别出稳定主题词。"
    return f"围绕 {keyword_list[0]}、{keyword_list[1] if len(keyword_list) > 1 else keyword_list[0]} 等关键词形成稳定讨论。"
