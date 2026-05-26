from __future__ import annotations

from datetime import datetime
from typing import Iterable, List

from ..models import ClassificationResult, InsightCard, SentimentResult, TopicCluster


def build_report_markdown(
    dataset_name: str,
    topics: Iterable[TopicCluster],
    sentiments: List[SentimentResult],
    classifications: List[ClassificationResult],
    cards: List[InsightCard],
) -> str:
    sample_count = len(sentiments) or len(classifications)
    positive_ratio = sum(1 for item in sentiments if item.label == "positive") / max(len(sentiments), 1)
    negative_ratio = sum(1 for item in sentiments if item.label == "negative") / max(len(sentiments), 1)
    dominant_class = max(
        {item.label: sum(1 for result in classifications if result.label == item.label) for item in classifications}.items(),
        key=lambda pair: pair[1],
        default=("未分类", 0),
    )
    topic_lines = "\n".join(
        f"- **{topic.name}**：{topic.summary} 关键词：{', '.join(topic.keywords[:5])}"
        for topic in topics
    )
    card_lines = "\n".join(
        f"- **{card.title}**：{card.summary} 证据：{card.evidences[0].snippet if card.evidences else '暂无'}"
        for card in cards[:6]
    )
    return f"""# {dataset_name} 研究报告

生成时间：{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

## 一、总体结论

- 样本量：{sample_count} 条文本
- 正向占比：{positive_ratio:.1%}
- 负向占比：{negative_ratio:.1%}
- 主导分类：{dominant_class[0]}

## 二、主题洞察

{topic_lines or '- 暂无主题结果'}

## 三、重点发现

{card_lines or '- 暂无洞察卡片'}

## 四、方法说明

- 基础层：中文分词、词频、共词、规则情感与关键词分类
- 语义层：TF-IDF 向量与 KMeans 聚类，自动生成主题摘要
- 报告层：基于证据片段生成可回溯研究简报
"""


def build_outline(cards: List[InsightCard]) -> List[str]:
    return [
        "研究背景与样本说明",
        "高频主题与语义聚类",
        "情感与分类结果",
        *[card.title for card in cards[:3]],
        "结论与下一步建议",
    ]
