from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

import httpx

from ..models import InsightCard, TopicCluster
from ..settings import Settings, get_settings


class BailianClientError(RuntimeError):
    """Raised when the Bailian request or response parsing fails."""


JSON_BLOCK_PATTERN = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def _extract_json(content: str) -> Any:
    match = JSON_BLOCK_PATTERN.search(content)
    candidate = match.group(0) if match else content
    return json.loads(candidate)


def _post_chat_completion(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    settings: Settings = get_settings()
    if not settings.llm_ready:
        raise BailianClientError("Bailian LLM is not configured.")
    request_kwargs = {
        "headers": {
            "Authorization": f"Bearer {settings.dashscope_api_key}",
            "Content-Type": "application/json",
        },
        "json": {
            "model": settings.dashscope_model,
            "messages": messages,
            "temperature": temperature,
        },
        "timeout": httpx.Timeout(settings.dashscope_timeout_seconds, connect=15.0),
    }
    try:
        response = httpx.post(f"{settings.dashscope_base_url}/chat/completions", **request_kwargs)
    except httpx.ReadTimeout:
        response = httpx.post(f"{settings.dashscope_base_url}/chat/completions", **request_kwargs)
    response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        raise BailianClientError("Bailian returned no choices.")
    return choices[0]["message"]["content"]


def enrich_topics_and_report(
    *,
    dataset_name: str,
    top_terms: List[Dict[str, Any]],
    topics: List[TopicCluster],
    insight_cards: List[InsightCard],
    sentiment_summary: Dict[str, int],
    classification_summary: Dict[str, int],
) -> Optional[Dict[str, Any]]:
    if not topics:
        return None
    payload = {
        "dataset_name": dataset_name,
        "top_terms": [item["term"] for item in top_terms[:20]],
        "sentiment_summary": sentiment_summary,
        "classification_summary": classification_summary,
        "topic_candidates": [
            {
                "topic_id": topic.topic_id,
                "name": topic.name,
                "size": topic.size,
                "size_ratio": round(topic.size / max(sum(item.size for item in topics), 1), 4),
                "keywords": topic.keywords[:8],
                "evidences": [evidence.snippet[:120] for evidence in topic.evidences[:2]],
            }
            for topic in topics[:6]
        ],
        "existing_cards": [{"title": card.title, "summary": card.summary, "kind": card.kind} for card in insight_cards[:6]],
    }
    system_prompt = (
        "你是中文研究分析平台的资深分析师。"
        "你需要为主题簇重新命名、补充更自然的摘要，并基于提供的统计与证据写一份简洁可信的研究报告。"
        "你必须严格返回 JSON 对象，不要使用 Markdown 代码块。"
        "只能使用输入里已经提供的统计、条数、占比和证据，不得编造新的百分比、条数、因果关系或相关性结论。"
        "如果证据不足，请使用“从样本片段看”或“可初步判断”为表述。"
    )
    user_prompt = (
        "请基于下面的数据返回 JSON，字段结构如下："
        '{"topic_overrides":[{"topic_id":"topic_1","name":"主题名","summary":"主题摘要"}],'
        '"insight_cards":[{"title":"洞察标题","summary":"一句洞察","kind":"report","topic_id":"topic_1"}],'
        '"report_markdown":"# 报告标题\\n..."}'
        "。主题名要像分析师命名，不要只是关键词拼接；报告中要包含总体结论、主题分析、情绪观察和建议。"
        "报告写作要求：1. 优先写事实判断和运营建议；2. 不要写没有输入支撑的行业黑话；3. 不要虚构精确比例。"
        f"\n\n数据：{json.dumps(payload, ensure_ascii=False)}"
    )
    raw = _post_chat_completion(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.25,
    )
    return _extract_json(raw)


def name_topic_clusters(*, dataset_name: str, topics: List[TopicCluster]) -> List[Dict[str, Any]]:
    if not topics:
        return []
    payload = {
        "dataset_name": dataset_name,
        "topic_candidates": [
            {
                "topic_id": topic.topic_id,
                "index_name": topic.name,
                "size": topic.size,
                "keywords": topic.keywords[:10],
                "evidences": [evidence.snippet[:140] for evidence in topic.evidences[:3]],
            }
            for topic in topics
        ],
    }
    system_prompt = (
        "你是中文文本分析平台的主题命名助手。"
        "你的任务是根据每个主题簇的高频关键词和少量原文片段，给出简洁、正式、可作为分类标签候选的中文名称。"
        "你必须严格返回 JSON 数组，不要使用 Markdown 代码块。"
        "不得逐条标注文本，不得新增输入中没有依据的主题，不得虚构数量、占比或结论。"
        "如果主题过于宽泛或证据不足，名称应体现保守判断，例如“综合评价”或“待确认主题”。"
    )
    user_prompt = (
        "请为下面每个主题返回命名建议，字段结构如下："
        '[{"topic_id":"topic_1","name":"主题名称","summary":"一句话说明","confidence":0.0}]'
        "。name 控制在 2-8 个汉字，summary 控制在 40 字以内。"
        f"\n\n数据：{json.dumps(payload, ensure_ascii=False)}"
    )
    raw = _post_chat_completion(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.15,
    )
    payload = _extract_json(raw)
    if not isinstance(payload, list):
        raise BailianClientError("Bailian topic naming did not return a JSON array.")
    return [item for item in payload if isinstance(item, dict)]


def analyze_documents(
    *,
    documents: Sequence[Dict[str, str]],
    classification_labels: Sequence[str],
    batch_size: int = 12,
) -> List[Dict[str, Any]]:
    if not documents:
        return []
    label_text = "、".join(classification_labels)
    system_prompt = (
        "你是中文文本分析引擎。"
        "你需要对每条文本同时完成主题分类和情感判别。"
        "你必须严格返回 JSON 数组，不要使用 Markdown 代码块。"
        "分类标签必须严格从给定标签集中选择；情感标签只能是 positive、neutral、negative。"
        "证据片段必须直接摘自原文，长度控制在 10-40 个字。"
    )
    outputs: List[Dict[str, Any]] = []
    for start in range(0, len(documents), batch_size):
        batch = list(documents[start : start + batch_size])
        user_prompt = (
            "请分析下面这批文本。分类标签集合为："
            f"{label_text}。\n"
            "对每条文本返回如下字段："
            '[{"document_id":"...","classification_label":"...","classification_confidence":0.0,'
            '"classification_evidence":"...","sentiment_label":"positive","sentiment_confidence":0.0,'
            '"sentiment_evidence":"...","aspect_sentiment":{"产品":0.0,"服务":0.0,"价格":0.0}}]'
            "。不要输出额外解释。"
            f"\n\n文本批次：{json.dumps(batch, ensure_ascii=False)}"
        )
        raw = _post_chat_completion(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
        )
        payload = _extract_json(raw)
        if not isinstance(payload, list):
            raise BailianClientError("Bailian batch analysis did not return a JSON array.")
        outputs.extend(payload)
    return outputs
