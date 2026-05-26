from __future__ import annotations

import numpy as np

from app.models import Document
from app.services.topic_models import build_topics


def _documents() -> list[Document]:
    return [
        Document(id="doc1", dataset_id="ds1", source_row=1, content="奶茶配送很快，包装很好。"),
        Document(id="doc2", dataset_id="ds1", source_row=2, content="咖啡口感不错，配送也及时。"),
        Document(id="doc3", dataset_id="ds1", source_row=3, content="财经新闻关注利率和市场走势。"),
        Document(id="doc4", dataset_id="ds1", source_row=4, content="股票行情和基金投资热度上升。"),
    ]


def test_build_topics_uses_tfidf_kmeans_as_single_production_mode(monkeypatch):
    topics, strategy = build_topics(_documents(), topic_count=2, allow_embeddings=False)

    assert strategy == "tfidf_kmeans"
    assert len(topics) == 2
    assert {topic.name for topic in topics} == {"主题 1", "主题 2"}
    assert all(topic.keywords for topic in topics)
    assert all(topic.topic_id.startswith("topic_") for topic in topics)


def test_build_topics_falls_back_to_tfidf_kmeans_when_bertopic_is_unavailable(monkeypatch):
    monkeypatch.setattr("app.services.topic_models._load_bertopic", lambda: (None, None))

    topics, strategy = build_topics(_documents(), topic_count=2, allow_embeddings=False)

    assert strategy == "tfidf_kmeans"
    assert len(topics) == 2
    assert all(topic.size > 0 for topic in topics)
    assert all(topic.keywords for topic in topics)


def test_build_topics_falls_back_to_embedding_kmeans_when_bertopic_errors(monkeypatch):
    class ExplodingBERTopic:
        def __init__(self, **kwargs):
            pass

        def fit_transform(self, texts, embeddings=None):
            raise RuntimeError("boom")

    monkeypatch.setattr("app.services.topic_models._load_bertopic", lambda: (ExplodingBERTopic, None))
    monkeypatch.setattr(
        "app.services.topic_models.embed_texts",
        lambda texts: np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
            ]
        ),
    )

    topics, strategy = build_topics(_documents(), topic_count=2, allow_embeddings=True)

    assert strategy == "dashscope_embedding_kmeans"
    assert len(topics) == 2
    assert all(topic.evidences for topic in topics)
