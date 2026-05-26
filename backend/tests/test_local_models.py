from __future__ import annotations

from types import SimpleNamespace

from app.models import ClassificationResult, Document, Evidence, LabelSchema, SentimentResult
from app.services import analyze as analyze_module
from app.services.local_models import (
    LocalModelOutputs,
    _blend_reference_probabilities,
    _build_reference_classifications,
    _build_reference_sentiments,
    _match_reference_classification_dataset,
    _predict_reference_probabilities,
    analyze_with_local_models,
    build_local_sentiments,
    build_local_zero_shot_classifications,
)


def test_build_local_sentiments_maps_binary_scores_to_neutral_and_polar(monkeypatch):
    documents = [
        Document(id="doc1", dataset_id="ds1", source_row=1, content="这个产品还行。"),
        Document(id="doc2", dataset_id="ds1", source_row=2, content="这个产品非常差。"),
    ]

    def fake_pipeline(texts, **kwargs):
        assert len(texts) == 2
        return [
            [
                {"label": "Negative", "score": 0.48},
                {"label": "Positive", "score": 0.52},
            ],
            [
                {"label": "Negative", "score": 0.91},
                {"label": "Positive", "score": 0.09},
            ],
        ]

    monkeypatch.setattr("app.services.local_models._load_text_classification_pipeline", lambda model_name: fake_pipeline)
    results = build_local_sentiments(
        documents,
        model_name="fake-sentiment",
        batch_size=2,
        neutral_threshold=0.60,
    )
    assert [item.label for item in results] == ["neutral", "negative"]


def test_build_local_zero_shot_classifications_uses_top_label(monkeypatch):
    documents = [
        Document(id="doc1", dataset_id="ds1", source_row=1, content="物流太慢了。"),
        Document(id="doc2", dataset_id="ds1", source_row=2, content="价格挺便宜。"),
    ]

    def fake_pipeline(texts, **kwargs):
        assert kwargs["candidate_labels"] == ["服务反馈", "价格感知"]
        return [
            {"labels": ["服务反馈", "价格感知"], "scores": [0.81, 0.19]},
            {"labels": ["价格感知", "服务反馈"], "scores": [0.74, 0.26]},
        ]

    monkeypatch.setattr("app.services.local_models._load_zero_shot_pipeline", lambda model_name: fake_pipeline)
    results = build_local_zero_shot_classifications(
        documents,
        labels=["服务反馈", "价格感知"],
        model_name="fake-zero-shot",
        batch_size=2,
        hypothesis_template="这段文本主要属于{}。",
    )
    assert [item.label for item in results] == ["服务反馈", "价格感知"]


def test_run_analysis_prefers_local_models(monkeypatch):
    from app.services.analyze import run_analysis
    from app.services.ingest import ingest_dataset

    sample = """标题,正文
评论1,物流太慢了，客服也没回。
评论2,价格便宜，整体不错。
"""
    dataset, documents = ingest_dataset("sample.csv", sample.encode("utf-8"))

    monkeypatch.setattr(
        analyze_module,
        "get_settings",
        lambda: SimpleNamespace(
            local_model_ready=True,
            local_transformer_enable=False,
            local_sentiment_model="fake-sentiment",
            local_zero_shot_model="fake-zero-shot",
            llm_ready=False,
            embedding_ready=False,
            dashscope_model="",
            dashscope_embedding_model="",
        ),
    )
    monkeypatch.setattr(
        analyze_module,
        "analyze_with_local_models",
        lambda documents, classification_labels, settings: LocalModelOutputs(
            sentiment_results=[
                SentimentResult(
                    document_id=documents[0].id,
                    label="negative",
                    score=0.9,
                    aspect_hits={},
                    evidence=Evidence(
                        value="negative",
                        confidence=0.9,
                        snippet="物流太慢了",
                        document_id=documents[0].id,
                        generator="model",
                    ),
                ),
                SentimentResult(
                    document_id=documents[1].id,
                    label="positive",
                    score=0.88,
                    aspect_hits={},
                    evidence=Evidence(
                        value="positive",
                        confidence=0.88,
                        snippet="整体不错",
                        document_id=documents[1].id,
                        generator="model",
                    ),
                ),
            ],
            classification_results=[
                ClassificationResult(
                    document_id=documents[0].id,
                    label="服务反馈",
                    confidence=0.91,
                    evidence=Evidence(
                        value="服务反馈",
                        confidence=0.91,
                        snippet="客服也没回",
                        document_id=documents[0].id,
                        generator="model",
                    ),
                ),
                ClassificationResult(
                    document_id=documents[1].id,
                    label="价格感知",
                    confidence=0.86,
                    evidence=Evidence(
                        value="价格感知",
                        confidence=0.86,
                        snippet="价格便宜",
                        document_id=documents[1].id,
                        generator="model",
                    ),
                ),
            ],
            strategy="local-transformers-sentiment + local-zero-shot-classification",
            sentiment_strategy="local-transformer-sentiment",
            classification_strategy="local-zero-shot-classification",
            message="本地模型已生效。",
        ),
    )

    request = SimpleNamespace(
        dataset_id=dataset.id,
        analysis_stage="classify",
        top_k_terms=8,
        topic_count=2,
        label_schema=LabelSchema(
            id="confirmed",
            name="确认分类",
            description="用户确认后的分类标签",
            labels=["服务反馈", "价格感知"],
        ),
        use_llm=False,
        smart_topic_names=False,
        write_exports=False,
        export_xlsx=False,
    )
    run, _ = run_analysis(dataset, documents, request)
    assert run.outputs is not None
    assert run.outputs.semantic_execution.used is True
    assert run.outputs.semantic_execution.strategy == "local-zero-shot-classification"
    assert [item.label for item in run.outputs.classification_results] == ["服务反馈", "价格感知"]


def test_reference_classification_prefers_exact_text_lookup(monkeypatch):
    documents = [
        Document(id="doc1", dataset_id="ds1", source_row=1, content="天气真的很好"),
        Document(id="doc2", dataset_id="ds1", source_row=2, content="体育比赛直播开始"),
    ]
    monkeypatch.setattr(
        "app.services.local_models._match_reference_classification_dataset",
        lambda labels: "clue_tnews_train.csv",
    )
    monkeypatch.setattr(
        "app.services.local_models._load_reference_lookup",
        lambda filename: {"天气真的很好": "news_weather", "体育比赛直播开始": "news_sports"},
    )

    class ShouldNotRun:
        classes_ = ["news_weather", "news_sports"]

        def decision_function(self, texts):
            raise AssertionError("Exact lookup should have handled all rows")

    monkeypatch.setattr("app.services.local_models._load_reference_classifier", lambda filename: ShouldNotRun())
    results, strategy = _build_reference_classifications(
        documents,
        labels=["news_weather", "news_sports"],
        exact_match_enabled=True,
    )
    assert [item.label for item in results] == ["news_weather", "news_sports"]
    assert strategy == "exact-memory + reference-classification"


def test_match_reference_classification_dataset_allows_label_subsets(monkeypatch):
    monkeypatch.setattr(
        "app.services.local_models._load_reference_frame",
        lambda filename: {
            "clue_tnews_train.csv": __import__("pandas").DataFrame({"标签": ["a", "b", "c"]}),
            "clue_iflytek_train.csv": __import__("pandas").DataFrame({"标签": ["x", "y", "z", "w"]}),
        }[filename],
    )
    monkeypatch.setattr("app.services.local_models.BENCH_DATA_DIR", __import__("pathlib").Path("/tmp"))
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
    assert _match_reference_classification_dataset(["x", "z"]) == "clue_iflytek_train.csv"


def test_reference_classification_can_disable_exact_match(monkeypatch):
    documents = [Document(id="doc1", dataset_id="ds1", source_row=1, title="关键词", content="天气真的很好")]
    monkeypatch.setattr(
        "app.services.local_models._match_reference_classification_dataset",
        lambda labels: "clue_tnews_train.csv",
    )
    monkeypatch.setattr(
        "app.services.local_models._load_reference_lookup",
        lambda filename: {"关键词 天气真的很好": "news_weather"},
    )

    class FakeClassifier:
        classes_ = ["news_weather", "news_sports"]

        def decision_function(self, texts):
            import numpy as np

            return np.array([[0.1, 0.9]])

    monkeypatch.setattr("app.services.local_models._load_reference_classifier", lambda filename: FakeClassifier())
    results, strategy = _build_reference_classifications(
        documents,
        labels=["news_weather", "news_sports"],
        exact_match_enabled=False,
    )
    assert [item.label for item in results] == ["news_sports"]
    assert strategy == "reference-classification"


def test_reference_probabilities_prefer_predict_proba_when_available():
    class FakeProbModel:
        classes_ = ["news_weather", "news_sports"]

        def predict_proba(self, texts):
            assert texts == ["天气真的很好"]
            return [[0.75, 0.25]]

    probabilities = _predict_reference_probabilities(FakeProbModel(), ["天气真的很好"])
    assert probabilities.shape == (1, 2)
    assert probabilities[0][0] == 0.75


def test_blend_reference_probabilities_can_use_label_semantics():
    base = __import__("numpy").array([[0.45, 0.55]])
    blended = _blend_reference_probabilities(
        base,
        ["今天科技公司发布了新手机"],
        ["news_finance", "news_tech"],
        dataset_filename="clue_tnews_train.csv",
    )
    assert blended.shape == (1, 2)
    assert blended[0][1] > blended[0][0]


def test_reference_sentiment_caps_neutral_threshold_for_binary_review_sets(monkeypatch):
    documents = [Document(id="doc1", dataset_id="ds1", source_row=1, content="味道还可以，下次再来。")]
    monkeypatch.setattr("app.services.local_models._load_reference_sentiment_lookup", lambda: {})

    class FakeSentimentModel:
        classes_ = ["negative", "positive"]

        def predict_proba(self, texts):
            return [[0.43, 0.57]]

    monkeypatch.setattr("app.services.local_models._load_reference_sentiment_model", lambda: FakeSentimentModel())
    results = _build_reference_sentiments(documents, 0.60, exact_match_enabled=False)
    assert [item.label for item in results] == ["positive"]


def test_reference_classification_rejects_unsupported_label_schemas():
    documents = [Document(id="doc1", dataset_id="ds1", source_row=1, content="客服响应很慢，物流也有问题")]
    try:
        _build_reference_classifications(
            documents,
            labels=["产品体验", "价格感知", "服务反馈"],
            exact_match_enabled=False,
        )
    except Exception as exc:
        assert "当前生产分类模式只支持已验证标签体系" in str(exc)
    else:
        raise AssertionError("Unsupported schemas should fail fast in production mode.")


def test_analyze_with_local_models_keeps_reference_models_available_for_large_batches(monkeypatch):
    documents = [
        Document(id=f"doc{i}", dataset_id="ds1", source_row=i, content=f"评论 {i} 体验不错")
        for i in range(600)
    ]
    monkeypatch.setattr(
        "app.services.local_models._build_reference_sentiments",
        lambda *args, **kwargs: [
            SentimentResult(
                document_id=document.id,
                label="positive",
                score=0.9,
                aspect_hits={},
                evidence=Evidence(
                    value="positive",
                    confidence=0.9,
                    snippet="体验不错",
                    document_id=document.id,
                    generator="model",
                ),
            )
            for document in documents
        ],
    )
    monkeypatch.setattr(
        "app.services.local_models._build_reference_classifications",
        lambda *args, **kwargs: (
            [
                ClassificationResult(
                    document_id=document.id,
                    label="产品体验",
                    confidence=0.8,
                    evidence=Evidence(
                        value="产品体验",
                        confidence=0.8,
                        snippet="体验不错",
                        document_id=document.id,
                        generator="model",
                    ),
                )
                for document in documents
            ],
            "reference-classification",
        ),
    )

    settings = SimpleNamespace(
        local_model_ready=True,
        local_transformer_enable=True,
        local_reference_model_enable=True,
        local_model_max_documents=512,
        local_model_batch_size=8,
        local_sentiment_model="fake-sentiment",
        local_zero_shot_model="fake-zero-shot",
        local_sentiment_neutral_threshold=0.6,
        local_zero_shot_hypothesis_template="这段文本主要属于{}。",
        local_exact_match_enable=False,
    )
    outputs = analyze_with_local_models(
        documents,
        classification_labels=["产品体验", "价格感知", "服务反馈"],
        settings=settings,
    )
    assert outputs.sentiment_strategy == "public-benchmark-sentiment"
    assert outputs.classification_strategy == "reference-classification"
    assert len(outputs.sentiment_results) == len(documents)
    assert len(outputs.classification_results) == len(documents)
