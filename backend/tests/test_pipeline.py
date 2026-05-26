from app.services.analyze import run_analysis
from app.services.ingest import ingest_dataset
from app.models import DatasetWorkspace, LabelSchema, RunAnalysisRequest, SynonymGroup
from app.services.workspace import build_workspace_snapshot


SAMPLE = """标题,正文
评论1,这个产品体验很好，界面稳定，客服也很专业。
评论2,价格有点昂贵，但是功能确实不错。
评论3,售后响应太慢了，体验有些失望。
"""


def test_ingest_and_analysis_pipeline():
    dataset, documents = ingest_dataset("sample.csv", SAMPLE.encode("utf-8"))
    request = type(
        "Request",
        (),
        {
            "dataset_id": dataset.id,
            "analysis_stage": "full",
            "top_k_terms": 12,
            "topic_count": 2,
            "label_schema": None,
            "use_llm": False,
        },
    )()
    run, report = run_analysis(dataset, documents, request)
    assert dataset.document_count == 3
    assert run.status == "completed"
    assert run.outputs is not None
    assert len(run.outputs.top_terms) > 0
    assert len(run.outputs.topics) > 0
    assert run.outputs.classification_results == []
    assert run.settings["analysis_stage"] == "full"
    assert run.settings["classification_strategy"] == "unsupported"
    assert "研究报告" in report.markdown


def test_discover_stage_runs_topics_without_exploration_or_sentiment():
    dataset, documents = ingest_dataset("sample.csv", SAMPLE.encode("utf-8"))
    request = RunAnalysisRequest(
        dataset_id=dataset.id,
        analysis_stage="discover",
        top_k_terms=12,
        topic_count=2,
        use_llm=False,
        write_exports=False,
        export_xlsx=False,
    )

    run, _ = run_analysis(dataset, documents, request)

    assert run.outputs is not None
    assert run.outputs.top_terms == []
    assert run.outputs.tokenized_documents == []
    assert run.outputs.sentiment_results == []
    assert run.outputs.classification_results == []
    assert len(run.outputs.topics) > 0
    assert run.settings["analysis_stage"] == "discover"
    assert run.settings["topic_strategy"] != "not_requested"
    assert run.settings["sentiment_strategy"] == "not_requested"
    assert run.settings["classification_strategy"] == "not_requested"


def test_explore_stage_runs_text_tables_without_high_level_outputs():
    dataset, documents = ingest_dataset("sample.csv", SAMPLE.encode("utf-8"))
    request = RunAnalysisRequest(
        dataset_id=dataset.id,
        analysis_stage="explore",
        top_k_terms=12,
        topic_count=2,
        use_llm=False,
        write_exports=False,
        export_xlsx=False,
    )

    run, _ = run_analysis(dataset, documents, request)

    assert run.outputs is not None
    assert len(run.outputs.top_terms) > 0
    assert len(run.outputs.tokenized_documents) == 3
    assert len(run.outputs.selected_terms) > 0
    assert len(run.outputs.match_rows) > 0
    assert len(run.outputs.binary_matrix) == 3
    assert len(run.outputs.frequency_matrix) == 3
    assert run.outputs.topics == []
    assert run.outputs.sentiment_results == []
    assert run.outputs.classification_results == []
    assert run.settings["analysis_stage"] == "explore"
    assert run.settings["topic_strategy"] == "not_requested"
    assert run.settings["sentiment_strategy"] == "not_requested"
    assert run.settings["classification_strategy"] == "not_requested"


def test_sentiment_stage_runs_only_sentiment():
    dataset, documents = ingest_dataset("sample.csv", SAMPLE.encode("utf-8"))
    request = RunAnalysisRequest(
        dataset_id=dataset.id,
        analysis_stage="sentiment",
        top_k_terms=12,
        topic_count=2,
        use_llm=False,
        write_exports=False,
        export_xlsx=False,
    )

    run, _ = run_analysis(dataset, documents, request)

    assert run.outputs is not None
    assert run.outputs.top_terms == []
    assert run.outputs.tokenized_documents == []
    assert run.outputs.selected_terms == []
    assert run.outputs.match_rows == []
    assert run.outputs.binary_matrix == []
    assert run.outputs.cooccurrence_edges == []
    assert len(run.outputs.sentiment_results) == 3
    assert run.outputs.topics == []
    assert run.outputs.classification_results == []
    assert run.settings["analysis_stage"] == "sentiment"
    assert run.settings["topic_strategy"] == "not_requested"
    assert run.settings["sentiment_strategy"] == "public-benchmark-sentiment"
    assert run.settings["classification_strategy"] == "not_requested"


def test_classification_stage_requires_confirmed_labels():
    dataset, documents = ingest_dataset("sample.csv", SAMPLE.encode("utf-8"))
    request = RunAnalysisRequest(dataset_id=dataset.id, analysis_stage="classify", use_llm=False, write_exports=False)

    try:
        run_analysis(dataset, documents, request)
    except ValueError as exc:
        assert "标签名单" in str(exc)
    else:
        raise AssertionError("classification stage should require labels")


def test_classification_stage_runs_confirmed_label_schema(monkeypatch):
    dataset, documents = ingest_dataset("sample.csv", SAMPLE.encode("utf-8"))

    class Settings:
        local_model_ready = False
        local_sentiment_model = ""
        local_zero_shot_model = ""
        llm_ready = False
        embedding_ready = True
        dashscope_model = ""
        dashscope_embedding_model = "fake-embedding"

    monkeypatch.setattr("app.services.analyze.get_settings", lambda: Settings())
    monkeypatch.setattr(
        "app.services.analyze.embed_texts",
        lambda texts: __import__("numpy").array(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
        ),
    )
    monkeypatch.setattr(
        "app.services.semantic.embed_texts",
        lambda texts: __import__("numpy").array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
    )

    request = RunAnalysisRequest(
        dataset_id=dataset.id,
        analysis_stage="classify",
        label_schema=LabelSchema(
            id="confirmed_topics",
            name="确认主题",
            description="用户确认后的主题分类",
            labels=["产品体验", "服务反馈"],
        ),
        use_llm=False,
        write_exports=False,
    )
    run, _ = run_analysis(dataset, documents, request)

    assert run.outputs is not None
    assert len(run.outputs.classification_results) == 3
    assert run.outputs.topics == []
    assert run.settings["analysis_stage"] == "classify"
    assert run.settings["classification_strategy"] == "label-embedding-classification"


def test_classification_stage_uses_label_profiles(monkeypatch):
    dataset, documents = ingest_dataset("sample.csv", SAMPLE.encode("utf-8"))

    class Settings:
        local_model_ready = False
        local_sentiment_model = ""
        local_zero_shot_model = ""
        llm_ready = False
        embedding_ready = True
        dashscope_model = ""
        dashscope_embedding_model = "fake-embedding"

    captured = {}

    monkeypatch.setattr("app.services.analyze.get_settings", lambda: Settings())
    monkeypatch.setattr("app.services.analyze.embed_texts", lambda texts: __import__("numpy").eye(len(texts), 4))

    def fake_profile_classifier(documents, labels, document_vectors, profiles=None):
        captured["labels"] = labels
        captured["profiles"] = profiles
        from app.models import ClassificationResult, Evidence

        return [
            ClassificationResult(
                document_id=document.id,
                label=labels[0],
                confidence=0.9,
                evidence=Evidence(
                    value=labels[0],
                    confidence=0.9,
                    snippet=document.content[:30],
                    document_id=document.id,
                    generator="model",
                ),
            )
            for document in documents
        ]

    monkeypatch.setattr("app.services.analyze.build_label_semantic_classifications", fake_profile_classifier)

    request = RunAnalysisRequest(
        dataset_id=dataset.id,
        analysis_stage="classify",
        label_schema=LabelSchema(
            id="confirmed_topics",
            name="确认主题",
            description="用户确认后的主题分类",
            labels=["服务响应"],
            profiles=[
                {
                    "name": "服务响应",
                    "description": "涉及客服、售后、响应速度",
                    "keywords": ["客服", "售后"],
                    "positive_examples": ["客服也很专业"],
                    "source_topic_ids": ["topic_1"],
                }
            ],
        ),
        use_llm=False,
        write_exports=False,
    )
    run, _ = run_analysis(dataset, documents, request)

    assert run.outputs is not None
    assert run.settings["classification_strategy"] == "profile-embedding-classification"
    assert captured["labels"] == ["服务响应"]
    assert captured["profiles"][0].keywords == ["客服", "售后"]


def test_analysis_pipeline_handles_illegal_excel_characters():
    sample = """标题,正文
评论1,"配送挺快，但是有个\b奇怪字符。"
评论2,整体还行。
"""
    dataset, documents = ingest_dataset("sample_illegal.csv", sample.encode("utf-8"))
    request = type(
        "Request",
        (),
        {"dataset_id": dataset.id, "top_k_terms": 8, "topic_count": 2, "label_schema": None, "use_llm": False},
    )()
    run, _ = run_analysis(dataset, documents, request)
    assert run.status == "completed"


def test_analysis_pipeline_can_skip_exports():
    dataset, documents = ingest_dataset("sample.csv", SAMPLE.encode("utf-8"))
    request = type(
        "Request",
        (),
        {
            "dataset_id": dataset.id,
            "analysis_stage": "explore",
            "top_k_terms": 12,
            "topic_count": 2,
            "label_schema": None,
            "use_llm": False,
            "write_exports": False,
            "export_xlsx": False,
        },
    )()
    run, _ = run_analysis(dataset, documents, request)
    assert run.status == "completed"
    assert run.outputs is not None
    assert run.outputs.exports == []


def test_ingest_dataset_can_preserve_duplicate_rows():
    sample = """标题,正文,标签
评论1,重复文本,positive
评论2,重复文本,negative
"""
    dataset, documents = ingest_dataset("sample.csv", sample.encode("utf-8"), deduplicate=False)
    assert dataset.document_count == 2
    assert len(documents) == 2


def test_workspace_snapshot_supports_custom_terms_and_synonyms():
    dataset, documents = ingest_dataset(
        "sample.csv",
        "标题,正文\n评论1,送餐员服务很好，南瓜羹很香。\n评论2,送餐员补送了一份南瓜羹。\n".encode("utf-8"),
    )
    workspace = DatasetWorkspace(
        dataset_id=dataset.id,
        custom_terms=["送餐员", "南瓜羹"],
        synonym_groups=[SynonymGroup(canonical_term="配送员", aliases=["送餐员"])],
        curated_terms=["配送员", "南瓜羹"],
    )

    snapshot = build_workspace_snapshot(workspace, documents)

    assert snapshot.selected_terms[0]["term"] == "配送员"
    assert snapshot.selected_terms[1]["term"] == "南瓜羹"
    assert "配送员" in snapshot.tokenized_documents[0]["tokens"]
    assert "南瓜羹" in snapshot.tokenized_documents[0]["tokens"]
    assert snapshot.frequency_matrix[0]["配送员"] >= 1
    assert snapshot.frequency_matrix[0]["南瓜羹"] >= 1


def test_analysis_pipeline_uses_saved_workspace(monkeypatch):
    dataset, documents = ingest_dataset(
        "sample.csv",
        "标题,正文\n评论1,送餐员服务很好，南瓜羹很香。\n评论2,送餐员补送了一份南瓜羹。\n".encode("utf-8"),
    )
    workspace = DatasetWorkspace(
        dataset_id=dataset.id,
        custom_terms=["送餐员", "南瓜羹"],
        synonym_groups=[SynonymGroup(canonical_term="配送员", aliases=["送餐员"])],
        curated_terms=["配送员", "南瓜羹"],
    )
    monkeypatch.setattr("app.services.analyze.load_workspace", lambda dataset_id: workspace)

    request = RunAnalysisRequest(
        dataset_id=dataset.id,
        analysis_stage="explore",
        top_k_terms=12,
        topic_count=2,
        use_llm=False,
        write_exports=False,
        export_xlsx=False,
    )
    run, _ = run_analysis(dataset, documents, request)

    assert run.outputs is not None
    assert [item["term"] for item in run.outputs.selected_terms[:2]] == ["配送员", "南瓜羹"]
    assert run.outputs.frequency_matrix[0]["配送员"] >= 1
    assert "配送员" in run.outputs.tokenized_documents[0]["tokens"]
