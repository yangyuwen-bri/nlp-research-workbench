from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import app
from app.models import AnalysisOutputs, AnalysisRun, Dataset, Document
from app.storage import load_analysis, save_analysis, save_dataset

USER_ALPHA = {"X-User-Key": "user_alpha"}
USER_BETA = {"X-User-Key": "user_beta"}


def _configure_storage_dirs(monkeypatch, tmp_path):
    from app import storage

    data_dir = tmp_path / "data"
    datasets_dir = data_dir / "datasets"
    analyses_dir = data_dir / "analyses"
    exports_dir = data_dir / "exports"
    for directory in (data_dir, datasets_dir, analyses_dir, exports_dir):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(storage, "DATA_DIR", data_dir)
    monkeypatch.setattr(storage, "DATASETS_DIR", datasets_dir)
    monkeypatch.setattr(storage, "ANALYSES_DIR", analyses_dir)
    monkeypatch.setattr(storage, "EXPORTS_DIR", exports_dir)
    monkeypatch.setattr(
        storage,
        "get_storage_backend",
        lambda: storage.JsonStorageBackend(datasets_dir=datasets_dir, analyses_dir=analyses_dir),
    )


def _build_dataset(dataset_id: str) -> tuple[Dataset, list[Document]]:
    dataset = Dataset(
        id=dataset_id,
        owner_key="user_alpha",
        name=f"{dataset_id} dataset",
        source_filename=f"{dataset_id}.csv",
        created_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
        document_count=1,
        text_column="正文",
    )
    documents = [Document(id=f"{dataset_id}_doc1", dataset_id=dataset_id, source_row=1, content="测试文本")]
    return dataset, documents


def _build_run(
    run_id: str,
    dataset_id: str,
    created_at: datetime,
    *,
    has_outputs: bool = True,
    status: str = "completed",
) -> AnalysisRun:
    outputs = None
    if has_outputs:
        outputs = {
            "top_terms": [],
            "tokenized_documents": [],
            "selected_terms": [],
            "match_rows": [],
            "binary_matrix": [],
            "cooccurrence_edges": [],
            "sentiment_results": [],
            "classification_results": [],
            "topics": [],
            "insight_cards": [],
            "report_markdown": "",
            "exports": [],
            "semantic_execution": {
                "requested": False,
                "attempted": False,
                "used": False,
                "status": "not_requested",
                "provider": "dashscope",
                "model": None,
                "strategy": None,
                "message": "disabled",
                "error_type": None,
            },
            "llm_execution": {
                "requested": False,
                "attempted": False,
                "used": False,
                "status": "not_requested",
                "provider": "dashscope",
                "model": None,
                "message": "disabled",
                "error_type": None,
            },
        }
    return AnalysisRun.model_validate(
        {
            "id": run_id,
            "owner_key": "user_alpha",
            "dataset_id": dataset_id,
            "created_at": created_at,
            "status": status,
            "started_at": created_at if status != "queued" else None,
            "finished_at": created_at if status in {"completed", "failed"} else None,
            "generator_stack": ["rule", "model"],
            "settings": {"topic_count": 4},
            "outputs": outputs,
            "error": None,
        }
    )


def test_list_analyses_can_filter_by_dataset_id(monkeypatch, tmp_path):
    _configure_storage_dirs(monkeypatch, tmp_path)
    save_analysis(_build_run("run_older", "ds_alpha", datetime(2026, 5, 10, 9, tzinfo=timezone.utc), has_outputs=False))
    save_analysis(_build_run("run_latest", "ds_alpha", datetime(2026, 5, 12, 9, tzinfo=timezone.utc)))
    save_analysis(_build_run("run_other", "ds_beta", datetime(2026, 5, 11, 9, tzinfo=timezone.utc)))

    client = TestClient(app)
    response = client.get("/api/analyses", params={"dataset_id": "ds_alpha"}, headers=USER_ALPHA)

    assert response.status_code == 200
    payload = response.json()
    assert [item["id"] for item in payload] == ["run_latest", "run_older"]
    assert all(item["dataset_id"] == "ds_alpha" for item in payload)
    assert payload[0]["has_outputs"] is True
    assert payload[1]["has_outputs"] is False
    assert "outputs" not in payload[0]


def test_list_analyses_handles_mixed_naive_and_aware_datetimes(monkeypatch, tmp_path):
    _configure_storage_dirs(monkeypatch, tmp_path)
    save_analysis(_build_run("run_naive", "ds_alpha", datetime(2026, 5, 12, 8), has_outputs=False))
    save_analysis(_build_run("run_aware", "ds_alpha", datetime(2026, 5, 12, 9, tzinfo=timezone.utc)))

    client = TestClient(app)
    response = client.get("/api/analyses", params={"dataset_id": "ds_alpha"}, headers=USER_ALPHA)

    assert response.status_code == 200
    payload = response.json()
    assert [item["id"] for item in payload] == ["run_aware", "run_naive"]


def test_create_analysis_persists_history_entry(monkeypatch, tmp_path):
    _configure_storage_dirs(monkeypatch, tmp_path)
    dataset, documents = _build_dataset("ds_history")
    save_dataset(dataset, documents)

    created_at = datetime(2026, 5, 12, 12, tzinfo=timezone.utc)
    expected_run = _build_run("run_saved", dataset.id, created_at)

    def fake_enqueue(run, request):
        assert request.dataset_id == dataset.id
        save_analysis(expected_run.model_copy(update={"id": run.id, "created_at": run.created_at}))

    monkeypatch.setattr("app.routers.analyses._enqueue_analysis_run", fake_enqueue)

    client = TestClient(app)
    create_response = client.post(
        "/api/analyses/run",
        json={"dataset_id": dataset.id, "top_k_terms": 12, "topic_count": 4, "use_llm": False},
        headers=USER_ALPHA,
    )
    assert create_response.status_code == 200
    create_payload = create_response.json()
    assert create_payload["run"]["status"] == "queued"

    history_response = client.get("/api/analyses", params={"dataset_id": dataset.id}, headers=USER_ALPHA)
    assert history_response.status_code == 200
    payload = history_response.json()
    assert len(payload) == 1
    assert payload[0]["id"] == create_payload["run"]["id"]
    assert payload[0]["export_count"] == 0


def test_create_analysis_saves_queued_run_before_background_work(monkeypatch, tmp_path):
    _configure_storage_dirs(monkeypatch, tmp_path)
    dataset, documents = _build_dataset("ds_queue")
    save_dataset(dataset, documents)
    observed = {}

    def fake_enqueue(run, request):
        observed["saved"] = load_analysis(run.id)
        observed["dataset_id"] = request.dataset_id

    monkeypatch.setattr("app.routers.analyses._enqueue_analysis_run", fake_enqueue)

    client = TestClient(app)
    response = client.post(
        "/api/analyses/run",
        json={"dataset_id": dataset.id, "top_k_terms": 12, "topic_count": 4, "use_llm": False},
        headers=USER_ALPHA,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run"]["status"] == "queued"
    assert observed["dataset_id"] == dataset.id
    assert observed["saved"].status == "queued"
    assert observed["saved"].started_at is None
    assert observed["saved"].finished_at is None


def test_create_analysis_rejects_duplicate_active_run(monkeypatch, tmp_path):
    _configure_storage_dirs(monkeypatch, tmp_path)
    dataset, documents = _build_dataset("ds_active")
    save_dataset(dataset, documents)
    active = _build_run("run_active", dataset.id, datetime(2026, 5, 12, 12, tzinfo=timezone.utc), status="running")
    save_analysis(active)

    client = TestClient(app)
    response = client.post(
        "/api/analyses/run",
        json={"dataset_id": dataset.id, "top_k_terms": 12, "topic_count": 4, "use_llm": False},
        headers=USER_ALPHA,
    )

    assert response.status_code == 409
    assert "已有任务正在运行" in response.json()["detail"]


def test_get_analysis_returns_saved_run(monkeypatch, tmp_path):
    _configure_storage_dirs(monkeypatch, tmp_path)
    run = _build_run("run_detail", "ds_detail", datetime(2026, 5, 12, 18, tzinfo=timezone.utc))
    save_analysis(run)

    client = TestClient(app)
    response = client.get(f"/api/analyses/{run.id}", headers=USER_ALPHA)

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == run.id
    assert payload["dataset_id"] == run.dataset_id
    assert payload["outputs"]["report_markdown"] == ""


def test_get_analysis_summary_returns_lightweight_payload(monkeypatch, tmp_path):
    _configure_storage_dirs(monkeypatch, tmp_path)
    run = _build_run("run_detail", "ds_detail", datetime(2026, 5, 12, 18, tzinfo=timezone.utc))
    outputs = AnalysisOutputs.model_validate(
        {
            **run.outputs.model_dump(),
            "top_terms": [{"term": "味道", "term_frequency": 10, "document_frequency": 8}],
            "sentiment_results": [
                {
                    "document_id": "doc_1",
                    "label": "positive",
                    "score": 0.9,
                    "aspect_hits": {},
                    "evidence": {
                        "value": "positive",
                        "confidence": 0.9,
                        "snippet": "很好吃",
                        "document_id": "doc_1",
                        "generator": "rule",
                    },
                }
            ],
            "classification_results": [
                {
                    "document_id": "doc_1",
                    "label": "产品体验",
                    "confidence": 0.8,
                    "evidence": {
                        "value": "产品体验",
                        "confidence": 0.8,
                        "snippet": "很好吃",
                        "document_id": "doc_1",
                        "generator": "rule",
                    },
                }
            ],
        }
    )
    run = run.model_copy(update={"outputs": outputs})
    save_analysis(run)

    client = TestClient(app)
    response = client.get(f"/api/analyses/{run.id}/summary", headers=USER_ALPHA)

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == run.id
    assert payload["overview"]["sample_count"] == 1
    assert payload["overview"]["dominant_classification"] == "产品体验"
    assert payload["previews"]["top_terms"][0]["term"] == "味道"
    assert "outputs" not in payload


def test_name_analysis_topics_updates_run_with_llm_suggestions(monkeypatch, tmp_path):
    _configure_storage_dirs(monkeypatch, tmp_path)
    dataset, documents = _build_dataset("ds_topics")
    save_dataset(dataset, documents)
    run = _build_run("run_topics", dataset.id, datetime(2026, 5, 12, 18, tzinfo=timezone.utc))
    outputs = AnalysisOutputs.model_validate(
        {
            **run.outputs.model_dump(),
            "topics": [
                {
                    "topic_id": "topic_1",
                    "name": "主题 1",
                    "size": 12,
                    "keywords": ["酒店", "房间", "服务"],
                    "summary": "酒店、房间、服务",
                    "evidences": [
                        {
                            "value": "主题证据",
                            "confidence": 0.7,
                            "snippet": "酒店房间干净，服务不错。",
                            "document_id": documents[0].id,
                            "generator": "model",
                        }
                    ],
                }
            ],
        }
    )
    save_analysis(run.model_copy(update={"outputs": outputs}))

    monkeypatch.setattr(
        "app.routers.analyses.name_topic_clusters",
        lambda dataset_name, topics: [
            {"topic_id": "topic_1", "name": "酒店服务", "summary": "围绕酒店住宿体验。", "confidence": 0.91}
        ],
    )

    client = TestClient(app)
    response = client.post(f"/api/analyses/{run.id}/topics/name", headers=USER_ALPHA)

    assert response.status_code == 200
    payload = response.json()
    assert payload["suggestions"][0]["name"] == "酒店服务"
    saved = load_analysis(run.id)
    assert saved.outputs.topics[0].name == "酒店服务"
    assert saved.outputs.topics[0].suggested_name == "酒店服务"
    assert saved.outputs.topics[0].name_source == "llm"


def test_get_analysis_section_returns_paginated_rows(monkeypatch, tmp_path):
    _configure_storage_dirs(monkeypatch, tmp_path)
    run = _build_run("run_detail", "ds_detail", datetime(2026, 5, 12, 18, tzinfo=timezone.utc))
    outputs = AnalysisOutputs.model_validate(
        {
            **run.outputs.model_dump(),
            "tokenized_documents": [
                {"row_id": 1, "document_id": "doc_1", "content": "第一条", "tokens": "第一 条"},
                {"row_id": 2, "document_id": "doc_2", "content": "第二条", "tokens": "第二 条"},
            ],
        }
    )
    run = run.model_copy(update={"outputs": outputs})
    save_analysis(run)

    client = TestClient(app)
    response = client.get(f"/api/analyses/{run.id}/sections/tokenized", params={"page": 2, "page_size": 1}, headers=USER_ALPHA)

    assert response.status_code == 200
    payload = response.json()
    assert payload["section"] == "tokenized"
    assert payload["total"] == 2
    assert payload["page"] == 2
    assert payload["items"][0]["document_id"] == "doc_2"


def test_retry_analysis_clones_settings_and_creates_new_queued_run(monkeypatch, tmp_path):
    _configure_storage_dirs(monkeypatch, tmp_path)
    dataset, documents = _build_dataset("ds_retry")
    save_dataset(dataset, documents)
    original = _build_run("run_failed", dataset.id, datetime(2026, 5, 12, 10, tzinfo=timezone.utc), status="failed")
    original = original.model_copy(
        update={
            "settings": {
                "top_k_terms": 12,
                "topic_count": 6,
                "use_llm": False,
                "write_exports": True,
                "export_xlsx": False,
            },
            "error": "boom",
        }
    )
    save_analysis(original)
    observed = {}

    def fake_enqueue(run, request):
        observed["run"] = run
        observed["request"] = request

    monkeypatch.setattr("app.routers.analyses._enqueue_analysis_run", fake_enqueue)

    client = TestClient(app)
    response = client.post(f"/api/analyses/{original.id}/retry", headers=USER_ALPHA)

    assert response.status_code == 200
    payload = response.json()["run"]
    assert payload["id"] != original.id
    assert payload["dataset_id"] == dataset.id
    assert payload["status"] == "queued"
    assert payload["started_at"] is None
    assert payload["finished_at"] is None
    assert observed["request"].top_k_terms == 12
    assert observed["request"].topic_count == 6
    assert observed["request"].use_llm is False
    saved = load_analysis(payload["id"], owner_key="user_alpha")
    assert saved.status == "queued"


def test_analysis_endpoints_are_user_scoped(monkeypatch, tmp_path):
    _configure_storage_dirs(monkeypatch, tmp_path)
    alpha_run = _build_run("run_alpha", "ds_alpha", datetime(2026, 5, 12, 9, tzinfo=timezone.utc))
    beta_run = alpha_run.model_copy(update={"id": "run_beta", "owner_key": "user_beta", "dataset_id": "ds_beta"})
    save_analysis(alpha_run)
    save_analysis(beta_run)

    client = TestClient(app)
    alpha_list = client.get("/api/analyses", params={"dataset_id": "ds_alpha"}, headers=USER_ALPHA)
    beta_detail = client.get("/api/analyses/run_beta", headers=USER_ALPHA)

    assert alpha_list.status_code == 200
    assert [item["id"] for item in alpha_list.json()] == ["run_alpha"]
    assert beta_detail.status_code == 404
