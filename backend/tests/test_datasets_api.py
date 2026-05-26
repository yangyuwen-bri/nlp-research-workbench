from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import app
from app.models import AnalysisRun, Dataset, Document
from app.storage import save_analysis, save_dataset


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
        name=f"{dataset_id} dataset",
        source_filename=f"{dataset_id}.csv",
        created_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
        document_count=3,
        text_column="正文",
    )
    documents = [Document(id=f"{dataset_id}_doc1", dataset_id=dataset_id, source_row=1, content="测试文本")]
    return dataset, documents


def _build_run(run_id: str, dataset_id: str, status: str, created_at: datetime, export_count: int = 0) -> AnalysisRun:
    exports = [{"artifact": "report", "format": "md", "path": f"/tmp/{run_id}.md", "rows": 1}] * export_count
    return AnalysisRun.model_validate(
        {
            "id": run_id,
            "dataset_id": dataset_id,
            "created_at": created_at,
            "status": status,
            "started_at": created_at,
            "finished_at": created_at if status in {"completed", "failed"} else None,
            "generator_stack": ["rule"],
            "settings": {},
            "outputs": {
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
                "exports": exports,
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
            },
            "error": "boom" if status == "failed" else None,
        }
    )


def test_list_datasets_returns_project_summary(monkeypatch, tmp_path):
    _configure_storage_dirs(monkeypatch, tmp_path)
    alpha, alpha_docs = _build_dataset("ds_alpha")
    beta, beta_docs = _build_dataset("ds_beta")
    save_dataset(alpha, alpha_docs)
    save_dataset(beta, beta_docs)
    save_analysis(_build_run("run_queued", "ds_alpha", "queued", datetime(2026, 5, 12, 8, tzinfo=timezone.utc)))
    save_analysis(_build_run("run_done", "ds_alpha", "completed", datetime(2026, 5, 12, 10, tzinfo=timezone.utc), export_count=2))
    save_analysis(_build_run("run_failed", "ds_alpha", "failed", datetime(2026, 5, 12, 11, tzinfo=timezone.utc)))

    client = TestClient(app)
    response = client.get("/api/datasets")

    assert response.status_code == 200
    payload = {item["id"]: item for item in response.json()}
    assert payload["ds_alpha"]["analysis_count"] == 3
    assert payload["ds_alpha"]["completed_analysis_count"] == 1
    assert payload["ds_alpha"]["failed_analysis_count"] == 1
    assert payload["ds_alpha"]["export_count"] == 2
    assert payload["ds_alpha"]["last_run_status"] == "failed"
    assert payload["ds_beta"]["analysis_count"] == 0
    assert payload["ds_beta"]["last_run_at"] is None


def test_get_dataset_returns_preview_documents_by_default(monkeypatch, tmp_path):
    _configure_storage_dirs(monkeypatch, tmp_path)
    dataset = Dataset(
        id="ds_preview",
        name="preview dataset",
        source_filename="preview.csv",
        created_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
        document_count=30,
        text_column="正文",
    )
    documents = [
        Document(id=f"doc_{idx}", dataset_id=dataset.id, source_row=idx, content=f"测试文本 {idx}")
        for idx in range(1, 31)
    ]
    save_dataset(dataset, documents)

    client = TestClient(app)
    response = client.get(f"/api/datasets/{dataset.id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset"]["id"] == dataset.id
    assert len(payload["documents"]) == 20
    assert payload["documents"][0]["source_row"] == 1
