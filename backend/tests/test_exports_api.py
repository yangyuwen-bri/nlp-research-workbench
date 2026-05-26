from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import app
from app.models import AnalysisRun
from app.storage import save_analysis

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


def _build_run(run_id: str, dataset_id: str, created_at: datetime) -> AnalysisRun:
    return AnalysisRun.model_validate(
        {
            "id": run_id,
            "owner_key": "user_alpha",
            "dataset_id": dataset_id,
            "created_at": created_at,
            "status": "completed",
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
                "exports": [
                    {"artifact": "report", "format": "md", "path": f"/tmp/{run_id}-report.md", "rows": 1},
                    {"artifact": "term_frequency", "format": "csv", "path": f"/tmp/{run_id}-terms.csv", "rows": 12},
                ],
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
            "error": None,
        }
    )


def test_list_exports_can_filter_by_dataset(monkeypatch, tmp_path):
    _configure_storage_dirs(monkeypatch, tmp_path)
    save_analysis(_build_run("run_alpha", "ds_alpha", datetime(2026, 5, 12, 8, tzinfo=timezone.utc)))
    save_analysis(_build_run("run_beta", "ds_beta", datetime(2026, 5, 12, 9, tzinfo=timezone.utc)))

    client = TestClient(app)
    response = client.get("/api/exports", params={"dataset_id": "ds_alpha"}, headers=USER_ALPHA)

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 2
    assert all(item["dataset_id"] == "ds_alpha" for item in payload)
    assert {item["artifact"] for item in payload} == {"report", "term_frequency"}


def test_list_exports_orders_latest_first(monkeypatch, tmp_path):
    _configure_storage_dirs(monkeypatch, tmp_path)
    save_analysis(_build_run("run_older", "ds_history", datetime(2026, 5, 12, 8, tzinfo=timezone.utc)))
    save_analysis(_build_run("run_latest", "ds_history", datetime(2026, 5, 12, 10, tzinfo=timezone.utc)))

    client = TestClient(app)
    response = client.get("/api/exports", params={"dataset_id": "ds_history"}, headers=USER_ALPHA)

    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["run_id"] == "run_latest"


def test_exports_are_user_scoped(monkeypatch, tmp_path):
    _configure_storage_dirs(monkeypatch, tmp_path)
    alpha = _build_run("run_alpha", "ds_alpha", datetime(2026, 5, 12, 8, tzinfo=timezone.utc))
    beta = alpha.model_copy(update={"id": "run_beta", "owner_key": "user_beta", "dataset_id": "ds_beta"})
    save_analysis(alpha)
    save_analysis(beta)

    client = TestClient(app)
    response = client.get("/api/exports", headers=USER_ALPHA)
    blocked = client.get("/api/exports/run_beta/report/md", headers=USER_ALPHA)

    assert response.status_code == 200
    assert {item["run_id"] for item in response.json()} == {"run_alpha"}
    assert blocked.status_code == 404
