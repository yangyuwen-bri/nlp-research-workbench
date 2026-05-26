from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import app
from app.models import Dataset, Document
from app.storage import save_dataset


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


def _seed_dataset():
    dataset = Dataset(
        id="ds_workspace",
        name="workspace dataset",
        source_filename="workspace.csv",
        created_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
        document_count=2,
        text_column="正文",
    )
    documents = [
        Document(id="doc_1", dataset_id=dataset.id, source_row=1, content="送餐员服务很好，南瓜羹很香。"),
        Document(id="doc_2", dataset_id=dataset.id, source_row=2, content="送餐员补送了一份南瓜羹。"),
    ]
    return dataset, documents


def test_workspace_endpoints_support_curation_flow(monkeypatch, tmp_path):
    _configure_storage_dirs(monkeypatch, tmp_path)
    dataset, documents = _seed_dataset()
    save_dataset(dataset, documents)
    client = TestClient(app)

    initial = client.get(f"/api/datasets/{dataset.id}/workspace")
    assert initial.status_code == 200
    assert initial.json()["workspace"]["curated_terms"] == []

    custom = client.post(f"/api/datasets/{dataset.id}/workspace/custom-terms", json=["送餐员", "南瓜羹"])
    assert custom.status_code == 200
    assert "送餐员" in custom.json()["workspace"]["custom_terms"]

    synonym = client.post(
        f"/api/datasets/{dataset.id}/workspace/synonym-groups",
        json={"canonical_term": "配送员", "aliases": ["送餐员"]},
    )
    assert synonym.status_code == 200
    assert synonym.json()["workspace"]["synonym_groups"][0]["canonical_term"] == "配送员"

    curated = client.put(f"/api/datasets/{dataset.id}/workspace/curated-terms", json=["配送员", "南瓜羹"])
    assert curated.status_code == 200
    payload = curated.json()
    assert [item["term"] for item in payload["selected_terms"][:2]] == ["配送员", "南瓜羹"]
    assert payload["frequency_matrix"][0]["配送员"] >= 1
    assert payload["frequency_matrix"][0]["南瓜羹"] >= 1
    assert "配送员" in payload["tokenized_documents"][0]["tokens"]


def test_workspace_patch_can_exclude_terms(monkeypatch, tmp_path):
    _configure_storage_dirs(monkeypatch, tmp_path)
    dataset, documents = _seed_dataset()
    save_dataset(dataset, documents)
    client = TestClient(app)

    response = client.put(
        f"/api/datasets/{dataset.id}/workspace",
        json={"custom_terms": ["送餐员", "南瓜羹"], "excluded_terms": ["服务"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "服务" in payload["workspace"]["excluded_terms"]
    assert "服务" not in payload["tokenized_documents"][0]["tokens"]


def test_workspace_summary_and_sections_are_available(monkeypatch, tmp_path):
    _configure_storage_dirs(monkeypatch, tmp_path)
    dataset, documents = _seed_dataset()
    save_dataset(dataset, documents)
    client = TestClient(app)

    summary = client.get(f"/api/datasets/{dataset.id}/workspace/summary")
    assert summary.status_code == 200
    summary_payload = summary.json()
    assert summary_payload["summary"]["document_count"] == 2
    assert summary_payload["workspace"]["dataset_id"] == dataset.id

    section = client.get(f"/api/datasets/{dataset.id}/workspace/sections/top_terms?page=1&page_size=10")
    assert section.status_code == 200
    section_payload = section.json()
    assert section_payload["section"] == "top_terms"
    assert section_payload["total"] >= 1
    assert section_payload["items"][0]["term"]
