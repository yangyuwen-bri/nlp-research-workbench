from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import app
from app.models import Dataset, Document
from app.storage import save_dataset

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


def test_list_builtin_assets():
    client = TestClient(app)
    response = client.get("/api/sources/library")
    assert response.status_code == 200
    payload = response.json()
    assert payload
    assert any(item["id"] == "waimai_demo" for item in payload)
    assert any(item["id"] == "tnews_subset" for item in payload)


def test_import_builtin_asset_is_user_scoped(monkeypatch, tmp_path):
    _configure_storage_dirs(monkeypatch, tmp_path)
    client = TestClient(app)

    alpha_first = client.post("/api/sources/library/waimai_demo/import", headers=USER_ALPHA)
    alpha_second = client.post("/api/sources/library/waimai_demo/import", headers=USER_ALPHA)
    beta_first = client.post("/api/sources/library/waimai_demo/import", headers=USER_BETA)

    assert alpha_first.status_code == 200
    assert alpha_second.status_code == 200
    assert beta_first.status_code == 200
    assert alpha_first.json()["reused_existing"] is False
    assert alpha_second.json()["reused_existing"] is True
    assert beta_first.json()["reused_existing"] is False
    assert alpha_first.json()["dataset"]["id"] != beta_first.json()["dataset"]["id"]


def test_source_import_history_filters_non_upload_datasets(monkeypatch, tmp_path):
    _configure_storage_dirs(monkeypatch, tmp_path)
    upload_dataset = Dataset(
        id="ds_upload",
        owner_key="user_alpha",
        name="manual upload",
        source_filename="upload.csv",
        source_type="upload",
        created_at=datetime(2026, 6, 4, tzinfo=timezone.utc),
        document_count=1,
        text_column="content",
    )
    library_dataset = Dataset(
        id="ds_library",
        owner_key="user_alpha",
        name="builtin set",
        source_filename="public_waimai_demo.csv",
        source_type="library",
        source_label="Waimai",
        source_reference="waimai",
        created_at=datetime(2026, 6, 4, tzinfo=timezone.utc),
        document_count=2,
        text_column="content",
    )
    save_dataset(upload_dataset, [Document(id="doc_upload", dataset_id="ds_upload", source_row=1, content="x")])
    save_dataset(
        library_dataset,
        [
            Document(id="doc_library_1", dataset_id="ds_library", source_row=1, content="x"),
            Document(id="doc_library_2", dataset_id="ds_library", source_row=2, content="y"),
        ],
    )

    client = TestClient(app)
    response = client.get("/api/sources/imports", headers=USER_ALPHA)
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["dataset_id"] == "ds_library"


def test_openalex_preview_and_import(monkeypatch, tmp_path):
    _configure_storage_dirs(monkeypatch, tmp_path)
    from app.routers import sources

    async def _preview(query: str, *, limit: int):
        return {
            "connector": "openalex",
            "query": query,
            "total_returned": 1,
            "items": [
                {
                    "external_id": "https://openalex.org/W1",
                    "title": "数字人文研究",
                    "abstract": "这是摘要。",
                    "publication_year": 2026,
                    "authors": ["张三"],
                    "source_name": "Test Journal",
                    "source_url": "https://openalex.org/W1",
                }
            ],
        }

    async def _import(query: str, *, limit: int, owner_key: str):
        dataset = Dataset(
            id="ds_openalex",
            owner_key=owner_key,
            name=f"OpenAlex · {query}",
            source_filename="openalex.csv",
            source_type="connector",
            source_label="OpenAlex",
            source_reference=query,
            created_at=datetime(2026, 6, 4, tzinfo=timezone.utc),
            document_count=1,
            text_column="content",
            fingerprint=f"fp-{owner_key}-{query}",
        )
        documents = [
            Document(
                id="doc_openalex",
                dataset_id="ds_openalex",
                source_row=1,
                title="数字人文研究",
                content="这是摘要。",
                metadata={"source_connector": "openalex"},
            )
        ]
        return dataset, documents

    monkeypatch.setattr(sources, "preview_openalex", _preview)
    monkeypatch.setattr(sources, "import_openalex_query", _import)

    client = TestClient(app)
    preview = client.post("/api/sources/connectors/openalex/preview", headers=USER_ALPHA, json={"query": "数字人文", "limit": 5})
    imported = client.post("/api/sources/connectors/openalex/import", headers=USER_ALPHA, json={"query": "数字人文", "limit": 20})

    assert preview.status_code == 200
    assert preview.json()["items"][0]["title"] == "数字人文研究"
    assert imported.status_code == 200
    assert imported.json()["dataset"]["source_type"] == "connector"


def test_huggingface_search_preview_and_import(monkeypatch, tmp_path):
    _configure_storage_dirs(monkeypatch, tmp_path)
    from app.routers import sources

    async def _search(query: str, *, limit: int):
        return {
            "connector": "huggingface",
            "query": query,
            "total_returned": 1,
            "items": [
                {
                    "dataset_id": "dirtycomputer/weibo_senti_100k",
                    "author": "dirtycomputer",
                    "description": "weibo dataset",
                    "likes": 12,
                    "downloads": 3400,
                    "last_modified": "2026-01-01T00:00:00Z",
                }
            ],
        }

    async def _preview(dataset_id: str, *, limit: int):
        return {
            "connector": "huggingface",
            "dataset_id": dataset_id,
            "config": "default",
            "split": "train",
            "total_returned": 1,
            "items": [
                {
                    "row_idx": 0,
                    "text": "微博文本",
                    "title": None,
                    "label": "1",
                    "raw": {"review": "微博文本", "label": 1},
                }
            ],
        }

    async def _import(dataset_id: str, *, limit: int, owner_key: str):
        dataset = Dataset(
            id="ds_hf",
            owner_key=owner_key,
            name=f"Hugging Face · {dataset_id}",
            source_filename="hf.csv",
            source_type="connector",
            source_label="Hugging Face",
            source_reference=dataset_id,
            created_at=datetime(2026, 6, 4, tzinfo=timezone.utc),
            document_count=1,
            text_column="content",
            fingerprint=f"fp-hf-{owner_key}-{dataset_id}",
        )
        documents = [Document(id="doc_hf", dataset_id="ds_hf", source_row=1, content="微博文本")]
        return dataset, documents

    monkeypatch.setattr(sources, "search_huggingface_datasets", _search)
    monkeypatch.setattr(sources, "preview_huggingface_dataset", _preview)
    monkeypatch.setattr(sources, "import_huggingface_dataset", _import)

    client = TestClient(app)
    search = client.post("/api/sources/connectors/huggingface/search", headers=USER_ALPHA, json={"query": "weibo", "limit": 5})
    preview = client.post(
        "/api/sources/connectors/huggingface/preview",
        headers=USER_ALPHA,
        json={"dataset_id": "dirtycomputer/weibo_senti_100k", "limit": 5},
    )
    imported = client.post(
        "/api/sources/connectors/huggingface/import",
        headers=USER_ALPHA,
        json={"dataset_id": "dirtycomputer/weibo_senti_100k", "limit": 20},
    )

    assert search.status_code == 200
    assert search.json()["items"][0]["dataset_id"] == "dirtycomputer/weibo_senti_100k"
    assert preview.status_code == 200
    assert preview.json()["items"][0]["text"] == "微博文本"
    assert imported.status_code == 200
    assert imported.json()["dataset"]["source_label"] == "Hugging Face"
