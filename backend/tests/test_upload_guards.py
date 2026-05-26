from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.routers import datasets as datasets_router
from app.services.ingest import UploadValidationError, ingest_dataset
from app.settings import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _set_upload_limits(
    monkeypatch,
    *,
    max_bytes: int = 10 * 1024 * 1024,
    max_rows: int = 50000,
    max_columns: int = 128,
    max_text_length: int = 20000,
) -> None:
    monkeypatch.setenv("UPLOAD_MAX_FILE_BYTES", str(max_bytes))
    monkeypatch.setenv("UPLOAD_MAX_ROWS", str(max_rows))
    monkeypatch.setenv("UPLOAD_MAX_COLUMNS", str(max_columns))
    monkeypatch.setenv("UPLOAD_MAX_TEXT_LENGTH", str(max_text_length))
    get_settings.cache_clear()


def test_ingest_dataset_rejects_oversized_file(monkeypatch):
    _set_upload_limits(monkeypatch, max_bytes=16)

    with pytest.raises(UploadValidationError) as exc_info:
        ingest_dataset("sample.csv", "正文\n这是一段文本\n".encode("utf-8"))
    exc = exc_info.value
    assert exc.status_code == 413
    assert "上传文件过大" in str(exc)


def test_ingest_dataset_rejects_too_many_rows(monkeypatch):
    _set_upload_limits(monkeypatch, max_rows=2)
    payload = "正文\n第一行\n第二行\n第三行\n".encode("utf-8")

    with pytest.raises(UploadValidationError) as exc_info:
        ingest_dataset("sample.csv", payload)
    exc = exc_info.value
    assert exc.status_code == 400
    assert "当前最多支持 2 行" in str(exc)


def test_ingest_dataset_rejects_too_many_columns(monkeypatch):
    _set_upload_limits(monkeypatch, max_columns=2)
    payload = "标题,正文,标签\n评论1,内容,positive\n".encode("utf-8")

    with pytest.raises(UploadValidationError) as exc_info:
        ingest_dataset("sample.csv", payload)
    exc = exc_info.value
    assert exc.status_code == 400
    assert "当前最多支持 2 列" in str(exc)


def test_ingest_dataset_rejects_overlong_text(monkeypatch):
    _set_upload_limits(monkeypatch, max_text_length=4)
    payload = "正文\n这是一段超长文本\n".encode("utf-8")

    with pytest.raises(UploadValidationError) as exc_info:
        ingest_dataset("sample.csv", payload)
    assert "单条正文最多支持 4 个字符" in str(exc_info.value)


def test_upload_dataset_returns_validation_error(monkeypatch):
    _set_upload_limits(monkeypatch, max_rows=1)
    monkeypatch.setattr(datasets_router, "save_dataset", lambda dataset, documents: None)
    client = TestClient(app)

    response = client.post(
        "/api/datasets/upload",
        files={"file": ("sample.csv", "正文\n第一行\n第二行\n".encode("utf-8"), "text/csv")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "上传数据行数过多，当前最多支持 1 行。"


def test_upload_dataset_returns_preview_on_success(monkeypatch):
    _set_upload_limits(monkeypatch)
    monkeypatch.setattr(datasets_router, "save_dataset", lambda dataset, documents: None)
    client = TestClient(app)

    response = client.post(
        "/api/datasets/upload",
        files={"file": ("sample.csv", "标题,正文\n评论1,内容1\n评论2,内容2\n".encode("utf-8"), "text/csv")},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["dataset"]["document_count"] == 2
    assert body["preview_count"] == 2
    assert len(body["documents"]) == 2


def test_ingest_dataset_can_auto_detect_review_column():
    payload = "label,review\n1,很快，好吃，味道足，量大\n0,包装一般，但是送达及时\n".encode("utf-8")

    dataset, documents = ingest_dataset("waimai_10k.csv", payload)

    assert dataset.text_column == "review"
    assert dataset.document_count == 2
    assert documents[0].content == "很快，好吃，味道足，量大"


def test_ingest_dataset_can_auto_detect_text_like_column_without_standard_name():
    payload = "id,message_body,category\n1,这个平台的界面很清楚，操作也比较顺手,产品\n2,物流有点慢，不过客服回复挺及时,服务\n".encode("utf-8")

    dataset, documents = ingest_dataset("custom.csv", payload)

    assert dataset.text_column == "message_body"
    assert dataset.document_count == 2
    assert documents[1].content == "物流有点慢，不过客服回复挺及时"
