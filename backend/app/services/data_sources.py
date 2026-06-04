from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pandas as pd

from ..models import (
    DataSourceAsset,
    Dataset,
    Document,
    HuggingFaceDatasetCandidate,
    HuggingFacePreviewResponse,
    HuggingFacePreviewRow,
    HuggingFaceSearchResponse,
    OpenAlexPreviewResponse,
    OpenAlexPreviewRow,
)
from ..services.ingest import build_dataset_fingerprint, ingest_dataset


FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "fixtures"
PROJECT_ROOT = FIXTURES_DIR.parent
BENCH_DATA_DIR = PROJECT_ROOT / "bench_data"
DATA_SOURCE_ASSETS_PATH = FIXTURES_DIR / "data_source_assets.json"


def _asset_id_from_filename(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0]
    if stem.startswith("public_"):
        stem = stem[len("public_") :]
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    return stem


def _asset_name_from_filename(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0]
    label = stem.replace("public_", "").replace("_demo", "").replace("_", " ").strip()
    pieces = [part.capitalize() if part.isascii() else part for part in label.split()]
    return " ".join(pieces) or stem


def _asset_category(task: str, filename: str) -> str:
    task_value = (task or "").strip().lower()
    if "sentiment" in task_value:
        return "评论分析"
    if "topic" in task_value or "opinion" in task_value:
        return "舆情与主题"
    if "news" in task_value:
        return "新闻与标题"
    if "paper" in task_value or "abstract" in task_value:
        return "论文与摘要"
    if "class" in task_value:
        return "分类基准"
    return "官方样例"


def _asset_tags(task: str, filename: str) -> list[str]:
    tags = []
    lower_name = filename.lower()
    if "waimai" in lower_name:
        tags.extend(["评论", "外卖", "情感"])
    if "shopping" in lower_name:
        tags.extend(["评论", "电商", "情感"])
    if "weibo" in lower_name:
        tags.extend(["社交媒体", "微博", "情感"])
    if "htl" in lower_name:
        tags.extend(["酒店", "评论", "情感"])
    if "rumor" in lower_name:
        tags.extend(["舆情", "谣言", "主题"])
    if "sentiment" in (task or ""):
        tags.append("情感分析")
    if "topic" in (task or "") or "opinion" in (task or ""):
        tags.append("主题发现")
    return list(dict.fromkeys(tags))


@lru_cache
def _load_builtin_asset_registry() -> list[dict[str, Any]]:
    payload = json.loads(DATA_SOURCE_ASSETS_PATH.read_text(encoding="utf-8"))
    return payload.get("assets", [])


@lru_cache
def load_builtin_assets() -> list[DataSourceAsset]:
    payload = _load_builtin_asset_registry()
    items: list[DataSourceAsset] = []
    for item in payload:
        items.append(
            DataSourceAsset(
                id=str(item["id"]),
                name=str(item["name"]),
                category=str(item["category"]),
                kind="builtin",
                description=str(item.get("description", "")).strip(),
                source_name=str(item.get("source_name", "公开样例")),
                source_url=str(item.get("source_url", "")),
                license=str(item.get("license", "public-demo")),
                language=str(item.get("language", "zh-CN")),
                document_count=int(item.get("document_count", 0)),
                recommended_modules=list(item.get("recommended_modules", [])),
                fields=list(item.get("fields", [])),
                tags=list(item.get("tags", [])),
                notes=str(item.get("notes", "")).strip(),
            )
        )
    return items


def get_builtin_asset(asset_id: str) -> DataSourceAsset:
    for item in load_builtin_assets():
        if item.id == asset_id:
            return item
    raise KeyError(asset_id)


def _resolve_asset_path(raw_item: dict[str, Any]) -> Path:
    storage = str(raw_item.get("storage", "fixtures")).strip().lower()
    filename = str(raw_item["file"])
    if storage == "bench_data":
        return BENCH_DATA_DIR / filename
    return FIXTURES_DIR / filename


def _prepare_asset_bytes(raw_item: dict[str, Any]) -> tuple[str, bytes]:
    path = _resolve_asset_path(raw_item)
    row_limit = int(raw_item.get("row_limit", 0) or 0)
    if row_limit <= 0:
        return path.name, path.read_bytes()
    frame = pd.read_csv(path).head(row_limit)
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return path.with_suffix(".csv").name, buffer.getvalue().encode("utf-8")


def import_builtin_asset(asset_id: str, *, owner_key: str) -> tuple[Dataset, list[Document]]:
    asset = get_builtin_asset(asset_id)
    registry_item = next(item for item in _load_builtin_asset_registry() if str(item["id"]) == asset_id)
    source_filename, payload = _prepare_asset_bytes(registry_item)
    dataset, documents = ingest_dataset(
        source_filename,
        payload,
        owner_key=owner_key,
        deduplicate=False,
    )
    fingerprint = build_dataset_fingerprint(asset.name, dataset.text_column, documents)
    dataset = dataset.model_copy(
        update={
            "name": asset.name,
            "source_filename": source_filename,
            "source_type": "library",
            "source_label": asset.name,
            "source_reference": asset.id,
            "source_meta": {
                "asset_id": asset.id,
                "category": asset.category,
                "source_name": asset.source_name,
                "source_url": asset.source_url,
                "license": asset.license,
                "tags": asset.tags,
                "storage": registry_item.get("storage", "fixtures"),
                "source_file": registry_item.get("file"),
                "row_limit": registry_item.get("row_limit"),
            },
            "fingerprint": fingerprint,
        }
    )
    documents = [
        document.model_copy(
            update={
                "metadata": {
                    **document.metadata,
                    "source_asset_id": asset.id,
                    "source_asset_name": asset.name,
                }
            }
        )
        for document in documents
    ]
    return dataset, documents


async def search_huggingface_datasets(query: str, *, limit: int) -> HuggingFaceSearchResponse:
    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": "nlp-research-workbench/0.1"}) as client:
        response = await client.get(
            "https://huggingface.co/api/datasets",
            params={"search": query, "limit": limit},
        )
        response.raise_for_status()
    payload = response.json()
    items: list[HuggingFaceDatasetCandidate] = []
    for item in payload:
        items.append(
            HuggingFaceDatasetCandidate(
                dataset_id=str(item.get("id") or ""),
                author=str(item.get("author") or "") or None,
                description=str(item.get("description") or "")[:400],
                likes=int(item.get("likes") or 0),
                downloads=int(item.get("downloads") or 0),
                last_modified=str(item.get("lastModified") or "") or None,
            )
        )
    return HuggingFaceSearchResponse(query=query, total_returned=len(items), items=items)


async def _huggingface_primary_split(dataset_id: str) -> tuple[str, str]:
    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": "nlp-research-workbench/0.1"}) as client:
        response = await client.get(
            "https://datasets-server.huggingface.co/splits",
            params={"dataset": dataset_id},
        )
        response.raise_for_status()
    payload = response.json()
    splits = payload.get("splits") or []
    if not splits:
        raise ValueError("该数据集当前没有可用的 viewer split。")
    primary = splits[0]
    return str(primary["config"]), str(primary["split"])


def _pick_text_column_from_row(row: dict[str, Any]) -> tuple[str | None, str]:
    priority = ["content", "text", "review", "comment", "正文", "文本", "abstract", "body", "sentence"]
    for key in priority:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return None, value.strip()
    title = None
    for key in ["title", "标题", "subject", "question"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            title = value.strip()
            break
    if title:
        return title, title
    for key, value in row.items():
        if isinstance(value, str) and value.strip():
            return title, value.strip()
    return title, ""


async def preview_huggingface_dataset(dataset_id: str, *, limit: int) -> HuggingFacePreviewResponse:
    config, split = await _huggingface_primary_split(dataset_id)
    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": "nlp-research-workbench/0.1"}) as client:
        response = await client.get(
            "https://datasets-server.huggingface.co/first-rows",
            params={"dataset": dataset_id, "config": config, "split": split},
        )
        response.raise_for_status()
    payload = response.json()
    items: list[HuggingFacePreviewRow] = []
    for row_item in (payload.get("rows") or [])[:limit]:
        row = row_item.get("row") or {}
        title, text = _pick_text_column_from_row(row)
        if not text:
            continue
        label_value = None
        for key in ["label", "标签", "category", "类别"]:
            if key in row:
                value = row.get(key)
                label_value = str(value) if value is not None else None
                break
        items.append(
            HuggingFacePreviewRow(
                row_idx=int(row_item.get("row_idx") or 0),
                text=text,
                title=title,
                label=label_value,
                raw={key: value for key, value in row.items() if isinstance(value, (str, int, float, bool))},
            )
        )
    return HuggingFacePreviewResponse(
        dataset_id=dataset_id,
        config=config,
        split=split,
        total_returned=len(items),
        items=items,
    )


async def import_huggingface_dataset(dataset_id: str, *, limit: int, owner_key: str) -> tuple[Dataset, list[Document]]:
    config, split = await _huggingface_primary_split(dataset_id)
    async with httpx.AsyncClient(timeout=45.0, headers={"User-Agent": "nlp-research-workbench/0.1"}) as client:
        response = await client.get(
            "https://datasets-server.huggingface.co/rows",
            params={"dataset": dataset_id, "config": config, "split": split, "offset": 0, "length": limit},
        )
        response.raise_for_status()
    payload = response.json()
    rows = []
    for row_item in payload.get("rows") or []:
        row = row_item.get("row") or {}
        title, text = _pick_text_column_from_row(row)
        if not text:
            continue
        metadata = {key: value for key, value in row.items() if key not in {"content", "text", "review", "comment", "正文", "文本", "abstract", "body", "sentence", "title", "标题", "subject", "question"}}
        rows.append(
            {
                "title": title or "",
                "content": text,
                "label": metadata.get("label") or metadata.get("标签") or metadata.get("category") or metadata.get("类别") or "",
                "raw_metadata": json.dumps(metadata, ensure_ascii=False),
            }
        )
    if not rows:
        raise ValueError("Hugging Face 数据集没有可导入的文本列。")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["title", "content", "label", "raw_metadata"])
    writer.writeheader()
    writer.writerows(rows)
    filename = f"hf_{dataset_id.replace('/', '__')}.csv"
    dataset, documents = ingest_dataset(filename, buffer.getvalue().encode("utf-8"), owner_key=owner_key, deduplicate=True)
    fingerprint = build_dataset_fingerprint(f"Hugging Face · {dataset_id}", dataset.text_column, documents)
    dataset = dataset.model_copy(
        update={
            "name": f"Hugging Face · {dataset_id}",
            "source_filename": filename,
            "source_type": "connector",
            "source_label": "Hugging Face",
            "source_reference": dataset_id,
            "source_meta": {
                "connector_id": "huggingface",
                "dataset_id": dataset_id,
                "config": config,
                "split": split,
                "limit": len(documents),
                "imported_at": datetime.now(timezone.utc).isoformat(),
            },
            "fingerprint": fingerprint,
        }
    )
    documents = [
        document.model_copy(
            update={
                "metadata": {
                    **document.metadata,
                    "source_connector": "huggingface",
                    "source_dataset_id": dataset_id,
                    "source_config": config,
                    "source_split": split,
                }
            }
        )
        for document in documents
    ]
    return dataset, documents


def _reconstruct_openalex_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    if not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for token, indexes in inverted_index.items():
        for index in indexes:
            positions[int(index)] = token
    return " ".join(token for _, token in sorted(positions.items(), key=lambda item: item[0])).strip()


def _map_openalex_item(item: dict[str, Any]) -> OpenAlexPreviewRow:
    authors = []
    for authorship in item.get("authorships", [])[:5]:
        author = authorship.get("author") or {}
        display_name = str(author.get("display_name") or "").strip()
        if display_name:
            authors.append(display_name)
    primary_location = item.get("primary_location") or {}
    source = primary_location.get("source") or {}
    source_name = source.get("display_name")
    source_url = item.get("id") or item.get("doi") or item.get("landing_page_url")
    return OpenAlexPreviewRow(
        external_id=str(item.get("id") or ""),
        title=str(item.get("title") or "").strip(),
        abstract=_reconstruct_openalex_abstract(item.get("abstract_inverted_index")),
        publication_year=item.get("publication_year"),
        authors=authors,
        source_name=str(source_name) if source_name else None,
        source_url=str(source_url) if source_url else None,
    )


async def preview_openalex(query: str, *, limit: int) -> OpenAlexPreviewResponse:
    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": "nlp-research-workbench/0.1"}) as client:
        response = await client.get(
            "https://api.openalex.org/works",
            params={
                "search": query,
                "per-page": limit,
                "filter": "has_abstract:true",
            },
        )
        response.raise_for_status()
    payload = response.json()
    items = [_map_openalex_item(item) for item in payload.get("results", [])]
    items = [item for item in items if item.title or item.abstract]
    return OpenAlexPreviewResponse(query=query, total_returned=len(items), items=items)


async def import_openalex_query(query: str, *, limit: int, owner_key: str) -> tuple[Dataset, list[Document]]:
    remaining = limit
    page = 1
    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=45.0, headers={"User-Agent": "nlp-research-workbench/0.1"}) as client:
        while remaining > 0:
            per_page = min(remaining, 100)
            response = await client.get(
                "https://api.openalex.org/works",
                params={
                    "search": query,
                    "page": page,
                    "per-page": per_page,
                    "filter": "has_abstract:true",
                },
            )
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results", [])
            if not results:
                break
            for item in results:
                mapped = _map_openalex_item(item)
                content = mapped.abstract or mapped.title
                if not content:
                    continue
                rows.append(
                    {
                        "title": mapped.title,
                        "content": content,
                        "publication_year": mapped.publication_year,
                        "authors": " / ".join(mapped.authors),
                        "source_name": mapped.source_name or "",
                        "source_url": mapped.source_url or "",
                        "external_id": mapped.external_id,
                    }
                )
                remaining -= 1
                if remaining <= 0:
                    break
            if len(results) < per_page:
                break
            page += 1
    if not rows:
        raise ValueError("OpenAlex 没有返回可导入的标题或摘要。")

    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["title", "content", "publication_year", "authors", "source_name", "source_url", "external_id"],
    )
    writer.writeheader()
    writer.writerows(rows)
    filename = f"openalex_{query.strip().replace(' ', '_')[:40] or 'query'}.csv"
    dataset, documents = ingest_dataset(filename, buffer.getvalue().encode("utf-8"), owner_key=owner_key, deduplicate=True)
    fingerprint = build_dataset_fingerprint(f"OpenAlex · {query.strip()}", dataset.text_column, documents)
    dataset = dataset.model_copy(
        update={
            "name": f"OpenAlex · {query.strip()}",
            "source_filename": filename,
            "source_type": "connector",
            "source_label": "OpenAlex",
            "source_reference": query.strip(),
            "source_meta": {
                "connector_id": "openalex",
                "query": query.strip(),
                "limit": len(documents),
                "imported_at": datetime.now(timezone.utc).isoformat(),
            },
            "fingerprint": fingerprint,
        }
    )
    documents = [
        document.model_copy(
            update={
                "metadata": {
                    **document.metadata,
                    "source_connector": "openalex",
                    "source_query": query.strip(),
                }
            }
        )
        for document in documents
    ]
    return dataset, documents
