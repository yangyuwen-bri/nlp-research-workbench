from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..auth import get_user_key
from ..models import (
    DataSourceAsset,
    DataSourceImportRecord,
    HuggingFaceImportRequest,
    HuggingFacePreviewRequest,
    HuggingFacePreviewResponse,
    HuggingFaceSearchRequest,
    HuggingFaceSearchResponse,
    OpenAlexImportRequest,
    OpenAlexPreviewRequest,
    OpenAlexPreviewResponse,
    SourceConnectorItem,
)
from ..services.data_sources import (
    import_builtin_asset,
    import_huggingface_dataset,
    import_openalex_query,
    load_builtin_assets,
    preview_huggingface_dataset,
    preview_openalex,
    search_huggingface_datasets,
)
from ..storage import find_dataset_by_fingerprint, list_datasets, load_dataset, save_dataset


router = APIRouter(prefix="/sources", tags=["sources"])


@router.get("/library", response_model=list[DataSourceAsset])
def get_builtin_library():
    return load_builtin_assets()


@router.get("/connectors", response_model=list[SourceConnectorItem])
def get_source_connectors():
    return [
        SourceConnectorItem(
            id="openalex",
            name="OpenAlex",
            description="从开放学术目录中导入论文标题与摘要，适合文科研究、主题发现和术语分析。",
            input_hint="输入研究主题、关键词或学科方向，例如“数字人文”“platform governance”。",
            status="ready",
        ),
        SourceConnectorItem(
            id="huggingface",
            name="Hugging Face Datasets",
            description="搜索 Hugging Face Hub 上的公开文本数据集，并将样本直接导入当前分析系统。",
            input_hint="输入数据集关键词，例如“weibo sentiment”“news classification”“hotel review”。",
            status="ready",
        ),
    ]


@router.get("/imports", response_model=list[DataSourceImportRecord])
def get_source_imports(user_key: str = Depends(get_user_key)):
    items: list[DataSourceImportRecord] = []
    for dataset in list_datasets(owner_key=user_key):
        if dataset.source_type not in {"library", "connector"}:
            continue
        items.append(
            DataSourceImportRecord(
                dataset_id=dataset.id,
                dataset_name=dataset.name,
                source_type=dataset.source_type,
                source_label=dataset.source_label,
                source_reference=dataset.source_reference,
                created_at=dataset.created_at,
                document_count=dataset.document_count,
            )
        )
    return items


def _save_or_reuse_dataset(dataset, documents, user_key: str):
    existing = find_dataset_by_fingerprint(dataset.fingerprint or "", owner_key=user_key)
    if existing:
        payload = load_dataset(existing.id, owner_key=user_key)
        return {
            "dataset": existing,
            "documents": payload["documents"][:5],
            "preview_count": min(5, len(payload["documents"])),
            "reused_existing": True,
        }
    save_dataset(dataset, documents)
    return {
        "dataset": dataset,
        "documents": documents[:5],
        "preview_count": min(5, len(documents)),
        "reused_existing": False,
    }


@router.post("/library/{asset_id}/import")
def import_library_asset(asset_id: str, user_key: str = Depends(get_user_key)):
    try:
        dataset, documents = import_builtin_asset(asset_id, owner_key=user_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Data source asset not found") from exc
    return _save_or_reuse_dataset(dataset, documents, user_key)


@router.post("/connectors/openalex/preview", response_model=OpenAlexPreviewResponse)
async def preview_openalex_connector(request: OpenAlexPreviewRequest):
    try:
        return await preview_openalex(request.query, limit=request.limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAlex 预览失败：{exc}") from exc


@router.post("/connectors/openalex/import")
async def import_openalex_connector(request: OpenAlexImportRequest, user_key: str = Depends(get_user_key)):
    try:
        dataset, documents = await import_openalex_query(request.query, limit=request.limit, owner_key=user_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAlex 导入失败：{exc}") from exc
    return _save_or_reuse_dataset(dataset, documents, user_key)


@router.post("/connectors/huggingface/search", response_model=HuggingFaceSearchResponse)
async def search_huggingface_connector(request: HuggingFaceSearchRequest):
    try:
        return await search_huggingface_datasets(request.query, limit=request.limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Hugging Face 搜索失败：{exc}") from exc


@router.post("/connectors/huggingface/preview", response_model=HuggingFacePreviewResponse)
async def preview_huggingface_connector(request: HuggingFacePreviewRequest):
    try:
        return await preview_huggingface_dataset(request.dataset_id, limit=request.limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Hugging Face 预览失败：{exc}") from exc


@router.post("/connectors/huggingface/import")
async def import_huggingface_connector(request: HuggingFaceImportRequest, user_key: str = Depends(get_user_key)):
    try:
        dataset, documents = await import_huggingface_dataset(request.dataset_id, limit=request.limit, owner_key=user_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Hugging Face 导入失败：{exc}") from exc
    return _save_or_reuse_dataset(dataset, documents, user_key)
