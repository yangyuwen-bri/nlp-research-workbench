from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from ..models import (
    DatasetDetail,
    DatasetLibraryItem,
    DatasetWorkspaceOverview,
    DatasetWorkspacePatch,
    DatasetWorkspaceSnapshot,
    Document,
    SynonymGroup,
    WorkspaceSectionPage,
)
from ..services.ingest import UploadValidationError, ingest_dataset
from ..services.workspace import (
    add_custom_terms,
    add_excluded_terms,
    build_workspace_overview,
    build_workspace_section_page,
    build_workspace_snapshot,
    patch_workspace,
    remove_custom_term,
    remove_excluded_term,
    remove_synonym_group,
    set_curated_terms,
    upsert_synonym_group,
)
from ..storage import list_analyses, list_datasets, load_dataset, load_workspace, save_dataset, save_workspace
from ..storage import delete_dataset as delete_dataset_record
from ..storage import find_dataset_by_fingerprint


router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=list[DatasetLibraryItem])
def get_datasets():
    analyses = list_analyses()
    dataset_stats: dict[str, dict[str, object]] = {}
    for run in analyses:
        stats = dataset_stats.setdefault(
            run.dataset_id,
            {
                "analysis_count": 0,
                "completed_analysis_count": 0,
                "failed_analysis_count": 0,
                "export_count": 0,
                "last_run_at": None,
                "last_run_status": None,
            },
        )
        stats["analysis_count"] = int(stats["analysis_count"]) + 1
        if run.status == "completed":
            stats["completed_analysis_count"] = int(stats["completed_analysis_count"]) + 1
        if run.status == "failed":
            stats["failed_analysis_count"] = int(stats["failed_analysis_count"]) + 1
        stats["export_count"] = int(stats["export_count"]) + run.export_count
        last_run_at = stats["last_run_at"]
        if last_run_at is None or run.created_at >= last_run_at:
            stats["last_run_at"] = run.created_at
            stats["last_run_status"] = run.status

    return [
        DatasetLibraryItem(
            **dataset.model_dump(),
            **dataset_stats.get(dataset.id, {}),
        )
        for dataset in list_datasets()
    ]


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    payload = await file.read()
    try:
        dataset, documents = ingest_dataset(file.filename or "dataset.csv", payload)
    except UploadValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    existing = find_dataset_by_fingerprint(dataset.fingerprint or "")
    if existing:
        existing_payload = load_dataset(existing.id)
        existing_documents = existing_payload["documents"]
        return {
            "dataset": existing,
            "documents": existing_documents[:5],
            "preview_count": min(5, len(existing_documents)),
            "reused_existing": True,
        }
    save_dataset(dataset, documents)
    return {"dataset": dataset, "documents": documents[:5], "preview_count": min(5, len(documents)), "reused_existing": False}


@router.delete("/{dataset_id}")
def remove_dataset(dataset_id: str):
    deleted = delete_dataset_record(dataset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"ok": True, "dataset_id": dataset_id}


@router.get("/{dataset_id}", response_model=DatasetDetail)
def get_dataset(dataset_id: str, limit: int = Query(default=20, ge=1, le=200)):
    try:
        payload = load_dataset(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset not found") from exc
    preview_payload = dict(payload)
    preview_payload["documents"] = payload["documents"][:limit]
    return preview_payload


def _load_dataset_and_workspace(dataset_id: str):
    try:
        payload = load_dataset(dataset_id)
        workspace = load_workspace(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset not found") from exc
    documents = [Document.model_validate(item) for item in payload["documents"]]
    return payload, documents, workspace


@router.get("/{dataset_id}/workspace", response_model=DatasetWorkspaceSnapshot)
def get_dataset_workspace(dataset_id: str):
    payload, documents, workspace = _load_dataset_and_workspace(dataset_id)
    return build_workspace_snapshot(
        workspace,
        documents,
        top_k_terms=workspace.auto_top_k_terms,
    )


@router.get("/{dataset_id}/workspace/summary", response_model=DatasetWorkspaceOverview)
def get_dataset_workspace_summary(dataset_id: str):
    _, documents, workspace = _load_dataset_and_workspace(dataset_id)
    return build_workspace_overview(workspace, documents, top_k_terms=workspace.auto_top_k_terms)


@router.get("/{dataset_id}/workspace/sections/{section}", response_model=WorkspaceSectionPage)
def get_dataset_workspace_section(
    dataset_id: str,
    section: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
):
    _, documents, workspace = _load_dataset_and_workspace(dataset_id)
    try:
        return build_workspace_section_page(
            workspace,
            documents,
            section=section,
            page=page,
            page_size=page_size,
            top_k_terms=workspace.auto_top_k_terms,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Workspace section not found") from exc


@router.put("/{dataset_id}/workspace", response_model=DatasetWorkspaceSnapshot)
def update_dataset_workspace(dataset_id: str, patch: DatasetWorkspacePatch):
    payload, documents, workspace = _load_dataset_and_workspace(dataset_id)
    updated = patch_workspace(workspace, patch)
    save_workspace(updated)
    return build_workspace_snapshot(updated, documents, top_k_terms=updated.auto_top_k_terms)


@router.post("/{dataset_id}/workspace/custom-terms", response_model=DatasetWorkspaceSnapshot)
def add_dataset_custom_terms(dataset_id: str, terms: list[str]):
    payload, documents, workspace = _load_dataset_and_workspace(dataset_id)
    updated = add_custom_terms(workspace, terms)
    save_workspace(updated)
    return build_workspace_snapshot(updated, documents, top_k_terms=updated.auto_top_k_terms)


@router.delete("/{dataset_id}/workspace/custom-terms/{term}", response_model=DatasetWorkspaceSnapshot)
def delete_dataset_custom_term(dataset_id: str, term: str):
    payload, documents, workspace = _load_dataset_and_workspace(dataset_id)
    updated = remove_custom_term(workspace, term)
    save_workspace(updated)
    return build_workspace_snapshot(updated, documents, top_k_terms=updated.auto_top_k_terms)


@router.post("/{dataset_id}/workspace/excluded-terms", response_model=DatasetWorkspaceSnapshot)
def add_dataset_excluded_terms(dataset_id: str, terms: list[str]):
    payload, documents, workspace = _load_dataset_and_workspace(dataset_id)
    updated = add_excluded_terms(workspace, terms)
    save_workspace(updated)
    return build_workspace_snapshot(updated, documents, top_k_terms=updated.auto_top_k_terms)


@router.delete("/{dataset_id}/workspace/excluded-terms/{term}", response_model=DatasetWorkspaceSnapshot)
def delete_dataset_excluded_term(dataset_id: str, term: str):
    payload, documents, workspace = _load_dataset_and_workspace(dataset_id)
    updated = remove_excluded_term(workspace, term)
    save_workspace(updated)
    return build_workspace_snapshot(updated, documents, top_k_terms=updated.auto_top_k_terms)


@router.post("/{dataset_id}/workspace/synonym-groups", response_model=DatasetWorkspaceSnapshot)
def put_dataset_synonym_group(dataset_id: str, group: SynonymGroup):
    payload, documents, workspace = _load_dataset_and_workspace(dataset_id)
    updated = upsert_synonym_group(workspace, group)
    save_workspace(updated)
    return build_workspace_snapshot(updated, documents, top_k_terms=updated.auto_top_k_terms)


@router.delete("/{dataset_id}/workspace/synonym-groups/{canonical_term}", response_model=DatasetWorkspaceSnapshot)
def delete_dataset_synonym_group(dataset_id: str, canonical_term: str):
    payload, documents, workspace = _load_dataset_and_workspace(dataset_id)
    updated = remove_synonym_group(workspace, canonical_term)
    save_workspace(updated)
    return build_workspace_snapshot(updated, documents, top_k_terms=updated.auto_top_k_terms)


@router.put("/{dataset_id}/workspace/curated-terms", response_model=DatasetWorkspaceSnapshot)
def update_dataset_curated_terms(dataset_id: str, terms: list[str]):
    payload, documents, workspace = _load_dataset_and_workspace(dataset_id)
    updated = set_curated_terms(workspace, terms)
    save_workspace(updated)
    return build_workspace_snapshot(updated, documents, top_k_terms=updated.auto_top_k_terms)
