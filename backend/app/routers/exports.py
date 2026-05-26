from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from ..auth import get_user_key
from ..models import ExportArtifactSummary
from ..storage import list_export_artifacts, load_analysis


router = APIRouter(prefix="/exports", tags=["exports"])


@router.get("", response_model=List[ExportArtifactSummary])
def get_exports(dataset_id: Optional[str] = None, user_key: str = Depends(get_user_key)):
    return list_export_artifacts(owner_key=user_key, dataset_id=dataset_id)


@router.get("/{run_id}/{artifact}/{fmt}")
def download_export(run_id: str, artifact: str, fmt: str, user_key: str = Depends(get_user_key)):
    try:
        run = load_analysis(run_id, owner_key=user_key)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Analysis not found") from exc
    if not run.outputs:
        raise HTTPException(status_code=404, detail="Analysis outputs unavailable")
    matched = next(
        (item for item in run.outputs.exports if item.artifact == artifact and item.format == fmt),
        None,
    )
    if not matched:
        raise HTTPException(status_code=404, detail="Export not found")
    path = Path(matched.path)
    return FileResponse(path, filename=path.name)
