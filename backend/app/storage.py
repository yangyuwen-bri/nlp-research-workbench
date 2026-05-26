from __future__ import annotations

import json
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .models import AnalysisRun, AnalysisRunSummary, Dataset, DatasetWorkspace, Document, ExportArtifactSummary
from .settings import get_settings
from .services.ingest import build_dataset_fingerprint

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - optional dependency for postgres runtime only
    psycopg = None
    dict_row = None


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
ANALYSES_DIR = DATA_DIR / "analyses"
EXPORTS_DIR = DATA_DIR / "exports"

for directory in (DATA_DIR, DATASETS_DIR, ANALYSES_DIR, EXPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sortable_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class StorageBackend(ABC):
    @abstractmethod
    def save_dataset(self, dataset: Dataset, documents: List[Document]) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_dataset(self, dataset_id: str, owner_key: Optional[str] = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_datasets(self, owner_key: Optional[str] = None) -> List[Dataset]:
        raise NotImplementedError

    @abstractmethod
    def find_dataset_by_fingerprint(self, fingerprint: str, owner_key: Optional[str] = None) -> Optional[Dataset]:
        raise NotImplementedError

    @abstractmethod
    def delete_dataset(self, dataset_id: str, owner_key: Optional[str] = None) -> bool:
        raise NotImplementedError

    @abstractmethod
    def save_workspace(self, workspace: DatasetWorkspace) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_workspace(self, dataset_id: str) -> DatasetWorkspace:
        raise NotImplementedError

    @abstractmethod
    def save_analysis(self, run: AnalysisRun) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_analysis(self, run_id: str, owner_key: Optional[str] = None) -> AnalysisRun:
        raise NotImplementedError

    @abstractmethod
    def list_analyses(self, owner_key: Optional[str] = None, dataset_id: Optional[str] = None) -> List[AnalysisRunSummary]:
        raise NotImplementedError

    @abstractmethod
    def list_export_artifacts(self, owner_key: Optional[str] = None, dataset_id: Optional[str] = None) -> List[ExportArtifactSummary]:
        raise NotImplementedError


class JsonStorageBackend(StorageBackend):
    def __init__(
        self,
        *,
        datasets_dir: Path = DATASETS_DIR,
        analyses_dir: Path = ANALYSES_DIR,
    ) -> None:
        self.datasets_dir = datasets_dir
        self.analyses_dir = analyses_dir
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.analyses_dir.mkdir(parents=True, exist_ok=True)

    def save_dataset(self, dataset: Dataset, documents: List[Document]) -> None:
        path = self.datasets_dir / f"{dataset.id}.json"
        existing_workspace = None
        if path.exists():
            try:
                existing_workspace = _read_json(path).get("workspace")
            except Exception:
                existing_workspace = None
        _write_json(
            path,
            {
                "dataset": dataset.model_dump(mode="json"),
                "documents": [document.model_dump(mode="json") for document in documents],
                "workspace": existing_workspace
                or DatasetWorkspace(dataset_id=dataset.id).model_dump(mode="json"),
            },
        )

    def load_dataset(self, dataset_id: str, owner_key: Optional[str] = None) -> Dict[str, Any]:
        payload = _read_json(self.datasets_dir / f"{dataset_id}.json")
        if owner_key and payload["dataset"].get("owner_key") != owner_key:
            raise FileNotFoundError(dataset_id)
        return payload

    def list_datasets(self, owner_key: Optional[str] = None) -> List[Dataset]:
        datasets: List[Dataset] = []
        for path in sorted(self.datasets_dir.glob("*.json")):
            payload = _read_json(path)
            if owner_key and payload["dataset"].get("owner_key") != owner_key:
                continue
            datasets.append(Dataset.model_validate(payload["dataset"]))
        return datasets

    def find_dataset_by_fingerprint(self, fingerprint: str, owner_key: Optional[str] = None) -> Optional[Dataset]:
        if not fingerprint:
            return None
        for dataset in self.list_datasets(owner_key=owner_key):
            candidate_fingerprint = dataset.fingerprint
            if not candidate_fingerprint:
                try:
                    payload = self.load_dataset(dataset.id, owner_key=owner_key)
                    documents = [Document.model_validate(item) for item in payload["documents"]]
                    candidate_fingerprint = build_dataset_fingerprint(dataset.name, dataset.text_column, documents)
                except Exception:
                    candidate_fingerprint = None
            if candidate_fingerprint == fingerprint:
                return dataset
        return None

    def delete_dataset(self, dataset_id: str, owner_key: Optional[str] = None) -> bool:
        dataset_path = self.datasets_dir / f"{dataset_id}.json"
        if not dataset_path.exists():
            return False
        if owner_key:
            payload = _read_json(dataset_path)
            if payload["dataset"].get("owner_key") != owner_key:
                return False
        dataset_path.unlink(missing_ok=True)
        for analysis_path in self.analyses_dir.glob("*.json"):
            payload = _read_json(analysis_path)
            if payload.get("dataset_id") != dataset_id:
                continue
            run_id = str(payload.get("id") or analysis_path.stem)
            analysis_path.unlink(missing_ok=True)
            export_dir = EXPORTS_DIR / run_id
            if export_dir.exists():
                for child in export_dir.iterdir():
                    child.unlink(missing_ok=True)
                export_dir.rmdir()
        return True

    def save_workspace(self, workspace: DatasetWorkspace) -> None:
        path = self.datasets_dir / f"{workspace.dataset_id}.json"
        if not path.exists():
            raise FileNotFoundError(workspace.dataset_id)
        payload = _read_json(path)
        payload["workspace"] = workspace.model_dump(mode="json")
        _write_json(path, payload)

    def load_workspace(self, dataset_id: str) -> DatasetWorkspace:
        path = self.datasets_dir / f"{dataset_id}.json"
        payload = _read_json(path)
        workspace = payload.get("workspace")
        if not workspace:
            workspace = DatasetWorkspace(dataset_id=dataset_id).model_dump(mode="json")
            payload["workspace"] = workspace
            _write_json(path, payload)
        return DatasetWorkspace.model_validate(workspace)

    def save_analysis(self, run: AnalysisRun) -> None:
        _write_json(self.analyses_dir / f"{run.id}.json", run.model_dump(mode="json"))

    def load_analysis(self, run_id: str, owner_key: Optional[str] = None) -> AnalysisRun:
        payload = _read_json(self.analyses_dir / f"{run_id}.json")
        if owner_key and payload.get("owner_key") != owner_key:
            raise FileNotFoundError(run_id)
        return AnalysisRun.model_validate(payload)

    def list_analyses(self, owner_key: Optional[str] = None, dataset_id: Optional[str] = None) -> List[AnalysisRunSummary]:
        runs: List[AnalysisRunSummary] = []
        for path in self.analyses_dir.glob("*.json"):
            payload = _read_json(path)
            if owner_key and payload.get("owner_key") != owner_key:
                continue
            if dataset_id and payload.get("dataset_id") != dataset_id:
                continue
            outputs = payload.get("outputs")
            runs.append(
                AnalysisRunSummary(
                    id=payload["id"],
                    owner_key=payload["owner_key"],
                    dataset_id=payload["dataset_id"],
                    created_at=payload["created_at"],
                    status=payload["status"],
                    started_at=payload.get("started_at"),
                    finished_at=payload.get("finished_at"),
                    generator_stack=payload.get("generator_stack", []),
                    settings=payload.get("settings", {}),
                    has_outputs=outputs is not None,
                    export_count=len(outputs.get("exports", [])) if isinstance(outputs, dict) else 0,
                    error=payload.get("error"),
                )
            )
        return sorted(runs, key=lambda run: _sortable_datetime(run.created_at), reverse=True)

    def list_export_artifacts(self, owner_key: Optional[str] = None, dataset_id: Optional[str] = None) -> List[ExportArtifactSummary]:
        items: List[ExportArtifactSummary] = []
        for path in self.analyses_dir.glob("*.json"):
            payload = _read_json(path)
            if owner_key and payload.get("owner_key") != owner_key:
                continue
            if dataset_id and payload.get("dataset_id") != dataset_id:
                continue
            outputs = payload.get("outputs")
            exports = outputs.get("exports", []) if isinstance(outputs, dict) else []
            for artifact in exports:
                items.append(
                    ExportArtifactSummary(
                        run_id=payload["id"],
                        dataset_id=payload["dataset_id"],
                        created_at=payload["created_at"],
                        artifact=artifact["artifact"],
                        format=artifact["format"],
                        path=artifact["path"],
                        rows=artifact["rows"],
                    )
                )
        return sorted(items, key=lambda item: _sortable_datetime(item.created_at), reverse=True)


POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    owner_key TEXT NOT NULL,
    name TEXT NOT NULL,
    source_filename TEXT NOT NULL,
    language TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    document_count INTEGER NOT NULL,
    text_column TEXT NOT NULL,
    labels JSONB NOT NULL,
    fingerprint TEXT NULL,
    dataset_json JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_datasets_fingerprint
    ON datasets(fingerprint);
CREATE INDEX IF NOT EXISTS idx_datasets_owner_created
    ON datasets(owner_key, created_at DESC);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    source_row INTEGER NOT NULL,
    title TEXT NULL,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_dataset_id_source_row
    ON documents(dataset_id, source_row);

CREATE TABLE IF NOT EXISTS dataset_workspaces (
    dataset_id TEXT PRIMARY KEY REFERENCES datasets(id) ON DELETE CASCADE,
    workspace_json JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS analysis_runs (
    id TEXT PRIMARY KEY,
    owner_key TEXT NOT NULL,
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL,
    started_at TIMESTAMPTZ NULL,
    finished_at TIMESTAMPTZ NULL,
    generator_stack JSONB NOT NULL,
    settings JSONB NOT NULL,
    outputs JSONB NULL,
    error TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_analysis_runs_dataset_created
    ON analysis_runs(dataset_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_analysis_runs_status_created
    ON analysis_runs(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_runs_owner_created
    ON analysis_runs(owner_key, created_at DESC);
"""


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class PostgresStorageBackend(StorageBackend):
    def __init__(self, *, database_url: str) -> None:
        if not database_url:
            raise RuntimeError("DATABASE_URL is required for postgres storage backend")
        if psycopg is None:
            raise RuntimeError("psycopg is not installed; cannot use postgres storage backend")
        self.database_url = database_url
        self._ensure_schema()

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        assert psycopg is not None
        with psycopg.connect(self.database_url, row_factory=dict_row) as conn:
            yield conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(POSTGRES_SCHEMA)
                cur.execute("ALTER TABLE datasets ADD COLUMN IF NOT EXISTS owner_key TEXT")
                cur.execute("ALTER TABLE datasets ADD COLUMN IF NOT EXISTS fingerprint TEXT NULL")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_datasets_fingerprint ON datasets(fingerprint)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_datasets_owner_created ON datasets(owner_key, created_at DESC)")
                cur.execute("ALTER TABLE analysis_runs ADD COLUMN IF NOT EXISTS owner_key TEXT")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_analysis_runs_owner_created ON analysis_runs(owner_key, created_at DESC)")
            conn.commit()

    def save_dataset(self, dataset: Dataset, documents: List[Document]) -> None:
        dataset_json = json.dumps(dataset.model_dump(mode="json"), ensure_ascii=False, default=_json_default)
        labels_json = json.dumps(dataset.labels, ensure_ascii=False, default=_json_default)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO datasets (
                        id, owner_key, name, source_filename, language, created_at, document_count, text_column, labels, fingerprint, dataset_json
                    ) VALUES (
                        %(id)s, %(owner_key)s, %(name)s, %(source_filename)s, %(language)s, %(created_at)s, %(document_count)s,
                        %(text_column)s, %(labels)s::jsonb, %(fingerprint)s, %(dataset_json)s::jsonb
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        owner_key = EXCLUDED.owner_key,
                        name = EXCLUDED.name,
                        source_filename = EXCLUDED.source_filename,
                        language = EXCLUDED.language,
                        created_at = EXCLUDED.created_at,
                        document_count = EXCLUDED.document_count,
                        text_column = EXCLUDED.text_column,
                        labels = EXCLUDED.labels,
                        fingerprint = EXCLUDED.fingerprint,
                        dataset_json = EXCLUDED.dataset_json
                    """,
                    {
                        "id": dataset.id,
                        "owner_key": dataset.owner_key,
                        "name": dataset.name,
                        "source_filename": dataset.source_filename,
                        "language": dataset.language,
                        "created_at": dataset.created_at,
                        "document_count": dataset.document_count,
                        "text_column": dataset.text_column,
                        "labels": labels_json,
                        "fingerprint": dataset.fingerprint,
                        "dataset_json": dataset_json,
                    },
                )
                cur.execute("DELETE FROM documents WHERE dataset_id = %s", (dataset.id,))
                rows = [
                    (
                        document.id,
                        document.dataset_id,
                        document.source_row,
                        document.title,
                        document.content,
                        json.dumps(document.metadata, ensure_ascii=False, default=_json_default),
                    )
                    for document in documents
                ]
                if rows:
                    cur.executemany(
                        """
                        INSERT INTO documents (id, dataset_id, source_row, title, content, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                        """,
                        rows,
                    )
                cur.execute(
                    """
                    INSERT INTO dataset_workspaces (dataset_id, workspace_json)
                    VALUES (%s, %s::jsonb)
                    ON CONFLICT (dataset_id) DO NOTHING
                    """,
                    (
                        dataset.id,
                        json.dumps(DatasetWorkspace(dataset_id=dataset.id).model_dump(mode="json"), ensure_ascii=False, default=_json_default),
                    ),
                )
            conn.commit()

    def load_dataset(self, dataset_id: str, owner_key: Optional[str] = None) -> Dict[str, Any]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                if owner_key:
                    cur.execute("SELECT dataset_json FROM datasets WHERE id = %s AND owner_key = %s", (dataset_id, owner_key))
                else:
                    cur.execute("SELECT dataset_json FROM datasets WHERE id = %s", (dataset_id,))
                dataset_row = cur.fetchone()
                if not dataset_row:
                    raise FileNotFoundError(dataset_id)
                cur.execute("SELECT workspace_json FROM dataset_workspaces WHERE dataset_id = %s", (dataset_id,))
                workspace_row = cur.fetchone()
                cur.execute(
                    """
                    SELECT id, dataset_id, source_row, title, content, metadata
                    FROM documents
                    WHERE dataset_id = %s
                    ORDER BY source_row ASC, id ASC
                    """,
                    (dataset_id,),
                )
                document_rows = cur.fetchall()
        return {
            "dataset": dataset_row["dataset_json"],
            "workspace": workspace_row["workspace_json"]
            if workspace_row and workspace_row.get("workspace_json")
            else DatasetWorkspace(dataset_id=dataset_id).model_dump(mode="json"),
            "documents": [
                {
                    "id": row["id"],
                    "dataset_id": row["dataset_id"],
                    "source_row": row["source_row"],
                    "title": row["title"],
                    "content": row["content"],
                    "metadata": row["metadata"],
                }
                for row in document_rows
            ],
        }

    def list_datasets(self, owner_key: Optional[str] = None) -> List[Dataset]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                if owner_key:
                    cur.execute("SELECT dataset_json FROM datasets WHERE owner_key = %s ORDER BY created_at DESC, id ASC", (owner_key,))
                else:
                    cur.execute("SELECT dataset_json FROM datasets ORDER BY created_at DESC, id ASC")
                rows = cur.fetchall()
        return [Dataset.model_validate(row["dataset_json"]) for row in rows]

    def find_dataset_by_fingerprint(self, fingerprint: str, owner_key: Optional[str] = None) -> Optional[Dataset]:
        if not fingerprint:
            return None
        with self._connect() as conn:
            with conn.cursor() as cur:
                if owner_key:
                    cur.execute(
                        "SELECT dataset_json FROM datasets WHERE fingerprint = %s AND owner_key = %s ORDER BY created_at DESC, id DESC LIMIT 1",
                        (fingerprint, owner_key),
                    )
                else:
                    cur.execute(
                        "SELECT dataset_json FROM datasets WHERE fingerprint = %s ORDER BY created_at DESC, id DESC LIMIT 1",
                        (fingerprint,),
                    )
                row = cur.fetchone()
        if row:
            return Dataset.model_validate(row["dataset_json"])
        for dataset in self.list_datasets(owner_key=owner_key):
            if dataset.fingerprint:
                continue
            try:
                payload = self.load_dataset(dataset.id, owner_key=owner_key)
                documents = [Document.model_validate(item) for item in payload["documents"]]
                candidate_fingerprint = build_dataset_fingerprint(dataset.name, dataset.text_column, documents)
            except Exception:
                continue
            if candidate_fingerprint == fingerprint:
                return dataset
        return None

    def delete_dataset(self, dataset_id: str, owner_key: Optional[str] = None) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cur:
                if owner_key:
                    cur.execute("SELECT id FROM analysis_runs WHERE dataset_id = %s AND owner_key = %s", (dataset_id, owner_key))
                else:
                    cur.execute("SELECT id FROM analysis_runs WHERE dataset_id = %s", (dataset_id,))
                run_rows = cur.fetchall()
                run_ids = [str(row["id"]) for row in run_rows]
                if owner_key:
                    cur.execute("DELETE FROM datasets WHERE id = %s AND owner_key = %s", (dataset_id, owner_key))
                else:
                    cur.execute("DELETE FROM datasets WHERE id = %s", (dataset_id,))
                deleted = cur.rowcount > 0
            conn.commit()
        if deleted:
            for run_id in run_ids:
                export_dir = EXPORTS_DIR / run_id
                if export_dir.exists():
                    for child in export_dir.iterdir():
                        child.unlink(missing_ok=True)
                    export_dir.rmdir()
        return deleted

    def save_workspace(self, workspace: DatasetWorkspace) -> None:
        payload = json.dumps(workspace.model_dump(mode="json"), ensure_ascii=False, default=_json_default)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO dataset_workspaces (dataset_id, workspace_json)
                    VALUES (%s, %s::jsonb)
                    ON CONFLICT (dataset_id) DO UPDATE SET
                        workspace_json = EXCLUDED.workspace_json
                    """,
                    (workspace.dataset_id, payload),
                )
            conn.commit()

    def load_workspace(self, dataset_id: str) -> DatasetWorkspace:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT workspace_json FROM dataset_workspaces WHERE dataset_id = %s", (dataset_id,))
                row = cur.fetchone()
                if row and row.get("workspace_json"):
                    return DatasetWorkspace.model_validate(row["workspace_json"])
                cur.execute("SELECT 1 FROM datasets WHERE id = %s", (dataset_id,))
                exists = cur.fetchone()
        if not exists:
            raise FileNotFoundError(dataset_id)
        workspace = DatasetWorkspace(dataset_id=dataset_id)
        self.save_workspace(workspace)
        return workspace

    def save_analysis(self, run: AnalysisRun) -> None:
        outputs_json = (
            json.dumps(run.outputs.model_dump(mode="json"), ensure_ascii=False, default=_json_default)
            if run.outputs is not None
            else None
        )
        cur_payload = {
            "id": run.id,
            "owner_key": run.owner_key,
            "dataset_id": run.dataset_id,
            "created_at": run.created_at,
            "status": run.status,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "generator_stack": json.dumps(run.generator_stack, ensure_ascii=False, default=_json_default),
            "settings": json.dumps(run.settings, ensure_ascii=False, default=_json_default),
            "outputs": outputs_json,
            "error": run.error,
        }
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO analysis_runs (
                        id, owner_key, dataset_id, created_at, status, started_at, finished_at,
                        generator_stack, settings, outputs, error
                    ) VALUES (
                        %(id)s, %(owner_key)s, %(dataset_id)s, %(created_at)s, %(status)s, %(started_at)s, %(finished_at)s,
                        %(generator_stack)s::jsonb, %(settings)s::jsonb, %(outputs)s::jsonb, %(error)s
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        owner_key = EXCLUDED.owner_key,
                        dataset_id = EXCLUDED.dataset_id,
                        created_at = EXCLUDED.created_at,
                        status = EXCLUDED.status,
                        started_at = EXCLUDED.started_at,
                        finished_at = EXCLUDED.finished_at,
                        generator_stack = EXCLUDED.generator_stack,
                        settings = EXCLUDED.settings,
                        outputs = EXCLUDED.outputs,
                        error = EXCLUDED.error
                    """,
                    cur_payload,
                )
            conn.commit()

    def load_analysis(self, run_id: str, owner_key: Optional[str] = None) -> AnalysisRun:
        with self._connect() as conn:
            with conn.cursor() as cur:
                if owner_key:
                    cur.execute(
                        """
                        SELECT id, owner_key, dataset_id, created_at, status, started_at, finished_at, generator_stack, settings, outputs, error
                        FROM analysis_runs
                        WHERE id = %s AND owner_key = %s
                        """,
                        (run_id, owner_key),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, owner_key, dataset_id, created_at, status, started_at, finished_at, generator_stack, settings, outputs, error
                        FROM analysis_runs
                        WHERE id = %s
                        """,
                        (run_id,),
                    )
                row = cur.fetchone()
        if not row:
            raise FileNotFoundError(run_id)
        return AnalysisRun.model_validate(row)

    def list_analyses(self, owner_key: Optional[str] = None, dataset_id: Optional[str] = None) -> List[AnalysisRunSummary]:
        query = """
            SELECT id, owner_key, dataset_id, created_at, status, started_at, finished_at, generator_stack, settings, outputs, error
            FROM analysis_runs
        """
        clauses: list[str] = []
        params: list[Any] = []
        if owner_key:
            clauses.append("owner_key = %s")
            params.append(owner_key)
        if dataset_id:
            clauses.append("dataset_id = %s")
            params.append(dataset_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC, id DESC"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, tuple(params))
                rows = cur.fetchall()
        return [
            AnalysisRunSummary(
                id=row["id"],
                owner_key=row["owner_key"],
                dataset_id=row["dataset_id"],
                created_at=row["created_at"],
                status=row["status"],
                started_at=row["started_at"],
                finished_at=row["finished_at"],
                generator_stack=row["generator_stack"] or [],
                settings=row["settings"] or {},
                has_outputs=row["outputs"] is not None,
                export_count=len((row["outputs"] or {}).get("exports", [])),
                error=row["error"],
            )
            for row in rows
        ]

    def list_export_artifacts(self, owner_key: Optional[str] = None, dataset_id: Optional[str] = None) -> List[ExportArtifactSummary]:
        query = """
            SELECT id, owner_key, dataset_id, created_at, outputs
            FROM analysis_runs
            WHERE outputs IS NOT NULL
        """
        params: list[Any] = []
        if owner_key:
            query += " AND owner_key = %s"
            params.append(owner_key)
        if dataset_id:
            query += " AND dataset_id = %s"
            params.append(dataset_id)
        query += " ORDER BY created_at DESC, id DESC"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, tuple(params))
                rows = cur.fetchall()
        items: List[ExportArtifactSummary] = []
        for row in rows:
            exports = (row["outputs"] or {}).get("exports", [])
            for artifact in exports:
                items.append(
                    ExportArtifactSummary(
                        run_id=row["id"],
                        dataset_id=row["dataset_id"],
                        created_at=row["created_at"],
                        artifact=artifact["artifact"],
                        format=artifact["format"],
                        path=artifact["path"],
                        rows=artifact["rows"],
                    )
                )
        return items


@lru_cache
def get_storage_backend() -> StorageBackend:
    settings = get_settings()
    backend = settings.storage_backend
    if backend == "json":
        return JsonStorageBackend()
    if backend == "postgres":
        return PostgresStorageBackend(database_url=settings.database_url)
    raise RuntimeError(f"Unsupported storage backend: {backend}")


def save_dataset(dataset: Dataset, documents: List[Document]) -> None:
    get_storage_backend().save_dataset(dataset, documents)


def load_dataset(dataset_id: str, owner_key: Optional[str] = None) -> Dict[str, Any]:
    return get_storage_backend().load_dataset(dataset_id, owner_key=owner_key)


def list_datasets(owner_key: Optional[str] = None) -> List[Dataset]:
    return get_storage_backend().list_datasets(owner_key=owner_key)


def find_dataset_by_fingerprint(fingerprint: str, owner_key: Optional[str] = None) -> Optional[Dataset]:
    return get_storage_backend().find_dataset_by_fingerprint(fingerprint, owner_key=owner_key)


def delete_dataset(dataset_id: str, owner_key: Optional[str] = None) -> bool:
    return get_storage_backend().delete_dataset(dataset_id, owner_key=owner_key)


def save_workspace(workspace: DatasetWorkspace) -> None:
    get_storage_backend().save_workspace(workspace)


def load_workspace(dataset_id: str) -> DatasetWorkspace:
    return get_storage_backend().load_workspace(dataset_id)


def save_analysis(run: AnalysisRun) -> None:
    get_storage_backend().save_analysis(run)


def load_analysis(run_id: str, owner_key: Optional[str] = None) -> AnalysisRun:
    return get_storage_backend().load_analysis(run_id, owner_key=owner_key)


def list_analyses(owner_key: Optional[str] = None, dataset_id: Optional[str] = None) -> List[AnalysisRunSummary]:
    return get_storage_backend().list_analyses(owner_key=owner_key, dataset_id=dataset_id)


def list_export_artifacts(owner_key: Optional[str] = None, dataset_id: Optional[str] = None) -> List[ExportArtifactSummary]:
    return get_storage_backend().list_export_artifacts(owner_key=owner_key, dataset_id=dataset_id)


def check_storage_connection() -> tuple[bool, str]:
    settings = get_settings()
    if settings.storage_backend == "json":
        missing = [str(path) for path in (DATA_DIR, DATASETS_DIR, ANALYSES_DIR, EXPORTS_DIR) if not path.exists()]
        if missing:
            return False, f"Missing directories: {', '.join(missing)}"
        return True, "json storage directories ready"

    if settings.storage_backend == "postgres":
        try:
            backend = get_storage_backend()
            assert isinstance(backend, PostgresStorageBackend)
            with backend._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 AS ok")
                    cur.fetchone()
            return True, "postgres reachable"
        except Exception as exc:  # pragma: no cover - exercised in environment checks
            return False, str(exc)

    return False, f"Unsupported storage backend: {settings.storage_backend}"
