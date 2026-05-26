from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import httpx
import numpy as np

from ..settings import Settings, get_settings
from ..storage import DATA_DIR
from .llm import BailianClientError

EMBEDDING_BATCH_SIZE = 10
_EMBEDDING_CACHE: Dict[Tuple[str, int, str], List[float]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_DB = DATA_DIR / "cache" / "embeddings.sqlite3"


def _cache_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ensure_disk_cache() -> None:
    _CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_CACHE_DB) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                text_hash TEXT NOT NULL,
                vector TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (model, dimensions, text_hash)
            )
            """
        )


def _read_disk_cache(settings: Settings, texts: List[str]) -> Dict[int, List[float]]:
    if not texts:
        return {}
    with _CACHE_LOCK:
        _ensure_disk_cache()
        with sqlite3.connect(_CACHE_DB) as connection:
            hits: Dict[int, List[float]] = {}
            for index, text in enumerate(texts):
                row = connection.execute(
                    """
                    SELECT vector
                    FROM embeddings
                    WHERE model = ? AND dimensions = ? AND text_hash = ?
                    """,
                    (
                        settings.dashscope_embedding_model,
                        settings.dashscope_embedding_dimensions,
                        _cache_hash(text),
                    ),
                ).fetchone()
                if row:
                    hits[index] = json.loads(row[0])
            return hits


def _write_disk_cache(settings: Settings, rows: List[Tuple[str, List[float]]]) -> None:
    if not rows:
        return
    with _CACHE_LOCK:
        _ensure_disk_cache()
        with sqlite3.connect(_CACHE_DB) as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO embeddings (model, dimensions, text_hash, vector)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        settings.dashscope_embedding_model,
                        settings.dashscope_embedding_dimensions,
                        _cache_hash(text),
                        json.dumps(vector),
                    )
                    for text, vector in rows
                ],
            )


def _fetch_embedding_batch(settings: Settings, batch: List[str]) -> List[List[float]]:
    payload = {
        "model": settings.dashscope_embedding_model,
        "input": batch,
        "dimensions": settings.dashscope_embedding_dimensions,
        "encoding_format": "float",
    }
    request_kwargs = {
        "headers": {
            "Authorization": f"Bearer {settings.dashscope_api_key}",
            "Content-Type": "application/json",
        },
        "timeout": httpx.Timeout(settings.dashscope_timeout_seconds, connect=15.0),
    }
    with httpx.Client(trust_env=False) as client:
        try:
            response = client.post(
                f"{settings.dashscope_base_url}/embeddings",
                json=payload,
                **request_kwargs,
            )
        except (httpx.ConnectTimeout, httpx.ReadTimeout):
            response = client.post(
                f"{settings.dashscope_base_url}/embeddings",
                json=payload,
                **request_kwargs,
            )
    response.raise_for_status()
    body = response.json()
    data = body.get("data") or []
    if len(data) != len(batch):
        raise BailianClientError("DashScope embedding returned an unexpected number of vectors.")
    ordered = sorted(data, key=lambda item: item["index"])
    return [item["embedding"] for item in ordered]


def embed_texts(texts: List[str]) -> np.ndarray:
    settings: Settings = get_settings()
    if not settings.embedding_ready:
        raise BailianClientError("DashScope embedding is not configured.")
    if not texts:
        return np.zeros((0, settings.dashscope_embedding_dimensions), dtype=float)

    normalized_texts = [text[:4000] for text in texts]
    cache_keys = [
        (settings.dashscope_embedding_model, settings.dashscope_embedding_dimensions, text)
        for text in normalized_texts
    ]
    vectors: List[List[float] | None] = [_EMBEDDING_CACHE.get(key) for key in cache_keys]
    missing_positions = [index for index, vector in enumerate(vectors) if vector is None]

    disk_hits = _read_disk_cache(settings, [normalized_texts[index] for index in missing_positions])
    for relative_index, vector in disk_hits.items():
        position = missing_positions[relative_index]
        vectors[position] = vector
        _EMBEDDING_CACHE[cache_keys[position]] = vector

    missing_positions = [index for index, vector in enumerate(vectors) if vector is None]
    missing_texts = [normalized_texts[index] for index in missing_positions]
    batches = [
        missing_texts[index : index + EMBEDDING_BATCH_SIZE]
        for index in range(0, len(missing_texts), EMBEDDING_BATCH_SIZE)
    ]
    if batches:
        batch_offsets = []
        cursor = 0
        for batch in batches:
            batch_offsets.append(cursor)
            cursor += len(batch)

        max_workers = min(settings.dashscope_embedding_concurrency, len(batches))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_fetch_embedding_batch, settings, batch): (offset, batch)
                for offset, batch in zip(batch_offsets, batches)
            }
            for future in as_completed(futures):
                offset, batch = futures[future]
                embeddings = future.result()
                cache_rows: List[Tuple[str, List[float]]] = []
                for item_index, embedding in enumerate(embeddings):
                    position = missing_positions[offset + item_index]
                    vectors[position] = embedding
                    _EMBEDDING_CACHE[cache_keys[position]] = embedding
                    cache_rows.append((normalized_texts[position], embedding))
                _write_disk_cache(settings, cache_rows)
    return np.asarray(vectors, dtype=float)
