from __future__ import annotations

import io
import json
import re
import uuid
import hashlib
from datetime import datetime, timezone
from typing import List, Tuple

import pandas as pd

from ..models import Dataset, Document
from ..settings import Settings, get_settings


TEXT_COLUMN_CANDIDATES = [
    "正文",
    "内容",
    "content",
    "text",
    "评论",
    "文本",
    "review",
    "comment",
    "message",
    "description",
    "desc",
    "body",
    "document",
    "sentence",
    "post",
]
TITLE_COLUMN_CANDIDATES = ["标题", "title", "主题", "subject"]
NON_TEXT_COLUMN_HINTS = [
    "label",
    "标签",
    "分类",
    "cat",
    "score",
    "rating",
    "id",
    "编号",
    "时间",
    "date",
    "日期",
    "phone",
    "mobile",
    "邮箱",
    "email",
    "url",
    "link",
]
TEXT_CHAR_RE = re.compile(r"[\u4e00-\u9fffA-Za-z]")
NUMERIC_RE = re.compile(r"^[\d\s.\-_/:%]+$")


class UploadValidationError(ValueError):
    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def _validate_payload_size(payload: bytes, settings: Settings) -> None:
    if not payload:
        raise UploadValidationError("上传文件为空，请重新选择文件。")
    if len(payload) > settings.upload_max_file_bytes:
        max_megabytes = settings.upload_max_file_bytes / (1024 * 1024)
        raise UploadValidationError(
            f"上传文件过大，当前仅支持不超过 {max_megabytes:.0f} MB 的文件。",
            status_code=413,
        )


def _read_upload(filename: str, payload: bytes) -> pd.DataFrame:
    suffix = filename.lower().rsplit(".", 1)[-1]
    if suffix == "csv":
        return pd.read_csv(io.BytesIO(payload))
    if suffix in {"xlsx", "xls"}:
        return pd.read_excel(io.BytesIO(payload))
    if suffix == "jsonl":
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
        return pd.DataFrame(rows)
    raise UploadValidationError("文件格式不支持，请上传 CSV、XLSX、XLS 或 JSONL 文件。")


def _validate_frame(frame: pd.DataFrame, settings: Settings) -> None:
    row_count, column_count = frame.shape
    if column_count == 0:
        raise UploadValidationError("上传文件没有可用列，请检查文件内容。")
    if row_count == 0:
        raise UploadValidationError("上传文件没有可用数据，请检查文件内容。")
    if row_count > settings.upload_max_rows:
        raise UploadValidationError(f"上传数据行数过多，当前最多支持 {settings.upload_max_rows} 行。")
    if column_count > settings.upload_max_columns:
        raise UploadValidationError(f"上传数据列数过多，当前最多支持 {settings.upload_max_columns} 列。")


def _normalize_column_name(column: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(column).strip().lower())


def _name_score(column: str) -> float:
    normalized = _normalize_column_name(column)
    score = 0.0
    for candidate in TEXT_COLUMN_CANDIDATES:
        candidate_normalized = _normalize_column_name(candidate)
        if normalized == candidate_normalized:
            score += 120
        elif candidate_normalized and candidate_normalized in normalized:
            score += 48
    for hint in NON_TEXT_COLUMN_HINTS:
        hint_normalized = _normalize_column_name(hint)
        if normalized == hint_normalized:
            score -= 80
        elif hint_normalized and hint_normalized in normalized:
            score -= 24
    return score


def _series_text_score(series: pd.Series) -> float:
    values = [str(value).strip() for value in series.tolist() if str(value).strip()]
    if not values:
        return float("-inf")

    sample = values[:50]
    lengths = [len(value) for value in sample]
    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)
    unique_ratio = len(set(sample)) / len(sample)
    text_ratio = sum(1 for value in sample if TEXT_CHAR_RE.search(value)) / len(sample)
    numeric_ratio = sum(1 for value in sample if NUMERIC_RE.fullmatch(value)) / len(sample)
    multiline_ratio = sum(1 for value in sample if any(p in value for p in "，。！？；,.!?; ")) / len(sample)

    score = 0.0
    score += min(avg_len, 120) * 1.1
    score += min(max_len, 200) * 0.15
    score += unique_ratio * 12
    score += text_ratio * 28
    score += multiline_ratio * 8
    score -= numeric_ratio * 40

    if avg_len < 2:
        score -= 30
    if max_len < 6:
        score -= 12
    return score


def _pick_text_column(frame: pd.DataFrame) -> str:
    columns = frame.columns.tolist()
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate

    best_column = ""
    best_score = float("-inf")
    for column in columns:
        score = _name_score(column) + _series_text_score(frame[column])
        if score > best_score:
            best_score = score
            best_column = column

    if best_column and best_score >= 12:
        return best_column

    readable_columns = "、".join(str(column) for column in columns[:8])
    raise UploadValidationError(
        f"暂时无法判断哪一列是正文。请确认文件里包含可分析文本列；当前识别到的列有：{readable_columns}。"
    )


def build_dataset_fingerprint(name: str, text_column: str, documents: List[Document]) -> str:
    fingerprint_payload = [
        {
            "title": document.title or "",
            "content": document.content,
            "metadata": document.metadata,
        }
        for document in documents
    ]
    fingerprint_source = json.dumps(
        {
            "name": name,
            "text_column": text_column,
            "documents": fingerprint_payload,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()


def ingest_dataset(
    filename: str,
    payload: bytes,
    *,
    deduplicate: bool = True,
    settings: Settings | None = None,
) -> Tuple[Dataset, List[Document]]:
    runtime = settings or get_settings()
    _validate_payload_size(payload, runtime)
    try:
        frame = _read_upload(filename, payload).fillna("")
    except UnicodeDecodeError as exc:
        raise UploadValidationError("文件编码无法识别，请保存为 UTF-8 后重新上传。") from exc
    except json.JSONDecodeError as exc:
        raise UploadValidationError("JSONL 文件格式错误，请确保每行都是完整的 JSON 记录。") from exc
    except (ValueError, TypeError) as exc:
        if isinstance(exc, UploadValidationError):
            raise
        raise UploadValidationError("文件内容解析失败，请检查文件格式后重试。") from exc
    _validate_frame(frame, runtime)
    if deduplicate:
        frame = frame.drop_duplicates().reset_index(drop=True)
    else:
        frame = frame.reset_index(drop=True)
    text_column = _pick_text_column(frame)
    title_column = next((column for column in TITLE_COLUMN_CANDIDATES if column in frame.columns), None)
    dataset_id = f"ds_{uuid.uuid4().hex[:10]}"
    documents: List[Document] = []
    for index, row in frame.iterrows():
        content = str(row[text_column]).strip()
        if not content:
            continue
        if len(content) > runtime.upload_max_text_length:
            raise UploadValidationError(
                f"存在超长文本，单条正文最多支持 {runtime.upload_max_text_length} 个字符。"
            )
        metadata = {column: row[column] for column in frame.columns if column not in {text_column, title_column}}
        documents.append(
            Document(
                id=f"doc_{uuid.uuid4().hex[:12]}",
                dataset_id=dataset_id,
                source_row=index + 1,
                title=str(row[title_column]).strip() if title_column else None,
                content=content,
                metadata=metadata,
            )
        )
    if not documents:
        raise UploadValidationError("未读取到有效正文，请确认正文列中包含文本内容。")
    fingerprint = build_dataset_fingerprint(filename.rsplit(".", 1)[0], text_column, documents)
    dataset = Dataset(
        id=dataset_id,
        name=filename.rsplit(".", 1)[0],
        source_filename=filename,
        created_at=datetime.now(timezone.utc),
        document_count=len(documents),
        text_column=text_column,
        fingerprint=fingerprint,
    )
    return dataset, documents
