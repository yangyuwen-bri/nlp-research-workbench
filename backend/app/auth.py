from __future__ import annotations

from typing import Optional

from fastapi import Header, HTTPException, Query


def get_user_key(
    x_user_key: Optional[str] = Header(default=None, alias="X-User-Key"),
    user_key: Optional[str] = Query(default=None),
) -> str:
    value = (x_user_key or user_key or "").strip()
    if not value:
        raise HTTPException(status_code=401, detail="Missing X-User-Key")
    if len(value) > 128:
        raise HTTPException(status_code=400, detail="Invalid X-User-Key")
    return value
