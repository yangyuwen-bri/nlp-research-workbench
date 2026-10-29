#!/usr/bin/env python3
"""Resolve the latest Hot News Radar snapshot before Asia/Shanghai today and crawl Tieba topics."""
from __future__ import annotations
import datetime as dt
import html
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ARCHIVE = "https://raw.githubusercontent.com/LeilaoMi/hot-news-radar/reports/reports/archive.html"
SNAP_RE = re.compile(r'href=["\']([^"\']*/(20\d{2}-\d{2}-\d{2})/(\d{2}-\d{2})\.html)["\']', re.I)


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def resolve_snapshot() -> str:
    text = html.unescape(fetch(ARCHIVE))
    today = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=8)).date()
    found = []
    for href, ds, ts in SNAP_RE.findall(text):
        d = dt.date.fromisoformat(ds)
        if d < today:
            found.append((d, ts, href))
    if not found:
        raise RuntimeError("archive contains no snapshot before today")
    d = max(x[0] for x in found)
    same = [x for x in found if x[0] == d]
    _, _, href = max(same, key=lambda x: x[1])
    if href.startswith("http"):
        return href
    return "https://raw.githubusercontent.com/LeilaoMi/hot-news-radar/reports/reports/" + href.lstrip("./")


def main():
    snap = resolve_snapshot()
    print("snapshot=" + snap)
    root = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        str(root / "tieba_collector.py"),
        "--snapshot", snap,
        "--db", str(root / "data/tieba.sqlite3"),
        "--export-dir", str(root / "data/export"),
        "--max-hot-topics", "8",
        "--max-threads-per-topic", "12",
        "--max-pages", "3",
        "--sleep", "1.3",
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
