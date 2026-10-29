#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Daily runner for the standalone Tieba Hotspot Collector.

Selects the most recent Hot News Radar snapshot strictly before today's
Asia/Shanghai date, then expands Tieba hottopics and collects discussions.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ARCHIVE_URL = "https://raw.githubusercontent.com/LeilaoMi/hot-news-radar/reports/reports/archive.html"
SNAPSHOT_RE = re.compile(r"(?:https?://raw\.githubusercontent\.com/LeilaoMi/hot-news-radar/reports/reports/)?(\d{4}-\d{2}-\d{2})/(\d{2}-\d{2})\.html")
UTC8 = dt.timezone(dt.timedelta(hours=8))


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "tieba-hotspot-collector/0.2"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    return raw.decode("utf-8", "replace")


def select_snapshot(archive_html: str, today: dt.date) -> tuple[str, str, str]:
    decoded = html.unescape(archive_html)
    candidates: dict[str, set[str]] = {}
    for date_str, time_str in SNAPSHOT_RE.findall(decoded):
        if dt.date.fromisoformat(date_str) >= today:
            continue
        candidates.setdefault(date_str, set()).add(time_str)
    if not candidates:
        raise RuntimeError("Hot News Radar archive has no snapshot before today")
    date_str = max(candidates)
    time_str = max(candidates[date_str])
    url = f"https://raw.githubusercontent.com/LeilaoMi/hot-news-radar/reports/reports/{date_str}/{time_str}.html"
    return date_str, time_str, url


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-hot-topics", type=int, default=12)
    parser.add_argument("--max-threads-per-topic", type=int, default=3)
    parser.add_argument("--max-pages", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=1.2)
    parser.add_argument("--no-subposts", action="store_true")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()

    now = dt.datetime.now(UTC8)
    date_str, time_str, snapshot_url = select_snapshot(fetch_text(ARCHIVE_URL), now.date())
    print(f"selected Hot News Radar snapshot: {date_str} {time_str} {snapshot_url}", file=sys.stderr)

    root = Path(args.data_root)
    export_dir = root / "runs" / date_str / time_str
    export_dir.mkdir(parents=True, exist_ok=True)
    db_path = root / "tieba.sqlite3"
    metadata_path = export_dir / "run-meta.json"
    metadata_path.write_text(
        "{\n"
        f'  "data_date": "{date_str}",\n'
        f'  "snapshot_time": "{time_str}",\n'
        f'  "snapshot_url": "{snapshot_url}"\n'
        "}\n",
        encoding="utf-8",
    )

    collector = Path(__file__).with_name("tieba_collector_v2.py")
    cmd = [
        sys.executable,
        str(collector),
        "--snapshot",
        snapshot_url,
        "--max-hot-topics",
        str(args.max_hot_topics),
        "--max-threads-per-topic",
        str(args.max_threads_per_topic),
        "--max-pages",
        str(args.max_pages),
        "--sleep",
        str(args.sleep),
        "--db",
        str(db_path),
        "--export-dir",
        str(export_dir),
        "--min-success",
        "1",
    ]
    if args.no_subposts:
        cmd.append("--no-subposts")
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
