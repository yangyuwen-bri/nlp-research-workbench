#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import html
import json
import re
import urllib.parse
import urllib.request
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

ARCHIVE = "https://raw.githubusercontent.com/LeilaoMi/hot-news-radar/reports/reports/archive.html"
RAW_BASE = "https://raw.githubusercontent.com/LeilaoMi/hot-news-radar/reports/reports"
UA = "Mozilla/5.0 benchmark-builder/1.0"
START_DATE = date(2026, 9, 20)
END_DATE = date(2026, 10, 28)
TARGET_SIZE = 100


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def all_snapshots(archive: str, start: date, end: date) -> list[tuple[str, str]]:
    rows = []
    for ds, tm in re.findall(r'href="(\d{4}-\d{2}-\d{2})/(\d{2}-\d{2})\.html"', archive):
        d = date.fromisoformat(ds)
        if start <= d <= end:
            rows.append((ds, tm))
    return sorted(set(rows), reverse=True)


class TiebaListParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_source_title = False
        self.tieba_section = False
        self.in_rank = False
        self.rank_buf = []
        self.current_rank = None
        self.items = []
        self._pending_href = None
        self._pending_title = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        cls = attrs.get("class", "")
        if tag == "div" and "new-source-title" in cls.split():
            self.in_source_title = True
            self.tieba_section = False
        elif tag == "div" and self.tieba_section and "new-item-rank" in cls.split():
            self.in_rank = True
            self.rank_buf = []
        elif tag == "a" and self.tieba_section:
            href = html.unescape(attrs.get("href", ""))
            if "tieba.baidu.com/hottopic/browse/hottopic" in href:
                self._pending_href = href
                self._pending_title = []
            else:
                self._pending_href = None

    def handle_endtag(self, tag):
        if tag == "div" and self.in_source_title:
            self.in_source_title = False
        if tag == "div" and self.in_rank:
            self.in_rank = False
            self.current_rank = "".join(self.rank_buf).strip() or None
        if tag == "a" and self._pending_href:
            title = html.unescape("".join(self._pending_title)).strip()
            if title:
                self.items.append({"title": title, "rank": self.current_rank, "url": self._pending_href})
            self._pending_href = None

    def handle_data(self, data):
        if self.in_source_title:
            if data.strip().startswith("贴吧"):
                self.tieba_section = True
            elif data.strip():
                self.tieba_section = False
        if self.in_rank:
            self.rank_buf.append(data)
        if self._pending_href:
            self._pending_title.append(data)


def parse_snapshot(text: str) -> list[dict]:
    p = TiebaListParser()
    p.feed(text)
    out, seen = [], set()
    for x in p.items:
        qs = urllib.parse.parse_qs(urllib.parse.urlsplit(x["url"]).query)
        topic_id = (qs.get("topic_id") or [""])[0]
        key = topic_id or x["title"]
        if key in seen:
            continue
        seen.add(key)
        x["topic_id"] = topic_id
        out.append(x)
    return out


def main():
    outdir = Path("benchmark_out")
    outdir.mkdir(exist_ok=True)
    archive = fetch(ARCHIVE)
    snaps = all_snapshots(archive, START_DATE, END_DATE)

    # Scan newest snapshots first, but count a topic only once by topic_id/title.
    # This gives 100 distinct decision cases without overweighting repeated hourly listings.
    unique = []
    seen = set()
    scanned = 0
    active_snapshots = 0
    for ds, tm in snaps:
        url = f"{RAW_BASE}/{ds}/{tm}.html"
        items = parse_snapshot(fetch(url))
        scanned += 1
        if items:
            active_snapshots += 1
        for item in items:
            key = item["topic_id"] or item["title"]
            if key in seen:
                continue
            seen.add(key)
            unique.append({
                "date": ds,
                "snapshot_time": tm.replace("-", ":"),
                "snapshot_url": url,
                **item,
            })
        print(ds, tm, "Tieba items", len(items), "unique", len(unique))
        if len(unique) >= TARGET_SIZE:
            break

    sample = unique[:TARGET_SIZE]
    fields = ["id","date","snapshot_time","rank","title","topic_id","url","snapshot_url"]
    with (outdir / "tieba_hotspots_100.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, row in enumerate(sample, 1):
            w.writerow({"id": i, **row})

    with (outdir / "tieba_hotspots_100.jsonl").open("w", encoding="utf-8") as f:
        for i, row in enumerate(sample, 1):
            f.write(json.dumps({"id": i, **row}, ensure_ascii=False) + "\n")

    manifest = {
        "source": "Hot News Radar reports branch only",
        "date_range": [START_DATE.isoformat(), END_DATE.isoformat()],
        "snapshot_policy": "scan historical snapshots newest-first; deduplicate globally by topic_id/title",
        "reason": "benchmark needs distinct decision cases; hourly repeats count once",
        "snapshots_scanned": scanned,
        "snapshots_with_tieba_items": active_snapshots,
        "unique_hotspots_found": len(unique),
        "sample_size": len(sample),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    if len(sample) < TARGET_SIZE:
        raise SystemExit(f"need {TARGET_SIZE} samples, only got {len(sample)}")

if __name__ == "__main__":
    main()
