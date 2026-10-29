#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import html
import json
import re
import urllib.parse
import urllib.request
from datetime import date, timedelta
from html.parser import HTMLParser
from pathlib import Path

ARCHIVE = "https://raw.githubusercontent.com/LeilaoMi/hot-news-radar/reports/reports/archive.html"
RAW_BASE = "https://raw.githubusercontent.com/LeilaoMi/hot-news-radar/reports/reports"
UA = "Mozilla/5.0 benchmark-builder/1.0"
START_DATE = date(2026, 9, 1)
END_DATE = date(2026, 10, 28)
TARGET_SIZE = 100


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
    return raw.decode("utf-8", "replace")


def latest_snapshots(archive: str, start: date, end: date) -> list[tuple[str, str]]:
    found = []
    for d in (start + timedelta(days=i) for i in range((end-start).days + 1)):
        ds = d.isoformat()
        times = re.findall(rf'href="{re.escape(ds)}/(\d{{2}}-\d{{2}})\.html"', archive)
        if times:
            found.append((ds, max(times)))
    return found


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
            txt = "".join(self.rank_buf).strip()
            self.current_rank = txt or None
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
        key = topic_id or x["url"]
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
    snaps = latest_snapshots(archive, START_DATE, END_DATE)
    rows = []
    active_days = 0
    for ds, tm in snaps:
        url = f"{RAW_BASE}/{ds}/{tm}.html"
        text = fetch(url)
        items = parse_snapshot(text)
        print(ds, tm, "Tieba items", len(items))
        if items:
            active_days += 1
        for item in items:
            rows.append({"date": ds, "snapshot_time": tm.replace("-", ":"), "snapshot_url": url, **item})

    # Keep newest occurrence of each topic. Sampling from daily final snapshots avoids
    # over-weighting days that happened to have more hourly captures.
    unique_rev = []
    seen = set()
    for r in reversed(rows):
        key = r["topic_id"] or r["title"]
        if key in seen:
            continue
        seen.add(key)
        unique_rev.append(r)
    unique = list(reversed(unique_rev))
    sample = unique[-TARGET_SIZE:]

    fields = ["id","date","snapshot_time","rank","title","topic_id","url","snapshot_url"]
    with (outdir / "tieba_hotspots_100.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(sample, 1):
            w.writerow({"id": i, **r})

    with (outdir / "tieba_hotspots_100.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(sample, 1):
            f.write(json.dumps({"id": i, **r}, ensure_ascii=False) + "\n")

    manifest = {
        "source": "Hot News Radar reports branch only",
        "date_range": [START_DATE.isoformat(), END_DATE.isoformat()],
        "snapshot_policy": "latest snapshot per natural day",
        "days_with_tieba_items": active_days,
        "unique_hotspots_total": len(unique),
        "sample_policy": f"newest {TARGET_SIZE} unique topic_id/title records",
        "sample_size": len(sample),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    if len(sample) < TARGET_SIZE:
        raise SystemExit(f"need {TARGET_SIZE} samples, only got {len(sample)}")

if __name__ == "__main__":
    main()
