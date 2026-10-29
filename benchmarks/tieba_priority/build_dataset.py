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
        else:
            if tag == "a":
                self._pending_href = None

    def handle_endtag(self, tag):
        if tag == "div" and self.in_source_title:
            self.in_source_title = False
        if tag == "div" and self.in_rank:
            self.in_rank = False
            txt = "".join(self.rank_buf).strip()
            self.current_rank = txt or None
        if tag == "a" and getattr(self, "_pending_href", None):
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
        if getattr(self, "_pending_href", None):
            self._pending_title.append(data)


def parse_snapshot(text: str) -> list[dict]:
    p = TiebaListParser()
    p.feed(text)
    # de-dup within snapshot by topic id/url
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
    # Build from two weeks of latest snapshots; enough diversity for 100 unique topics.
    snaps = latest_snapshots(archive, date(2026, 10, 15), date(2026, 10, 28))
    rows = []
    for ds, tm in snaps:
        url = f"{RAW_BASE}/{ds}/{tm}.html"
        text = fetch(url)
        items = parse_snapshot(text)
        print(ds, tm, "Tieba items", len(items))
        for item in items:
            rows.append({"date": ds, "snapshot_time": tm.replace("-", ":"), "snapshot_url": url, **item})

    # Keep the newest occurrence of the same topic_id/title, then take 100 unique examples.
    unique = []
    seen = set()
    for r in reversed(rows):
        key = r["topic_id"] or r["title"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    unique.reverse()
    sample = unique[-100:]

    with (outdir / "tieba_hotspots_100.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["id","date","snapshot_time","rank","title","topic_id","url","snapshot_url"])
        w.writeheader()
        for i, r in enumerate(sample, 1):
            w.writerow({"id": i, **r})

    with (outdir / "tieba_hotspots_100.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(sample, 1):
            f.write(json.dumps({"id": i, **r}, ensure_ascii=False) + "\n")

    manifest = {
        "source": "Hot News Radar reports branch",
        "date_range": ["2026-10-15", "2026-10-28"],
        "snapshot_policy": "latest snapshot per day",
        "unique_hotspots_total": len(unique),
        "sample_size": len(sample),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    if len(sample) < 100:
        raise SystemExit(f"need 100 samples, only got {len(sample)}")

if __name__ == "__main__":
    main()
