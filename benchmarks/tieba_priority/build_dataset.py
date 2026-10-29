#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a reproducible Tieba hotspot-priority benchmark from Hot News Radar.

Policy:
- read Hot News Radar archive.html
- for each day use only the latest snapshot of that day
- extract Tieba hottopic URLs
- deduplicate globally by topic_id
- enrich each case with up to 3 representative thread titles from the public hottopic page
- emit a simple keyword baseline plus empty gold-label columns

Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ARCHIVE = "https://raw.githubusercontent.com/LeilaoMi/hot-news-radar/reports/reports/archive.html"
RAW_BASE = "https://raw.githubusercontent.com/LeilaoMi/hot-news-radar/reports/reports/"
UA = "Mozilla/5.0 benchmark-builder/1.1"
SNAP_RE = re.compile(r'href=["\']([^"\']*(20\d{2}-\d{2}-\d{2})/(\d{2}-\d{2})\.html)["\']', re.I)
HOT_RE = re.compile(r'https?://(?:c\.)?tieba\.baidu\.com/hottopic/browse/hottopic\?[^"\'<>\s]+', re.I)
THREAD_RE = re.compile(r'<a[^>]+href=["\'](?:https?://(?:c\.)?tieba\.baidu\.com)?/p/(\d+)[^"\']*["\'][^>]*>(.*?)</a>', re.I | re.S)
TAG_RE = re.compile(r'<[^>]+>')

POSITIVE = [
    "打工","日结","工资","薪资","就业","失业","招工","招聘","裁员","职场","劳动","工伤","社保","维权","赔偿",
    "房租","租房","买房","房贷","物业","住房","养老","老人","医保","医疗","医院","看病","学校","学生","教师","教育",
    "消费","价格","外卖","快递","食品","隐私","平台","诈骗","被骗","救助","流浪","生存","摆摊","低保","婚姻","离婚",
    "彩礼","事故","安全","未成年","儿童","女性","家暴","AI","人工智能","算法","网约车","骑手","保洁","保安","农民工",
]
NEGATIVE = [
    "皇马","巴萨","NBA","英超","欧冠","足球","篮球","球员","球队","电竞","游戏","王者荣耀","英雄联盟","LOL",
    "明星","偶像","演唱会","电影","电视剧","综艺","票房","粉丝","战队","比赛","夺冠","比分","转会",
]


def fetch(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    for enc in ("utf-8", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", "replace")


def latest_snapshots(limit_days: int) -> list[tuple[str, str, str]]:
    archive = html.unescape(fetch(ARCHIVE))
    by_day: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for href, ds, tm in SNAP_RE.findall(archive):
        by_day[ds].append((tm, href))
    out = []
    for ds in sorted(by_day, reverse=True)[:limit_days]:
        tm, href = max(by_day[ds], key=lambda x: x[0])
        url = href if href.startswith("http") else RAW_BASE + href.lstrip("./")
        out.append((ds, tm, url))
    return out


def canonical_hot_url(raw: str) -> str:
    return html.unescape(raw).replace("&amp%3B", "&").replace("&amp;", "&")


def topic_meta(url: str) -> tuple[str, str]:
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    topic_id = (qs.get("topic_id") or [""])[0]
    title = urllib.parse.unquote((qs.get("topic_name") or [""])[0]).strip()
    return topic_id, title


def clean_text(value: str) -> str:
    value = html.unescape(TAG_RE.sub("", value))
    return re.sub(r"\s+", " ", value).strip()


def extract_topics(snapshot: str) -> list[tuple[str, str, str]]:
    decoded = html.unescape(snapshot).replace("&amp%3B", "&")
    out, seen = [], set()
    for m in HOT_RE.finditer(decoded):
        url = canonical_hot_url(m.group(0))
        topic_id, title = topic_meta(url)
        key = topic_id or url
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((topic_id, title, url))
    return out


def representative_threads(url: str, limit: int) -> list[dict]:
    try:
        page = fetch(url)
    except Exception as exc:
        return [{"error": str(exc)}]
    rows, seen = [], set()
    for tid, raw_title in THREAD_RE.findall(page):
        if tid in seen:
            continue
        seen.add(tid)
        title = clean_text(raw_title)
        if not title:
            continue
        rows.append({"tid": tid, "title": title, "url": f"https://tieba.baidu.com/p/{tid}"})
        if len(rows) >= limit:
            break
    return rows


def keyword_baseline(title: str, rep_titles: list[str]) -> tuple[int, float, list[str]]:
    text = " ".join([title, *rep_titles]).lower()
    pos = [k for k in POSITIVE if k.lower() in text]
    neg = [k for k in NEGATIVE if k.lower() in text]
    raw = len(pos) - 1.4 * len(neg)
    if raw >= 3:
        priority = 3
    elif raw >= 1:
        priority = 2
    elif raw > -1:
        priority = 1
    else:
        priority = 0
    confidence = min(0.95, 0.50 + abs(raw) * 0.08)
    return priority, round(confidence, 3), pos + [f"-{x}" for x in neg]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=100)
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--threads-per-topic", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.15)
    ap.add_argument("--out-dir", default="benchmark_out")
    args = ap.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    scanned = []

    for ds, tm, snap_url in latest_snapshots(args.days):
        try:
            snapshot = fetch(snap_url)
        except Exception as exc:
            print(f"WARN snapshot failed {snap_url}: {exc}")
            continue
        topics = extract_topics(snapshot)
        scanned.append({"date": ds, "snapshot_time": tm, "url": snap_url, "tieba_topics": len(topics)})
        print(ds, tm, "tieba_topics", len(topics), "unique_before", len(rows))
        for topic_id, title, hot_url in topics:
            key = topic_id or hot_url
            if key in seen:
                continue
            seen.add(key)
            reps = representative_threads(hot_url, args.threads_per_topic)
            rep_titles = [x.get("title", "") for x in reps if x.get("title")]
            if not title:
                title = rep_titles[0] if rep_titles else f"Tieba topic {topic_id}"
            rule_priority, rule_conf, rule_hits = keyword_baseline(title, rep_titles)
            rows.append({
                "benchmark_id": f"tb-{len(rows)+1:03d}",
                "data_date": ds,
                "snapshot_time": tm.replace("-", ":"),
                "snapshot_url": snap_url,
                "topic_id": topic_id,
                "title": title,
                "hottopic_url": hot_url,
                "representative_threads": reps,
                "representative_thread_titles": rep_titles,
                "rule_priority": rule_priority,
                "rule_confidence": rule_conf,
                "rule_hits": rule_hits,
                "gold_priority": "",
                "gold_category": "",
                "gold_social_relevance": "",
                "gold_first_person_value": "",
                "gold_research_value": "",
                "gold_noise": "",
                "gold_note": "",
            })
            if len(rows) >= args.target:
                break
            time.sleep(args.sleep)
        if len(rows) >= args.target:
            break

    if len(rows) < args.target:
        raise SystemExit(f"need {args.target} unique topics, only got {len(rows)}")

    with (outdir / "tieba_priority_100.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fields = [
        "benchmark_id","data_date","snapshot_time","topic_id","title","representative_thread_titles",
        "rule_priority","rule_confidence","rule_hits","gold_priority","gold_category","gold_social_relevance",
        "gold_first_person_value","gold_research_value","gold_noise","gold_note","hottopic_url","snapshot_url",
    ]
    with (outdir / "tieba_priority_100.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            flat = dict(row)
            flat["representative_thread_titles"] = " | ".join(row["representative_thread_titles"])
            flat["rule_hits"] = " | ".join(row["rule_hits"])
            w.writerow({k: flat.get(k, "") for k in fields})

    manifest = {
        "source": "Hot News Radar reports branch only",
        "sample_size": len(rows),
        "snapshot_policy": "latest snapshot per day only",
        "dedupe": "topic_id",
        "context": "hotspot title + up to 3 representative Tieba thread titles",
        "gold_priority_scale": {"0": "ignore", "1": "index only", "2": "sample crawl", "3": "deep crawl"},
        "gold_dimension_scale": "0-3",
        "snapshots_scanned": scanned,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"sample_size": len(rows), "snapshots_used": len(scanned)}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
