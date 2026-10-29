#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone Tieba hotspot collector (v2 test build).

Public-data research collector supporting:
1) a Tieba thread URL / tid
2) a Tieba hottopic page
3) a Hot News Radar HTML snapshot containing Tieba hottopic links

Outputs SQLite + JSONL. Standard library only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import random
import re
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

CLIENT_VERSION = "9.7.8.0"
CLIENT_ID = "wappc_1534235498291_633"
CLIENT_TYPE = "2"
PHONE_IMEI = "000000000000000"
API_BASE = "http://c.tieba.baidu.com"
UA_API = f"bdtb for Android {CLIENT_VERSION}"
UA_WEB = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140 Safari/537.36"

THREAD_RE = re.compile(r"(?:https?://(?:c\.)?tieba\.baidu\.com)?/p/(\d+)", re.I)
HOT_TOPIC_RE = re.compile(r"https?://(?:c\.)?tieba\.baidu\.com/hottopic/browse/hottopic\?[^\"'<>\s]+", re.I)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def to_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def api_error_code(payload: dict) -> int:
    """Return Tieba error code without treating integer 0 as falsy/error.

    The old collector used `x or -1`, which converted a valid 0 into -1.
    """
    if "error_code" not in payload:
        # Some successful variants omit error_code but contain normal data.
        if payload.get("thread") or payload.get("post_list") or payload.get("subpost_list") is not None:
            return 0
        return -1
    return to_int(payload.get("error_code"), -1)


def sign_params(params: Dict[str, str]) -> str:
    raw = "".join(f"{k}={params[k]}" for k in sorted(params)) + "tiebaclient!!!"
    return hashlib.md5(raw.encode("utf-8")).hexdigest().upper()


def request_bytes(
    url: str,
    *,
    data: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 20,
    retries: int = 3,
    base_delay: float = 1.0,
) -> bytes:
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # network boundary
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(base_delay * (2**attempt) + random.uniform(0, 0.35))
    raise RuntimeError(f"request failed: {url}: {last_error}")


def request_text(url: str, *, headers: Optional[Dict[str, str]] = None) -> str:
    raw = request_bytes(url, headers=headers)
    for encoding in ("utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def api_request(endpoint: str, params: Dict[str, object]) -> dict:
    payload = {
        "_client_id": CLIENT_ID,
        "_client_type": CLIENT_TYPE,
        "_client_version": CLIENT_VERSION,
        "_phone_imei": PHONE_IMEI,
    }
    payload.update({k: str(v) for k, v in params.items()})
    payload["sign"] = sign_params(payload)
    body = urllib.parse.urlencode(payload).encode("utf-8")
    raw = request_bytes(
        API_BASE + endpoint,
        data=body,
        headers={
            "User-Agent": UA_API,
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
    )
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError as exc:
        preview = raw[:200].decode("utf-8", "replace")
        raise RuntimeError(f"Tieba API returned non-JSON: {preview!r}") from exc


def extract_tid(value: str) -> str:
    value = value.strip()
    if value.isdigit():
        return value
    match = THREAD_RE.search(value)
    if not match:
        raise ValueError(f"cannot extract Tieba tid from {value!r}")
    return match.group(1)


def extract_tids(text: str) -> List[str]:
    decoded = html.unescape(text)
    return sorted(set(m.group(1) for m in THREAD_RE.finditer(decoded)), key=int)


def parse_content(items: Iterable[dict]) -> Tuple[str, List[str]]:
    parts: List[str] = []
    images: List[str] = []
    for item in items or []:
        item_type = to_int(item.get("type"), 0)
        if item_type == 0:
            parts.append(str(item.get("text") or ""))
        elif item_type == 1:
            label = str(item.get("text") or "")
            link = str(item.get("link") or "")
            parts.append(f"{label} ({link})" if link else label)
        elif item_type == 2:
            parts.append(str(item.get("text") or item.get("c") or "[表情]"))
        elif item_type == 3:
            image_url = str(item.get("origin_src") or item.get("big_cdn_src") or item.get("cdn_src") or "")
            if image_url:
                images.append(image_url)
        elif item_type == 4:
            parts.append("@" + str(item.get("text") or ""))
        elif item_type == 9:
            parts.append("[视频]")
        elif item_type == 10:
            parts.append("#" + str(item.get("text") or "") + "#")
        elif item.get("text"):
            parts.append(str(item["text"]))
    return "".join(parts).strip(), images


class Store:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS threads (
              tid TEXT PRIMARY KEY,
              title TEXT, forum TEXT, author TEXT,
              create_time INTEGER, reply_num INTEGER,
              agree_num INTEGER, collect_num INTEGER,
              url TEXT, source_url TEXT,
              first_seen TEXT, last_seen TEXT
            );
            CREATE TABLE IF NOT EXISTS posts (
              pid TEXT PRIMARY KEY,
              tid TEXT NOT NULL, floor INTEGER, author TEXT,
              create_time INTEGER, content TEXT,
              agree_num INTEGER, subpost_num INTEGER,
              images_json TEXT, last_seen TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_posts_tid_floor ON posts(tid, floor);
            CREATE TABLE IF NOT EXISTS subposts (
              spid TEXT PRIMARY KEY,
              tid TEXT NOT NULL, pid TEXT NOT NULL,
              author TEXT, create_time INTEGER, content TEXT,
              images_json TEXT, last_seen TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_subposts_pid ON subposts(pid);
            CREATE TABLE IF NOT EXISTS discoveries (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              source_url TEXT, source_type TEXT, tid TEXT,
              discovered_at TEXT,
              UNIQUE(source_url, tid)
            );
            CREATE TABLE IF NOT EXISTS failures (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              source_url TEXT, tid TEXT, stage TEXT,
              error TEXT, created_at TEXT
            );
            """
        )
        self.conn.commit()

    def upsert_thread(self, row: dict) -> None:
        now = utc_now()
        self.conn.execute(
            """
            INSERT INTO threads(tid,title,forum,author,create_time,reply_num,agree_num,collect_num,url,source_url,first_seen,last_seen)
            VALUES(:tid,:title,:forum,:author,:create_time,:reply_num,:agree_num,:collect_num,:url,:source_url,:first_seen,:last_seen)
            ON CONFLICT(tid) DO UPDATE SET
              title=excluded.title, forum=excluded.forum, author=excluded.author,
              create_time=excluded.create_time, reply_num=excluded.reply_num,
              agree_num=excluded.agree_num, collect_num=excluded.collect_num,
              source_url=COALESCE(excluded.source_url,threads.source_url),
              last_seen=excluded.last_seen
            """,
            {**row, "first_seen": now, "last_seen": now},
        )

    def upsert_post(self, row: dict) -> None:
        self.conn.execute(
            """
            INSERT INTO posts(pid,tid,floor,author,create_time,content,agree_num,subpost_num,images_json,last_seen)
            VALUES(:pid,:tid,:floor,:author,:create_time,:content,:agree_num,:subpost_num,:images_json,:last_seen)
            ON CONFLICT(pid) DO UPDATE SET
              content=excluded.content, agree_num=excluded.agree_num,
              subpost_num=excluded.subpost_num, images_json=excluded.images_json,
              last_seen=excluded.last_seen
            """,
            {**row, "last_seen": utc_now()},
        )

    def upsert_subpost(self, row: dict) -> None:
        self.conn.execute(
            """
            INSERT INTO subposts(spid,tid,pid,author,create_time,content,images_json,last_seen)
            VALUES(:spid,:tid,:pid,:author,:create_time,:content,:images_json,:last_seen)
            ON CONFLICT(spid) DO UPDATE SET
              content=excluded.content, images_json=excluded.images_json,
              last_seen=excluded.last_seen
            """,
            {**row, "last_seen": utc_now()},
        )

    def discovery(self, source_url: str, source_type: str, tid: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO discoveries(source_url,source_type,tid,discovered_at) VALUES(?,?,?,?)",
            (source_url, source_type, tid, utc_now()),
        )

    def failure(self, source_url: str, tid: str, stage: str, exc: Exception) -> None:
        self.conn.execute(
            "INSERT INTO failures(source_url,tid,stage,error,created_at) VALUES(?,?,?,?,?)",
            (source_url, tid, stage, str(exc), utc_now()),
        )
        self.conn.commit()

    def commit(self) -> None:
        self.conn.commit()


def fetch_thread_page(tid: str, page: int, page_size: int = 30) -> dict:
    return api_request("/c/f/pb/page", {"kz": tid, "pn": page, "rn": page_size, "r": 0, "lz": 0, "st": 0, "z": 0})


def fetch_subposts(tid: str, pid: str, expected: int, sleep_seconds: float) -> List[dict]:
    rows: List[dict] = []
    page = 1
    while True:
        payload = api_request("/c/f/pb/floor", {"kz": tid, "pid": pid, "pn": page, "rn": 20})
        if api_error_code(payload) != 0:
            break
        batch = payload.get("subpost_list") or []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 20 or (expected and len(rows) >= expected):
            break
        page += 1
        time.sleep(sleep_seconds + random.uniform(0, 0.25))
    return rows


def crawl_thread(
    store: Store,
    tid: str,
    *,
    source_url: str,
    max_pages: int,
    sleep_seconds: float,
    include_subposts: bool,
) -> dict:
    first = fetch_thread_page(tid, 1)
    code = api_error_code(first)
    if code != 0:
        raise RuntimeError(f"Tieba API error {code}: {first.get('error_msg') or first.get('error') or ''}")

    thread = first.get("thread") or {}
    page_info = first.get("page") or {}
    total_pages = max(1, to_int(page_info.get("total_page"), 1))
    if max_pages > 0:
        total_pages = min(total_pages, max_pages)

    author = thread.get("author") or {}
    forum = str(
        thread.get("fname")
        or (thread.get("twzhibo_info") or {}).get("forum_name")
        or (first.get("display_forum") or {}).get("name")
        or ""
    )
    store.upsert_thread(
        {
            "tid": tid,
            "title": str(thread.get("title") or ""),
            "forum": forum,
            "author": str(author.get("name_show") or author.get("name") or ""),
            "create_time": to_int(thread.get("create_time")),
            "reply_num": to_int(thread.get("reply_num")),
            "agree_num": to_int(thread.get("agree_num")),
            "collect_num": to_int(thread.get("collect_num")),
            "url": f"https://tieba.baidu.com/p/{tid}",
            "source_url": source_url,
        }
    )

    post_count = 0
    subpost_count = 0
    users: Dict[str, str] = {}

    for page_number in range(1, total_pages + 1):
        payload = first if page_number == 1 else fetch_thread_page(tid, page_number)
        if api_error_code(payload) != 0:
            continue

        for user in payload.get("user_list") or []:
            uid = str(user.get("id") or "")
            if uid:
                users[uid] = str(user.get("name_show") or user.get("name") or f"用户{uid}")

        for post in payload.get("post_list") or []:
            pid = str(post.get("id") or "")
            if not pid:
                continue
            author_id = str(post.get("author_id") or "")
            text, images = parse_content(post.get("content") or [])
            subpost_num = to_int(post.get("sub_post_number"))
            store.upsert_post(
                {
                    "pid": pid,
                    "tid": tid,
                    "floor": to_int(post.get("floor")),
                    "author": users.get(author_id, f"用户{author_id}" if author_id else ""),
                    "create_time": to_int(post.get("time")),
                    "content": text,
                    "agree_num": to_int((post.get("agree") or {}).get("agree_num")),
                    "subpost_num": subpost_num,
                    "images_json": json.dumps(images, ensure_ascii=False),
                }
            )
            post_count += 1

            if include_subposts and subpost_num > 0:
                for subpost in fetch_subposts(tid, pid, subpost_num, sleep_seconds):
                    spid = str(subpost.get("id") or subpost.get("spid") or "")
                    if not spid:
                        continue
                    sub_author = subpost.get("author") or {}
                    sub_text, sub_images = parse_content(subpost.get("content") or [])
                    store.upsert_subpost(
                        {
                            "spid": spid,
                            "tid": tid,
                            "pid": pid,
                            "author": str(sub_author.get("name_show") or sub_author.get("name") or ""),
                            "create_time": to_int(subpost.get("time")),
                            "content": sub_text,
                            "images_json": json.dumps(sub_images, ensure_ascii=False),
                        }
                    )
                    subpost_count += 1
                time.sleep(sleep_seconds + random.uniform(0, 0.25))

        store.commit()
        if page_number < total_pages:
            time.sleep(sleep_seconds + random.uniform(0, 0.35))

    return {
        "tid": tid,
        "title": str(thread.get("title") or ""),
        "forum": forum,
        "pages": total_pages,
        "posts": post_count,
        "subposts": subpost_count,
    }


def web_headers(cookie: str = "") -> Dict[str, str]:
    headers = {"User-Agent": UA_WEB, "Referer": "https://tieba.baidu.com/"}
    if cookie:
        headers["Cookie"] = cookie
    return headers


def discover_from_hottopic(url: str, cookie: str = "", max_threads: int = 0) -> List[str]:
    text = request_text(url, headers=web_headers(cookie))
    tids = extract_tids(text)
    if max_threads > 0:
        tids = tids[:max_threads]
    return tids


def discover_from_snapshot(
    url: str,
    *,
    cookie: str,
    max_hot_topics: int,
    max_threads_per_topic: int,
) -> List[Tuple[str, str]]:
    snapshot = request_text(url, headers={"User-Agent": UA_WEB})
    decoded = html.unescape(snapshot).replace("&amp%3B", "&")
    hot_urls: List[str] = []
    seen_urls = set()
    for match in HOT_TOPIC_RE.finditer(decoded):
        hot_url = match.group(0).replace("&amp;", "&")
        if hot_url not in seen_urls:
            seen_urls.add(hot_url)
            hot_urls.append(hot_url)
    if max_hot_topics > 0:
        hot_urls = hot_urls[:max_hot_topics]

    jobs: List[Tuple[str, str]] = []
    seen_tids = set()
    for hot_url in hot_urls:
        try:
            tids = discover_from_hottopic(hot_url, cookie, max_threads_per_topic)
        except Exception as exc:
            print(f"WARN hottopic discovery failed: {hot_url}: {exc}", file=sys.stderr)
            continue
        for tid in tids:
            if tid not in seen_tids:
                seen_tids.add(tid)
                jobs.append((hot_url, tid))
    return jobs


def export_jsonl(db_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    for table in ("threads", "posts", "subposts", "discoveries", "failures"):
        with (out_dir / f"{table}.jsonl").open("w", encoding="utf-8") as handle:
            for row in conn.execute(f"SELECT * FROM {table}"):
                handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect public Tieba hotspot discussion data")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--thread", help="Tieba /p/ URL or numeric tid")
    source.add_argument("--hot-topic", help="Tieba hottopic URL")
    source.add_argument("--snapshot", help="Hot News Radar HTML snapshot URL")
    parser.add_argument("--db", default="data/tieba.sqlite3")
    parser.add_argument("--export-dir", default="data/export")
    parser.add_argument("--sleep", type=float, default=1.2)
    parser.add_argument("--max-pages", type=int, default=0, help="0 means all pages")
    parser.add_argument("--max-hot-topics", type=int, default=0)
    parser.add_argument("--max-threads-per-topic", type=int, default=0)
    parser.add_argument("--no-subposts", action="store_true")
    parser.add_argument("--cookie", default="")
    parser.add_argument("--min-success", type=int, default=0, help="exit non-zero unless at least N threads succeed")
    args = parser.parse_args()

    db_path = Path(args.db)
    store = Store(db_path)
    jobs: List[Tuple[str, str, str]] = []

    try:
        if args.thread:
            tid = extract_tid(args.thread)
            jobs.append(("thread", args.thread, tid))
        elif args.hot_topic:
            tids = discover_from_hottopic(args.hot_topic, args.cookie, args.max_threads_per_topic)
            print(f"discovered {len(tids)} thread(s) from hottopic", file=sys.stderr)
            jobs.extend(("hot_topic", args.hot_topic, tid) for tid in tids)
        elif args.snapshot:
            discovered = discover_from_snapshot(
                args.snapshot,
                cookie=args.cookie,
                max_hot_topics=args.max_hot_topics,
                max_threads_per_topic=args.max_threads_per_topic,
            )
            print(f"discovered {len(discovered)} unique thread(s) from snapshot", file=sys.stderr)
            jobs.extend(("hot_news_radar", hot_url, tid) for hot_url, tid in discovered)
    except Exception as exc:
        store.failure(args.thread or args.hot_topic or args.snapshot or "", "", "discover", exc)
        print(f"DISCOVERY FAILED: {exc}", file=sys.stderr)

    unique_jobs: List[Tuple[str, str, str]] = []
    seen = set()
    for job in jobs:
        if job[2] in seen:
            continue
        seen.add(job[2])
        unique_jobs.append(job)
        store.discovery(job[1], job[0], job[2])
    store.commit()

    success = 0
    stats: List[dict] = []
    for index, (_, source_url, tid) in enumerate(unique_jobs, start=1):
        print(f"[{index}/{len(unique_jobs)}] crawl tid={tid}", file=sys.stderr)
        try:
            stat = crawl_thread(
                store,
                tid,
                source_url=source_url,
                max_pages=args.max_pages,
                sleep_seconds=args.sleep,
                include_subposts=not args.no_subposts,
            )
            stats.append(stat)
            success += 1
            print(json.dumps(stat, ensure_ascii=False))
        except Exception as exc:
            store.failure(source_url, tid, "crawl", exc)
            print(f"FAILED tid={tid}: {exc}", file=sys.stderr)
        if index < len(unique_jobs):
            time.sleep(args.sleep + random.uniform(0, 0.35))

    export_jsonl(db_path, Path(args.export_dir))
    summary = {
        "jobs": len(unique_jobs),
        "success": success,
        "failed": len(unique_jobs) - success,
        "db": str(db_path),
        "export_dir": args.export_dir,
        "threads": stats,
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), file=sys.stderr)

    if success < args.min_success:
        print(f"minimum success requirement not met: {success} < {args.min_success}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
