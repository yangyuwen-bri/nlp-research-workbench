#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tieba collector for public research data.

Inputs:
- single Tieba thread URL / tid
- Tieba hot-topic page
- Hot News Radar snapshot containing Tieba links

Storage:
- SQLite (incremental, deduplicated)
- JSONL exports for downstream analysis

Uses public page HTML plus the Tieba Android client API for thread floors/subposts.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import html
import json
import os
import random
import re
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CLIENT_VERSION = "9.7.8.0"
CLIENT_ID = "wappc_1534235498291_633"
CLIENT_TYPE = "2"
PHONE_IMEI = "000000000000000"
API_BASE = "http://c.tieba.baidu.com"
UA_API = f"bdtb for Android {CLIENT_VERSION}"
UA_WEB = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/140 Safari/537.36"

THREAD_RE = re.compile(r"(?:https?://(?:c\.)?tieba\.baidu\.com)?/p/(\d+)", re.I)
HOT_TOPIC_URL_RE = re.compile(r"https?://(?:c\.)?tieba\.baidu\.com/hottopic/browse/hottopic\?[^\"'<>\s]+", re.I)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sign_params(params: Dict[str, str]) -> str:
    raw = "".join(k + "=" + str(params[k]) for k in sorted(params)) + "tiebaclient!!!"
    return hashlib.md5(raw.encode("utf-8")).hexdigest().upper()


def request_bytes(url: str, *, data: Optional[bytes] = None, headers: Optional[Dict[str,str]] = None,
                  timeout: int = 20, retries: int = 3, delay: float = 1.0) -> bytes:
    headers = headers or {}
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            if attempt + 1 < retries:
                time.sleep(delay * (2 ** attempt) + random.uniform(0, 0.4))
    raise RuntimeError(f"request failed: {url}: {last}")


def request_text(url: str, **kw) -> str:
    b = request_bytes(url, **kw)
    for enc in ("utf-8", "gb18030"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            pass
    return b.decode("utf-8", "replace")


def api_request(endpoint: str, params: Dict[str, str], timeout: int = 20, retries: int = 3) -> dict:
    base = {
        "_client_id": CLIENT_ID,
        "_client_type": CLIENT_TYPE,
        "_client_version": CLIENT_VERSION,
        "_phone_imei": PHONE_IMEI,
    }
    base.update({k: str(v) for k, v in params.items()})
    base["sign"] = sign_params(base)
    body = urllib.parse.urlencode(base).encode("utf-8")
    raw = request_bytes(
        API_BASE + endpoint,
        data=body,
        headers={
            "User-Agent": UA_API,
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        timeout=timeout,
        retries=retries,
    )
    return json.loads(raw.decode("utf-8", "replace"))


def extract_tid(value: str) -> str:
    value = value.strip()
    if value.isdigit():
        return value
    m = THREAD_RE.search(value)
    if not m:
        raise ValueError(f"cannot extract Tieba thread id from: {value}")
    return m.group(1)


def parse_content(items: list) -> Tuple[str, List[str]]:
    text, images = [], []
    for item in items or []:
        t = int(item.get("type", 0) or 0)
        if t == 0:
            text.append(item.get("text", ""))
        elif t == 1:
            label = item.get("text", "")
            url = item.get("link", "")
            text.append(f"{label} ({url})" if url else label)
        elif t == 2:
            text.append(item.get("text") or item.get("c") or "[表情]")
        elif t == 3:
            url = item.get("origin_src") or item.get("big_cdn_src") or item.get("cdn_src") or ""
            if url:
                images.append(url)
        elif t == 4:
            text.append("@" + (item.get("text") or ""))
        elif t == 9:
            text.append("[视频]")
        elif t == 10:
            text.append("#" + (item.get("text") or "") + "#")
        elif item.get("text"):
            text.append(item["text"])
    return "".join(text).strip(), images


class DB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init()

    def _init(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS threads (
          tid TEXT PRIMARY KEY,
          title TEXT, forum TEXT, author TEXT,
          create_time INTEGER, reply_num INTEGER, agree_num INTEGER, collect_num INTEGER,
          url TEXT, source_url TEXT, first_seen TEXT, last_seen TEXT
        );
        CREATE TABLE IF NOT EXISTS posts (
          pid TEXT PRIMARY KEY,
          tid TEXT NOT NULL,
          floor INTEGER, author TEXT, create_time INTEGER,
          content TEXT, agree_num INTEGER, subpost_num INTEGER,
          images_json TEXT, last_seen TEXT,
          FOREIGN KEY(tid) REFERENCES threads(tid)
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
          source_url TEXT, source_type TEXT, tid TEXT, discovered_at TEXT,
          UNIQUE(source_url, tid)
        );
        CREATE TABLE IF NOT EXISTS failures (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          source_url TEXT, tid TEXT, stage TEXT, error TEXT, created_at TEXT
        );
        """)
        self.conn.commit()

    def upsert_thread(self, row: dict):
        ts = now_iso()
        self.conn.execute("""
        INSERT INTO threads(tid,title,forum,author,create_time,reply_num,agree_num,collect_num,url,source_url,first_seen,last_seen)
        VALUES(:tid,:title,:forum,:author,:create_time,:reply_num,:agree_num,:collect_num,:url,:source_url,:first_seen,:last_seen)
        ON CONFLICT(tid) DO UPDATE SET
          title=excluded.title, forum=excluded.forum, author=excluded.author,
          create_time=excluded.create_time, reply_num=excluded.reply_num,
          agree_num=excluded.agree_num, collect_num=excluded.collect_num,
          source_url=COALESCE(excluded.source_url,threads.source_url), last_seen=excluded.last_seen
        """, {**row, "first_seen": ts, "last_seen": ts})

    def upsert_post(self, row: dict):
        self.conn.execute("""
        INSERT INTO posts(pid,tid,floor,author,create_time,content,agree_num,subpost_num,images_json,last_seen)
        VALUES(:pid,:tid,:floor,:author,:create_time,:content,:agree_num,:subpost_num,:images_json,:last_seen)
        ON CONFLICT(pid) DO UPDATE SET
          content=excluded.content, agree_num=excluded.agree_num,
          subpost_num=excluded.subpost_num, images_json=excluded.images_json,
          last_seen=excluded.last_seen
        """, {**row, "last_seen": now_iso()})

    def upsert_subpost(self, row: dict):
        self.conn.execute("""
        INSERT INTO subposts(spid,tid,pid,author,create_time,content,images_json,last_seen)
        VALUES(:spid,:tid,:pid,:author,:create_time,:content,:images_json,:last_seen)
        ON CONFLICT(spid) DO UPDATE SET content=excluded.content, images_json=excluded.images_json,last_seen=excluded.last_seen
        """, {**row, "last_seen": now_iso()})

    def discover(self, source_url: str, source_type: str, tid: str):
        self.conn.execute("INSERT OR IGNORE INTO discoveries(source_url,source_type,tid,discovered_at) VALUES(?,?,?,?)",
                          (source_url, source_type, tid, now_iso()))

    def fail(self, source_url: str, tid: str, stage: str, error: Exception):
        self.conn.execute("INSERT INTO failures(source_url,tid,stage,error,created_at) VALUES(?,?,?,?,?)",
                          (source_url, tid, stage, str(error), now_iso()))
        self.conn.commit()

    def commit(self):
        self.conn.commit()


def fetch_thread_page(tid: str, pn: int, rn: int = 30) -> dict:
    return api_request("/c/f/pb/page", {"kz":tid,"pn":pn,"rn":rn,"r":0,"lz":0,"st":0,"z":0})


def fetch_subposts(tid: str, pid: str, total_hint: int, sleep_s: float) -> List[dict]:
    out, pn = [], 1
    while True:
        data = api_request("/c/f/pb/floor", {"kz":tid,"pid":pid,"pn":pn,"rn":20})
        if int(data.get("error_code", -1) or -1) != 0:
            break
        batch = data.get("subpost_list") or []
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 20 or (total_hint and len(out) >= total_hint):
            break
        pn += 1
        time.sleep(sleep_s + random.uniform(0, 0.3))
    return out


def crawl_thread(db: DB, tid: str, source_url: str = "", sleep_s: float = 1.0, max_pages: int = 0,
                 include_subposts: bool = True) -> dict:
    first = fetch_thread_page(tid, 1)
    if int(first.get("error_code", -1) or -1) != 0:
        raise RuntimeError(f"Tieba API error {first.get('error_code')}: {first.get('error_msg','')}")
    th = first.get("thread") or {}
    page = first.get("page") or {}
    total_pages = int(page.get("total_page", 1) or 1)
    if max_pages:
        total_pages = min(total_pages, max_pages)
    users = {}
    thread_author = th.get("author") or {}
    forum = th.get("fname") or (th.get("twzhibo_info") or {}).get("forum_name") or (first.get("display_forum") or {}).get("name") or ""
    db.upsert_thread({
        "tid":tid, "title":th.get("title") or "", "forum":forum,
        "author":thread_author.get("name_show") or thread_author.get("name") or "",
        "create_time":int(th.get("create_time",0) or 0), "reply_num":int(th.get("reply_num",0) or 0),
        "agree_num":int(th.get("agree_num",0) or 0), "collect_num":int(th.get("collect_num",0) or 0),
        "url":f"https://tieba.baidu.com/p/{tid}", "source_url":source_url,
    })
    posts_count = subs_count = 0
    for pn in range(1, total_pages + 1):
        data = first if pn == 1 else fetch_thread_page(tid, pn)
        if int(data.get("error_code", -1) or -1) != 0:
            continue
        for u in data.get("user_list") or []:
            uid = str(u.get("id") or "")
            if uid:
                users[uid] = u.get("name_show") or u.get("name") or f"用户{uid}"
        for p in data.get("post_list") or []:
            pid = str(p.get("id") or "")
            if not pid:
                continue
            author_id = str(p.get("author_id") or "")
            content, imgs = parse_content(p.get("content") or [])
            subn = int(p.get("sub_post_number",0) or 0)
            db.upsert_post({
                "pid":pid, "tid":tid, "floor":int(p.get("floor",0) or 0),
                "author":users.get(author_id, f"用户{author_id}" if author_id else ""),
                "create_time":int(p.get("time",0) or 0), "content":content,
                "agree_num":int((p.get("agree") or {}).get("agree_num",0) or 0), "subpost_num":subn,
                "images_json":json.dumps(imgs, ensure_ascii=False),
            })
            posts_count += 1
            if include_subposts and subn > 0:
                for sp in fetch_subposts(tid, pid, subn, sleep_s):
                    spid = str(sp.get("id") or sp.get("spid") or "")
                    if not spid:
                        continue
                    au = sp.get("author") or {}
                    txt, simgs = parse_content(sp.get("content") or [])
                    db.upsert_subpost({
                        "spid":spid,"tid":tid,"pid":pid,
                        "author":au.get("name_show") or au.get("name") or "",
                        "create_time":int(sp.get("time",0) or 0),"content":txt,
                        "images_json":json.dumps(simgs,ensure_ascii=False),
                    })
                    subs_count += 1
                time.sleep(sleep_s + random.uniform(0,0.3))
        db.commit()
        if pn < total_pages:
            time.sleep(sleep_s + random.uniform(0,0.4))
    return {"tid":tid,"title":th.get("title") or "","forum":forum,"pages":total_pages,"posts":posts_count,"subposts":subs_count}


def discover_tids_from_html(text: str) -> List[str]:
    text = html.unescape(text)
    return sorted(set(m.group(1) for m in THREAD_RE.finditer(text)), key=int)


def discover_from_url(url: str, cookie: str = "", max_hot_topics: int = 0, max_threads_per_topic: int = 0) -> Tuple[str,List[str]]:
    headers = {"User-Agent":UA_WEB, "Referer":"https://tieba.baidu.com/"}
    if cookie:
        headers["Cookie"] = cookie
    text = request_text(url, headers=headers)
    tids = discover_tids_from_html(text)
    source_type = "hot_topic" if "hottopic" in url else ("hot_news_radar" if "hot-news-radar" in url or "raw.githubusercontent.com" in url else "html")

    if source_type == "hot_news_radar":
        decoded = html.unescape(text).replace("&amp%3B", "&")
        hot_urls = []
        seen = set()
        for m in HOT_TOPIC_URL_RE.finditer(decoded):
            u = m.group(0).replace("&amp;", "&")
            if u not in seen:
                seen.add(u)
                hot_urls.append(u)
        if max_hot_topics:
            hot_urls = hot_urls[:max_hot_topics]
        for hu in hot_urls:
            try:
                htext = request_text(hu, headers=headers)
                htids = discover_tids_from_html(htext)
                if max_threads_per_topic:
                    htids = htids[:max_threads_per_topic]
                tids.extend(htids)
            except Exception as e:
                print(f"WARN hot-topic discovery failed: {hu}: {e}", file=sys.stderr)
    if source_type == "hot_topic" and max_threads_per_topic:
        tids = tids[:max_threads_per_topic]
    return source_type, sorted(set(tids), key=int)


def export_jsonl(db_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    for table in ("threads","posts","subposts","failures"):
        with (out_dir / f"{table}.jsonl").open("w",encoding="utf-8") as f:
            for row in conn.execute(f"SELECT * FROM {table}"):
                f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Collect public Baidu Tieba thread/floor/subpost data")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--thread", help="Tieba thread URL or numeric tid")
    src.add_argument("--hot-topic", help="Tieba hottopic URL")
    src.add_argument("--snapshot", help="Hot News Radar snapshot URL/HTML URL containing Tieba links")
    src.add_argument("--input-file", help="text file: one thread URL/tid per line")
    ap.add_argument("--db", default="data/tieba.sqlite3")
    ap.add_argument("--export-dir", default="data/export")
    ap.add_argument("--sleep", type=float, default=1.2, help="delay between requests")
    ap.add_argument("--max-pages", type=int, default=0, help="0=all pages")
    ap.add_argument("--max-hot-topics", type=int, default=0, help="snapshot mode: 0=all Tieba hot topics")
    ap.add_argument("--max-threads-per-topic", type=int, default=0, help="0=all discovered threads per hot topic")
    ap.add_argument("--no-subposts", action="store_true")
    ap.add_argument("--cookie", default=os.environ.get("TIEBA_COOKIE",""), help="optional cookie for HTML discovery only")
    args = ap.parse_args()

    db_path = Path(args.db)
    db = DB(db_path)
    sources: List[Tuple[str,str]] = []
    if args.thread:
        sources=[("thread",args.thread)]
    elif args.hot_topic:
        sources=[("hot_topic",args.hot_topic)]
    elif args.snapshot:
        sources=[("hot_news_radar",args.snapshot)]
    else:
        lines=[x.strip() for x in Path(args.input_file).read_text(encoding="utf-8").splitlines() if x.strip() and not x.lstrip().startswith("#")]
        sources=[("file",x) for x in lines]

    jobs: List[Tuple[str,str,str]] = []
    for stype, value in sources:
        try:
            if stype in ("thread","file"):
                tid=extract_tid(value)
                jobs.append((stype,value,tid))
                db.discover(value,stype,tid)
            else:
                real_type,tids=discover_from_url(value,args.cookie,args.max_hot_topics,args.max_threads_per_topic)
                print(f"discovered {len(tids)} thread ids from {value}", file=sys.stderr)
                for tid in tids:
                    jobs.append((real_type,value,tid))
                    db.discover(value,real_type,tid)
                db.commit()
        except Exception as e:
            db.fail(value,"","discover",e)
            print(f"DISCOVERY FAILED: {value}: {e}", file=sys.stderr)

    seen=set()
    jobs=[j for j in jobs if not (j[2] in seen or seen.add(j[2]))]
    ok=0
    for i,(stype,source_url,tid) in enumerate(jobs,1):
        print(f"[{i}/{len(jobs)}] crawl tid={tid}", file=sys.stderr)
        try:
            stat=crawl_thread(db,tid,source_url,args.sleep,args.max_pages,not args.no_subposts)
            ok+=1
            print(json.dumps(stat,ensure_ascii=False))
        except Exception as e:
            db.fail(source_url,tid,"crawl",e)
            print(f"FAILED tid={tid}: {e}", file=sys.stderr)
        if i < len(jobs):
            time.sleep(args.sleep + random.uniform(0,0.5))

    export_jsonl(db_path,Path(args.export_dir))
    print(f"done: {ok}/{len(jobs)} threads succeeded; db={db_path}; export={args.export_dir}", file=sys.stderr)

if __name__ == "__main__":
    main()
