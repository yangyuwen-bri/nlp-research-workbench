import hashlib
import json
import re
import urllib.parse
import urllib.request

CLIENT_VERSION = "9.7.8.0"
UA = f"bdtb for Android {CLIENT_VERSION}"
WEB_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/140 Safari/537.36"


def api(endpoint, extra):
    params = {
        "_client_id": "wappc_1534235498291_633",
        "_client_type": "2",
        "_client_version": CLIENT_VERSION,
        "_phone_imei": "000000000000000",
        **{k: str(v) for k, v in extra.items()},
    }
    raw = "".join(k + "=" + str(params[k]) for k in sorted(params)) + "tiebaclient!!!"
    params["sign"] = hashlib.md5(raw.encode("utf-8")).hexdigest().upper()
    req = urllib.request.Request(
        "http://c.tieba.baidu.com" + endpoint,
        data=urllib.parse.urlencode(params).encode("utf-8"),
        headers={"User-Agent": UA, "Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8", "replace"))
    code = payload.get("error_code", -1)
    if int(code) != 0:
        raise RuntimeError(f"Tieba API error {code}: {payload.get('error_msg', '')}")
    return payload


def web_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": WEB_UA, "Referer": "https://tieba.baidu.com/", "Accept": "application/json,text/plain,*/*"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def collect_tids(value):
    tids = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"tid", "thread_id"} and str(item).isdigit():
                tids.add(str(item))
            tids.update(collect_tids(item))
    elif isinstance(value, list):
        for item in value:
            tids.update(collect_tids(item))
    elif isinstance(value, str):
        tids.update(re.findall(r"/p/(\d+)", value))
    return tids


def collect_paging_fields(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if any(token in key.lower() for token in ("last", "page", "next", "more", "cursor")) and isinstance(item, (str, int, float, bool, type(None))):
                out[path] = item
            out.update(collect_paging_fields(item, path))
    elif isinstance(value, list):
        for idx, item in enumerate(value[:3]):
            out.update(collect_paging_fields(item, f"{prefix}[{idx}]"))
    return out


# 1) thread/floor page
payload = api("/c/f/pb/page", {"kz": "11063972585", "pn": 1, "rn": 30, "r": 0, "lz": 0, "st": 0, "z": 0})
thread = payload.get("thread") or {}
posts = payload.get("post_list") or []
page = payload.get("page") or {}
print("THREAD", json.dumps({
    "error_code": payload.get("error_code"),
    "title": thread.get("title"),
    "forum": thread.get("fname") or (payload.get("display_forum") or {}).get("name"),
    "posts_on_page": len(posts),
    "total_page": page.get("total_page"),
    "total_num": page.get("total_num"),
}, ensure_ascii=False))
if not posts:
    raise RuntimeError("API returned success but no post_list")

# 2) subposts
candidate = next((p for p in posts if int(p.get("sub_post_number", 0) or 0) > 0), None)
if candidate:
    sub = api("/c/f/pb/floor", {"kz": "11063972585", "pid": candidate["id"], "pn": 1, "rn": 20})
    subposts = sub.get("subpost_list") or []
    print("SUBPOST", json.dumps({
        "pid": candidate["id"],
        "hint": candidate.get("sub_post_number"),
        "returned": len(subposts),
        "first_spid": (subposts[0].get("id") or subposts[0].get("spid")) if subposts else None,
    }, ensure_ascii=False))

# 3) hottopic -> related threads + paging structure
hot_list = web_json("https://tieba.baidu.com/hottopic/browse/topicList")
topics = (((hot_list.get("data") or {}).get("bang_topic") or {}).get("topic_list") or [])
if not topics:
    raise RuntimeError("topicList returned no topics")
topic = topics[0]
base = {
    "topic_name": topic.get("topic_name", ""),
    "page_no": 1,
    "last_id": 0,
    "topic_id": topic.get("topic_id", ""),
    "sort_type": 1,
}
related = web_json("https://tieba.baidu.com/hottopic/browse/getTopicRelateThread?" + urllib.parse.urlencode(base))
tids = sorted(collect_tids(related), key=int)
print("HOTTOPIC", json.dumps({
    "topic_id": topic.get("topic_id"),
    "topic_name": topic.get("topic_name"),
    "thread_count": len(tids),
    "sample_tids": tids[:10],
    "data_keys": list((related.get("data") or {}).keys()) if isinstance(related.get("data"), dict) else [],
    "paging": collect_paging_fields(related),
}, ensure_ascii=False))
if not tids:
    raise RuntimeError("getTopicRelateThread returned no discoverable thread ids")
