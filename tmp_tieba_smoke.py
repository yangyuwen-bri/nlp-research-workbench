import hashlib
import html
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

# 2) subposts / 楼中楼 if this sample has any
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
else:
    print("SUBPOST", json.dumps({"skipped": "sample page has no subposts"}, ensure_ascii=False))

# 3) hottopic aggregate page -> actual /p/<tid> discovery
hot_url = "https://tieba.baidu.com/hottopic/browse/hottopic?topic_id=28366306"
req = urllib.request.Request(hot_url, headers={"User-Agent": WEB_UA, "Referer": "https://tieba.baidu.com/"})
with urllib.request.urlopen(req, timeout=20) as resp:
    text = resp.read().decode("utf-8", "replace")
tids = sorted(set(re.findall(r"(?:https?://(?:c\\.)?tieba\\.baidu\\.com)?/p/(\\d+)", html.unescape(text))), key=int)
print("HOTTOPIC", json.dumps({"topic_id": "28366306", "thread_count": len(tids), "sample_tids": tids[:10]}, ensure_ascii=False))
if not tids:
    raise RuntimeError("hottopic page returned no discoverable /p/<tid> links")
