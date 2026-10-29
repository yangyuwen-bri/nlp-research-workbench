import hashlib
import json
import urllib.parse
import urllib.request

CLIENT_VERSION = "9.7.8.0"
params = {
    "_client_id": "wappc_1534235498291_633",
    "_client_type": "2",
    "_client_version": CLIENT_VERSION,
    "_phone_imei": "000000000000000",
    "kz": "11063972585",
    "pn": "1",
    "rn": "30",
    "r": "0",
    "lz": "0",
    "st": "0",
    "z": "0",
}
raw = "".join(k + "=" + str(params[k]) for k in sorted(params)) + "tiebaclient!!!"
params["sign"] = hashlib.md5(raw.encode("utf-8")).hexdigest().upper()
req = urllib.request.Request(
    "http://c.tieba.baidu.com/c/f/pb/page",
    data=urllib.parse.urlencode(params).encode("utf-8"),
    headers={
        "User-Agent": f"bdtb for Android {CLIENT_VERSION}",
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    },
)
with urllib.request.urlopen(req, timeout=20) as resp:
    payload = json.loads(resp.read().decode("utf-8", "replace"))

error_code = payload.get("error_code", -1)
if int(error_code) != 0:
    raise RuntimeError(f"Tieba API error {error_code}: {payload.get('error_msg', '')}")
thread = payload.get("thread") or {}
posts = payload.get("post_list") or []
page = payload.get("page") or {}
print(json.dumps({
    "error_code": error_code,
    "title": thread.get("title"),
    "forum": thread.get("fname") or (payload.get("display_forum") or {}).get("name"),
    "posts_on_page": len(posts),
    "total_page": page.get("total_page"),
    "total_num": page.get("total_num"),
    "first_post_id": posts[0].get("id") if posts else None,
}, ensure_ascii=False))
if not posts:
    raise RuntimeError("API returned success but no post_list")
