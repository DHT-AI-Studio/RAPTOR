"""
Raptor 0.3 全 API 驗證腳本
跳過破壞性操作：delfile、filearchive
"""
import os
import sys
from raptor_client import RaptorClient, RaptorAPIError

BASE_URL   = "http://raptor_open_0_3_api.dhtsolution.com:8012"
USERNAME   = "test_basicuser"
PASSWORD   = "12345"

# 動態查找，避免 shell 對全形字元的 encoding 問題
_DIR = "/opt/home/cing/cing_test/Raptor_0.2_Modular_TH_20260409/Raptor_0.3"
VIDEO_PATH = next(
    (os.path.join(_DIR, f) for f in os.listdir(_DIR) if f.endswith(".mp4")),
    None,
)
if VIDEO_PATH is None:
    print("[ERROR] 找不到測試用 mp4 檔案")
    sys.exit(1)
print(f"[INFO] 測試影片: {VIDEO_PATH}")

# 已知的資產資料（上次上傳）
ASSET_PATH  = "video/mp4/用水沖一次就乾淨的衣服__1080p"
VERSION_ID  = "1786020961fbc1719bbb7a1e25b08d96871a84a29417d9812e108e87f41e81ae"
FILENAME    = "用水沖一次就乾淨的衣服__1080p.mp4"
CORR_ID     = "89de0bff-b371-484b-b254-99c7f36a183a"

SEP = "─" * 60
results = []

def check(name, fn):
    try:
        data = fn()
        results.append(("PASS", name, ""))
        return data
    except RaptorAPIError as e:
        results.append(("FAIL", name, f"HTTP {e.status_code}: {str(e.detail)[:80]}"))
        return None
    except Exception as e:
        results.append(("FAIL", name, str(e)[:100]))
        return None

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

# ── 初始化 ────────────────────────────────────────────────────
client = RaptorClient(BASE_URL, USERNAME, PASSWORD)

# ── SSO ──────────────────────────────────────────────────────
section("SSO")

token = check("POST /sso/login",
    lambda: client.login())
print(f"  token: {client._token[:40]}..." if client._token else "  no token")

# logout 需要 browser cookie，預期回 400/422，算作已知行為
def _logout():
    try:
        return client.logout()
    except RaptorAPIError as e:
        if e.status_code in (400, 401, 422):
            return {"note": f"expected — requires realm_name cookie (HTTP {e.status_code})"}
        raise
check("POST /sso/logout (no cookie → expected error)", _logout)

client.login()   # 重新取得 token

# ── Asset ────────────────────────────────────────────────────
section("Asset")

upload_data = check("POST /asset/fileupload_analysis",
    lambda: client.upload_file(VIDEO_PATH))
if upload_data:
    ur = upload_data["upload_result"]
    print(f"  asset_path: {ur['asset_path']}")
    print(f"  exists    : {ur.get('existence_info', {}).get('exists')}")

check("POST /asset/fileupload_analysis_batch",
    lambda: client.upload_files_batch([VIDEO_PATH]))

dl_data = check("GET /asset/filedownload/{asset_path}/{version_id}",
    lambda: client.download_asset(ASSET_PATH, VERSION_ID))
if dl_data:
    pf = dl_data.get("primary_file", {})
    url = pf.get("url", "")
    print(f"  primary_file url: {url[:80]}...")

ver_data = check("GET /asset/fileversions/{asset_path}/{filename}",
    lambda: client.list_versions(ASSET_PATH, FILENAME))
if ver_data:
    versions = ver_data.get("versions", [])
    print(f"  versions count: {len(versions)}")

commits_data = check("GET /asset/users/commits",
    lambda: client.get_user_commits(keyword="用水", page=1, page_size=5))
if commits_data:
    print(f"  total_count: {commits_data.get('total_count')}")

check("POST /asset/file-expiration/{asset_path}/{version_id}",
    lambda: client.update_expiration(ASSET_PATH, VERSION_ID,
                                     archive_date_or_ttl="9999",
                                     destroy_date_or_ttl="9999"))

# ── Processing ───────────────────────────────────────────────
section("Processing")

all_status = check("GET /processing/cache/all",
    lambda: client.get_all_processing_status())
if all_status:
    print(f"  task count: {all_status.get('count')}")

single_status = check("GET /processing/cache/{m_type}/{key}",
    lambda: client.get_processing_status("video", CORR_ID))
if single_status:
    step = single_status.get("value", {}).get("step", "?")
    print(f"  step: {step}")

# ── Search ───────────────────────────────────────────────────
section("Search")

QUERY = "衣服 用水清洗"

hybrid = check("POST /search/hybrid",
    lambda: client.search_hybrid(QUERY, top_k=3))
if hybrid:
    print(f"  results: {len(hybrid.get('results', []))}")

bm25 = check("POST /search/bm25",
    lambda: client.search_bm25(QUERY, top_k=3))
if bm25:
    print(f"  results: {len(bm25.get('results', []))}")

vector = check("POST /search/vector",
    lambda: client.search_vector(QUERY, top_k=3))
if vector:
    print(f"  results: {len(vector.get('results', []))}")

graphrag = check("POST /search/graphrag",
    lambda: client.search_graphrag("PEM-5 塗層材料", max_depth=2, limit=20))
if graphrag:
    print(f"  matched_entities: {len(graphrag.get('matched_entities', []))}")

tkg = check("POST /search/tkg",
    lambda: client.search_tkg("衣服清洗技術", max_depth=2, limit=20))
if tkg:
    print(f"  matched_entities: {len(tkg.get('matched_entities', []))}")

video_srch = check("POST /search/video_search",
    lambda: client.video_search(QUERY, top_k=3))
if video_srch:
    print(f"  total videos: {video_srch.get('total')}")
    for v in video_srch.get("results", []):
        url = v.get("asset_url", "")
        print(f"    [{v['score']:.4f}] {v['filename']}")
        print(f"    url: {url[:80]}...")

# ── RAG ──────────────────────────────────────────────────────
section("RAG")

rag = check("POST /a2a/query",
    lambda: client.query("這件衣服用水沖洗的原理是什麼？", top_k=3))
if rag:
    print(f"  chunks_used: {rag.get('chunks_used')}")
    answer = rag.get("answer", "")
    print(f"  answer: {answer[:100]}...")

# ── Summary ───────────────────────────────────────────────────
section("結果摘要")
passed = [r for r in results if r[0] == "PASS"]
failed = [r for r in results if r[0] == "FAIL"]

for status, name, msg in results:
    icon = "✓" if status == "PASS" else "✗"
    line = f"  {icon} {name}"
    if msg:
        line += f"\n      → {msg}"
    print(line)

print(f"\n  PASS {len(passed)} / {len(results)}   FAIL {len(failed)}")
if failed:
    sys.exit(1)
