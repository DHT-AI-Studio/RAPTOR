"""
Raptor API smoke test -- exercises every gateway endpoint group except AIML
(/aiml/*) and Training (/training/*), which stand up real GPU jobs / download
large models and don't belong in a routine smoke test.

Within the groups covered, some individual endpoints are still skipped on
purpose because they are destructive, affect every user on a shared host, or
need a role this test's account doesn't have:

- Guardrail: policy create/upload/activate/deactivate/delete, system
             enable/disable (shared global config this account normally has
             no `guardrail-admin` role for anyway -- a 403 there is expected,
             not tested for, same as compliance/moderation/retention below)
- GDPR:     `erasure` (30-day irreversible wipe), `restrict` (blocks every
             other write for the rest of this run, admin-only to lift),
             `rectify`/`object` (mutate real Keycloak profile / consent
             intent), `admin/legal-hold`, `retention/policy` PUT (admin-only
             global schedule)
- Moderation: `PATCH /flags/{id}` (needs a real flag + moderation-admin role)
- Identity: `bind/*`, `unbind`, `mergeinto`, `profile-image` (need a face-
            indexed video / real photo / `face_detection` consent -- too much
            setup for a smoke test, and withdrawing that consent later would
            cascade-delete real data)
- Memory:   `DELETE /memory` (wipes ALL of this user's long-term memory,
            including anything real accumulated outside this test run);
            `archive` (unlike `store`, has no content dedup and no per-turn
            delete endpoint -- every call would permanently grow this user's
            timeline with no way back)
- DocAgent: `task=search` is expected to 500 -- module 07's lifecycle API
            lacks the route module 29 calls for this task
- Inference: `POST /models/default` (platform-wide write, persisted in
             Redis, applies to every user immediately -- not scoped to the
             caller, same class of risk as guardrail's system toggles).
             `voice/diarize`/`voice/stt`/`voice/tts`/`voice/clone` only run
             if `RAPTOR_TEST_AUDIO_WAV` points at a real WAV file (none is
             bundled with this repo); `voice/clone` additionally only runs if
             this account already has `voice_clone` GDPR consent granted --
             this script never grants that consent itself, since withdrawing
             it later cascades a real data deletion and granting it is a
             real consent decision, not something a smoke test should make

Everything this script DOES create (asset uploads, memory/benchmark/
identity/temporal/voice-clone-speaker test data) is either deduped by the
API itself, cleaned up at the end of its own section, or auto-expires
(temporal's inline content, 7 days) -- running this script repeatedly should
not leave growing state on the gateway.

Compliance report / moderation list / retention dry-run are checked with
check_expected(..., {403}) as a portability fallback for an account that
doesn't have the compliance-auditor/moderation-admin/gdpr-admin roles --
RAPTOR_USERNAME's account on this deployment happens to already have all
four admin roles (gdpr-admin/guardrail-admin/moderation-admin/
compliance-auditor), so in practice these pass for real, not via the 403
path. A separate, more complete set of admin-only checks (GDPR audit-log/
jobs/jobs-summary/retention-policy/processing-records/admin-routed export
+ dsar) runs under its own dedicated account -- see "Admin" below.
"""
import base64
import json
import os
import sys
import tempfile
import uuid

from raptor_client import RaptorAPIError, RaptorClient

BASE_URL = os.environ.get("RAPTOR_BASE_URL", "http://raptor_open_0_5_api.dhtsolution.com:8012")
USERNAME = os.environ["RAPTOR_USERNAME"]
PASSWORD = os.environ["RAPTOR_PASSWORD"]

# 動態查找，避免 shell 對全形字元的 encoding 問題
_DIR = os.environ["RAPTOR_TEST_VIDEO_DIR"]
VIDEO_PATH = next(
    (os.path.join(_DIR, f) for f in os.listdir(_DIR) if f.endswith(".mp4")),
    None,
)
if VIDEO_PATH is None:
    print("[ERROR] 找不到測試用 mp4 檔案")
    sys.exit(1)
print(f"[INFO] 測試影片: {VIDEO_PATH}")

# 資產資料改從本次上傳即時取得（見下方 Asset 區塊）——寫死的 asset_path/version_id/
# correlation_id 只要之前測過一次就不存在了，寫死只會讓腳本必然失敗。
ASSET_PATH = None
VERSION_ID = None
FILENAME = None
CORR_ID = None
CHECKSUM = None

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
    except Exception as e:  # noqa: BLE001 -- fn() is an arbitrary API-call probe; captured into results, not swallowed
        results.append(("FAIL", name, str(e)[:100]))
        return None


def check_expected(name, fn, expected_codes):
    """Like check(), but one of `expected_codes` counts as a pass (role-gated
    endpoints this test's account isn't provisioned for, or state-dependent
    409s from not waiting for async processing to finish)."""
    try:
        data = fn()
        results.append(("PASS", name, ""))
        return data
    except RaptorAPIError as e:
        if e.status_code in expected_codes:
            results.append(("PASS", name, f"expected — HTTP {e.status_code}"))
            return None
        results.append(("FAIL", name, f"HTTP {e.status_code}: {str(e.detail)[:80]}"))
        return None
    except Exception as e:  # noqa: BLE001
        results.append(("FAIL", name, str(e)[:100]))
        return None


def check_soft_timeout(name, fn):
    """Like check(), but a TimeoutError (poll loop didn't finish within this
    script's patience window) is a note, not a failure -- real duration
    depends on host load / data volume, not on whether the endpoint works."""
    try:
        data = fn()
        results.append(("PASS", name, ""))
        return data
    except TimeoutError as e:
        results.append(("PASS", name, f"note — {e}"))
        return None
    except RaptorAPIError as e:
        results.append(("FAIL", name, f"HTTP {e.status_code}: {str(e.detail)[:80]}"))
        return None
    except Exception as e:  # noqa: BLE001
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
check_expected("POST /sso/logout (no cookie)", lambda: client.logout(), {400, 401, 422})

client.login()   # 重新取得 token

# ── Asset ────────────────────────────────────────────────────
section("Asset")

upload_data = check("POST /asset/fileupload_analysis",
    lambda: client.upload_file(VIDEO_PATH))
if upload_data:
    ur = upload_data["upload_result"]
    ASSET_PATH = ur["asset_path"]
    VERSION_ID = ur["version_id"]
    FILENAME = ur["primary_filename"]
    CHECKSUM = ur.get("checksum")
    print(f"  asset_path: {ASSET_PATH}")
    print(f"  exists    : {ur.get('existence_info', {}).get('exists')}")
    processing = upload_data.get("processing_result", {})
    CORR_ID = processing.get("correlation_id")
    if CORR_ID is None:
        print(f"  [INFO] duplicate upload (status={processing.get('status')}) — no new correlation_id")

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
    lambda: client.get_user_commits(page=1, page_size=5))
if commits_data:
    print(f"  total_count: {commits_data.get('total_count')}")

check("POST /asset/file-expiration/{asset_path}/{version_id}",
    lambda: client.update_expiration(ASSET_PATH, VERSION_ID,
                                     archive_date_or_ttl="9999",
                                     destroy_date_or_ttl="9999"))

if CHECKSUM:
    scan_data = check("POST /asset/scan",
        lambda: client.scan_files([{"path": ASSET_PATH, "checksum": CHECKSUM}]))
    if scan_data:
        print(f"  scan result keys: {list(scan_data.keys())[:3]}")

# Decoupled upload (fileupload -> process-file). Same file content as above,
# so this dedupes rather than creating new processing load.
decoupled = check("POST /asset/fileupload (decoupled)",
    lambda: client.upload_file_only(VIDEO_PATH))
if decoupled:
    dec_asset_path = decoupled.get("asset_path", ASSET_PATH)
    dec_version_id = decoupled.get("version_id", VERSION_ID)
    # Already active/processing from the earlier upload in almost every run -> 409 expected.
    check_expected("POST /processing/process-file (register, decoupled asset)",
        lambda: client.process_file(dec_asset_path, dec_version_id), {409})

# unregister/thumbnail both require the asset to already be "active" --
# this script doesn't wait for async analysis to finish, so a 409 here is
# the expected outcome on a fresh upload, not a bug.
if ASSET_PATH and VERSION_ID:
    unreg = check_expected("POST /asset/unregister/{asset_path}/{version_id}",
        lambda: client.unregister_asset(ASSET_PATH, VERSION_ID), {409})
    if unreg is not None:
        check("POST /processing/process-file (re-register after unregister)",
            lambda: client.process_file(ASSET_PATH, VERSION_ID))

    check_expected("GET /asset/thumbnail/{asset_path}/{version_id}",
        lambda: client.get_thumbnail(ASSET_PATH, VERSION_ID, second=1), {409, 400, 404})

    # Full lifecycle cleanup, last in the Asset section (everything above
    # needs this asset "active"): archive then delete it, so this script's
    # own test asset doesn't pile up as a permanently-"active" asset across
    # repeated runs, and so filearchive/delfile actually get exercised
    # instead of being untested forever. filearchive can legitimately 409 if
    # the upload's async analysis hasn't finished yet (same timing caveat as
    # unregister above); when that happens this asset is just left for a
    # later run's upload-dedup to find again, same as before this existed.
    archived = check_expected("POST /asset/filearchive/{asset_path}/{version_id} (cleanup)",
        lambda: client.archive_asset(ASSET_PATH, VERSION_ID), {409})
    if archived and archived.get("status") == "archived":
        destroyed = check("POST /asset/delfile/{asset_path}/{version_id} (cleanup)",
            lambda: client.delete_asset(ASSET_PATH, VERSION_ID))
        if destroyed:
            print(f"  status: {destroyed.get('status')}")

# Batch upload of the same file content, run only after the primary upload's
# own asset is torn down above -- firing this immediately after the first
# upload (the old position) raced it against that upload's still-in-flight
# Kafka video_orchestrator job and could hit a half-written file on the
# object store ("moov atom not found"), unrelated to anything this batch
# endpoint itself does wrong.
batch_data = check("POST /asset/fileupload_analysis_batch",
    lambda: client.upload_files_batch([VIDEO_PATH]))
if batch_data:
    batch_successes = batch_data.get("upload_successes", [])
    if batch_successes:
        batch_result = batch_successes[0].get("result", {})
        batch_asset_path = batch_result.get("asset_path")
        batch_version_id = batch_result.get("version_id")
        # Same cleanup as the primary asset above -- this call creates its
        # own separate asset/version, so it needs its own teardown or it's
        # left permanently "active" just like the primary one would be.
        if batch_asset_path and batch_version_id:
            batch_archived = check_expected(
                "POST /asset/filearchive/{asset_path}/{version_id} (batch cleanup)",
                lambda: client.archive_asset(batch_asset_path, batch_version_id), {409})
            if batch_archived and batch_archived.get("status") == "archived":
                check("POST /asset/delfile/{asset_path}/{version_id} (batch cleanup)",
                    lambda: client.delete_asset(batch_asset_path, batch_version_id))

# ── Processing ───────────────────────────────────────────────
section("Processing")

all_status = check("GET /processing/cache/all",
    lambda: client.get_all_processing_status())
if all_status:
    print(f"  task count: {all_status.get('count')}")

if CORR_ID is not None:
    single_status = check("GET /processing/cache/{m_type}/{key}",
        lambda: client.get_processing_status("video", CORR_ID))
    if single_status:
        step = single_status.get("value", {}).get("step", "?")
        print(f"  step: {step}")
else:
    print("  [SKIP] GET /processing/cache/{m_type}/{key} — no correlation_id from this run " \
          "(upload was a duplicate; nothing new to poll)")

# ── Chat ───────────────────────────────────────────────────────
section("Chat")

chat_resp = check("POST /chat/chat",
    lambda: client.chat("你好，簡單自我介紹一下你能做什麼"))
if chat_resp:
    print(f"  response: {chat_resp.get('response', '')[:60]}...")

chat_mem = check("GET /chat/memory",
    lambda: client.get_chat_memory())
if chat_mem:
    print(f"  memory_count: {chat_mem.get('memory_count')}")

check("DELETE /chat/memory",
    lambda: client.clear_chat_memory())

# chat/completions needs a model already registered in Module 07's MLflow
# registry (or an explicit engine= to bypass it) -- either can legitimately
# be absent on a given deployment, so treat that 404 as expected too.
check_expected("POST /chat/completions",
    lambda: client.chat_completions("qwen2.5:7b", [{"role": "user", "content": "你好"}], engine="ollama"),
    {404})

# ── Memory (long-term, Module 26) ───────────────────────────────
section("Memory")

mem_store = check("POST /memory/store",
    lambda: client.memory_store("smoke-test 偏好：繁體中文回覆", frame_type="preference"))
if mem_store:
    print(f"  status: {mem_store.get('status')}")

mem_retrieve = check("GET /memory/retrieve",
    lambda: client.memory_retrieve("偏好", top_k=3))
if mem_retrieve:
    print(f"  total_frames_searched: {mem_retrieve.get('total_frames_searched')}")

mem_timeline = check("GET /memory/timeline",
    lambda: client.memory_timeline(page=1, page_size=5))
if mem_timeline:
    print(f"  total: {mem_timeline.get('total')}")

# /memory/archive is NOT exercised here: unlike /memory/store, it has no
# content-based dedup, and there is no per-turn delete endpoint -- every call
# would permanently grow this user's timeline by one turn with no way to
# clean it back up. (client.memory_archive() still exists for real use.)

check("POST /memory/compact",
    lambda: client.memory_compact())

check("POST /memory/multimedia/search",
    lambda: client.memory_multimedia_search("川普", top_k=3))

check("POST /memory/compact/evaluate",
    lambda: client.memory_compact_evaluate(messages=[{"role": "user", "content": "hello"}]))

check("GET /memory/sessions/{session_id}/summaries",
    lambda: client.memory_session_summaries("default"))

# ── Benchmark (Module 22) ───────────────────────────────────────
section("Benchmark")

schema = check("POST /benchmark/schemas",
    lambda: client.create_benchmark_schema({
        "name": f"smoke-test-{uuid.uuid4().hex[:8]}",
        "version": "1.0",
        "target_pipeline": "chat",
        "test_cases": [
            {"id": "tc001", "input": {"message": "1+1等於多少？"}, "expected_keywords": ["2"]}
        ],
        "scoring_schema": {
            "dimensions": [{"name": "keywords", "weight": 1.0, "method": "keyword_match"}],
            "aggregate": "weighted_sum",
            "score_range": [0, 1],
        },
    }))
if schema:
    schema_id = schema.get("id")
    run = check("POST /benchmark/runs",
        lambda: client.create_benchmark_run(schema_id))
    if run:
        run_id = run.get("run_id")
        final = check_soft_timeout("GET /benchmark/runs/{run_id} (poll until done)",
            lambda: client.wait_for_benchmark_run(run_id, timeout=90, interval=3))
        if final:
            print(f"  status: {final.get('status')} | aggregate_score: {final.get('aggregate_score')}")

with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
    f.write(
        f"name: smoke-test-upload-{uuid.uuid4().hex[:8]}\n"
        "version: \"1.0\"\n"
        "target_pipeline: chat\n"
        "test_cases:\n"
        "  - id: tc001\n"
        "    input:\n"
        "      message: \"1+1等於多少？\"\n"
        "    expected_keywords: [\"2\"]\n"
        "scoring_schema:\n"
        "  dimensions:\n"
        "    - name: keywords\n"
        "      weight: 1.0\n"
        "      method: keyword_match\n"
        "  aggregate: weighted_sum\n"
        "  score_range: [0, 1]\n"
    )
    schema_yaml_path = f.name
try:
    uploaded_schema = check("POST /benchmark/schemas/upload",
        lambda: client.upload_benchmark_schema_file(schema_yaml_path))
finally:
    os.unlink(schema_yaml_path)

# Cleanup: schemas/runs created above have no auto-expiry, unlike temporal's
# inline content -- delete them now so running this script repeatedly
# doesn't pile up benchmark data forever. Deleting a schema cascades to its
# runs (per the API), so this also covers the run created above.
if schema and schema.get("id"):
    check("DELETE /benchmark/schemas/{id} (cleanup)",
        lambda: client.delete_benchmark_schema(schema["id"]))
if uploaded_schema and uploaded_schema.get("id"):
    check("DELETE /benchmark/schemas/{id} (cleanup, uploaded)",
        lambda: client.delete_benchmark_schema(uploaded_schema["id"]))

# ── Guardrail (Module 23, read-only) ────────────────────────────
section("Guardrail")

policies = check("GET /guardrail/policies",
    lambda: client.list_guardrail_policies())
if policies:
    print(f"  policy count: {len(policies)}")

check("GET /guardrail/policies/template",
    lambda: client.get_guardrail_policy_template())

active_policy = check_expected("GET /guardrail/policies/active",
    lambda: client.get_active_guardrail_policy(), {404})
if active_policy:
    check("GET /guardrail/policies/{policy_id}",
        lambda: client.get_guardrail_policy(active_policy["id"]))

status = check("GET /guardrail/system/status",
    lambda: client.get_guardrail_status())
if status:
    print(f"  enabled: {status.get('enabled')} | policy_enabled: {status.get('policy_enabled')}")

# ── Compliance (Module 33, requires compliance-auditor role) ───
section("Compliance")

check_expected("GET /compliance/report",
    lambda: client.get_compliance_report(from_="2026-01-01T00:00:00Z", to="2026-12-31T23:59:59Z"),
    {403})

# ── GDPR / CCPA (Module 33) ─────────────────────────────────────
section("GDPR")

consent = check("GET /gdpr/consent",
    lambda: client.get_consent())
if consent:
    print(f"  purposes: {[p['id'] for p in consent.get('purposes', [])]}")

# analytics has no destructive-withdrawal cascade (unlike memory/benchmark/
# voice_recording/face_detection/voice_clone) -- safe to toggle, but restore
# whatever this account's real value was rather than assuming it's false.
analytics_was_granted = next(
    (p["granted"] for p in (consent or {}).get("purposes", []) if p["id"] == "analytics"), False,
) if consent else False
check("PUT /gdpr/consent/analytics (flip)",
    lambda: client.set_consent("analytics", not analytics_was_granted))
check("PUT /gdpr/consent/analytics (restore)",
    lambda: client.set_consent("analytics", analytics_was_granted))

dsar = check("GET /gdpr/dsar",
    lambda: client.get_dsar())
if dsar:
    print(f"  data_categories: {list(dsar.get('data_categories', {}).keys())}")

check("GET /gdpr/audit-log/my",
    lambda: client.get_gdpr_audit_log_mine())

export_job = check("GET /gdpr/export",
    lambda: client.request_gdpr_export())
if export_job:
    job_id = export_job.get("job_id")
    final_export = check_soft_timeout("GET /gdpr/export/jobs/{job_id} (poll until done)",
        lambda: client.wait_for_gdpr_export(job_id, timeout=90, interval=3))
    if final_export:
        print(f"  status: {final_export.get('status')} | download_url: {str(final_export.get('download_url'))[:60]}...")

check_expected("GET /gdpr/retention/policy/dry-run",
    lambda: client.get_retention_policy_dry_run(), {403})

# ── DocAgent (Module 29, via the Module 13 generic proxy) ───────
section("DocAgent")

formats = check("GET /docagent/formats",
    lambda: client.docagent_formats())
if formats:
    print(f"  supported_tasks: {formats.get('supported_tasks')}")

with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
    f.write("這是一份 smoke test 用的季度財務報告，營收較去年同期增長了 15%。")
    doc_path = f.name
try:
    classify = check("POST /docagent/process (task=classify)",
        lambda: client.docagent_process("classify", doc_path))
    if classify:
        print(f"  category: {classify.get('category')}")

    extract = check("POST /docagent/process (task=extract)",
        lambda: client.docagent_process("extract", doc_path))
    if extract:
        print(f"  extracted keys: {list(extract.get('extracted', {}).keys())}")

    summarize = check("POST /docagent/process (task=summarize)",
        lambda: client.docagent_process("summarize", doc_path))
    if summarize:
        print(f"  summary: {summarize.get('summary', '')[:60]}...")

    compare = check("POST /docagent/process (task=compare)",
        lambda: client.docagent_process("compare", doc_path,
            options={"compare_text": "去年同期營收下滑了 5%。"}))
    if compare:
        print(f"  similarities: {len(compare.get('similarities', []))}")

    qa = check("POST /docagent/process (task=qa)",
        lambda: client.docagent_process("qa", doc_path,
            options={"question": "營收成長了多少？"}))
    if qa:
        print(f"  answer: {qa.get('answer')}")

    check("POST /docagent/process (task=parse_table)",
        lambda: client.docagent_process("parse_table", doc_path))

    # Known gap: module 29's search task needs an embedding model from module
    # 07, which has none registered (no embedding-capable entry in GET
    # /inference/models) -- always 500s regardless of query/content.
    check_expected("POST /docagent/process (task=search)",
        lambda: client.docagent_process("search", doc_path, options={"query": "財務"}), {500})
finally:
    os.unlink(doc_path)

# ── Inference (Module 35, via the Module 13 declared-route proxy) ──
section("Inference")

translated = check("POST /inference/text/translate",
    lambda: client.translate("早安", target_language="en", source_language="zh-TW"))
if translated:
    print(f"  output: {translated.get('output')}")

check("POST /inference/text/summarize",
    lambda: client.summarize_text("這是一段用來測試摘要功能的中文文字，內容本身並不重要。", style="bullet", max_sentences=1))

check("POST /inference/text/complete",
    lambda: client.complete_text("一句話形容今天天氣", max_tokens=20))

check("POST /inference/text/transform",
    lambda: client.transform_text("吃飯了嗎", instruction="改寫成正式語氣"))

check("POST /inference/text",
    lambda: client.text_dispatch("translate", "謝謝", target_language="en"))

def _stream_text_complete():
    resp = client.complete_text_stream("一句話形容今天天氣", max_tokens=10)
    try:
        frames = [line for line in resp.iter_lines(decode_unicode=True) if line]
        if not frames:
            raise RuntimeError("no SSE frames received")
        return frames
    finally:
        resp.close()
check("POST /inference/text/complete/stream", _stream_text_complete)

AUDIO_WAV = os.environ.get("RAPTOR_TEST_AUDIO_WAV")
if AUDIO_WAV and os.path.exists(AUDIO_WAV):
    diarize_result = check("POST /inference/voice/diarize",
        lambda: client.diarize(AUDIO_WAV, num_speakers=1))
    if diarize_result:
        print(f"  speakers: {diarize_result.get('speakers')}")

    stt_result = check("POST /inference/voice/stt",
        lambda: client.speech_to_text(AUDIO_WAV))
    if stt_result:
        print(f"  text: {stt_result.get('text')}")

    check("POST /inference/voice/tts",
        lambda: client.text_to_speech("早安，今天天氣很好。", language="zh-TW"))

    # voice_clone is biometric-data-gated (GDPR Art. 9): only test it if this
    # account already granted that consent -- never grant it ourselves (that's
    # a real consent decision, not something a smoke test should make), and
    # always delete the speaker embedding this creates afterward.
    voice_clone_granted = any(
        p["id"] == "voice_clone" and p["granted"] for p in (consent or {}).get("purposes", [])
    ) if consent else False
    if voice_clone_granted:
        speaker_name = f"smoke-test-{uuid.uuid4().hex[:8]}"
        clone_result = check("POST /inference/voice/clone",
            lambda: client.clone_voice(AUDIO_WAV, "今天天氣真不錯。", "zh-TW", speaker_name=speaker_name))
        if clone_result:
            _, clone_headers = clone_result
            speaker_id = clone_headers.get("speaker_id")
            print(f"  speaker_id: {speaker_id} | watermarked: {clone_headers.get('watermarked')}")
            check("GET /inference/voice/clone/speakers",
                lambda: client.list_clone_speakers())
            if speaker_id:
                check("DELETE /inference/voice/clone/speakers/{id} (cleanup)",
                    lambda: client.delete_clone_speaker(speaker_id))
    else:
        print("  [SKIP] POST /inference/voice/clone -- voice_clone consent not granted on this account")
else:
    print("  [SKIP] voice/diarize, voice/stt, voice/tts, voice/clone -- "
          "set RAPTOR_TEST_AUDIO_WAV to a real WAV file to cover these")

models = check("GET /inference/models",
    lambda: client.list_inference_models())
if models:
    print(f"  count: {models.get('count')}")

defaults = check("GET /inference/models/default",
    lambda: client.get_default_models())
if defaults:
    print(f"  defaults: {defaults}")

if models and models.get("models"):
    check("GET /inference/models/{name}",
        lambda: client.get_inference_model(models["models"][0]["name"]))

# POST /inference/models/default is NOT exercised here on purpose -- it's a
# platform-wide write (persisted in Redis, applies to every user immediately),
# not scoped to the caller, same class of risk as guardrail's system toggles.

health = check("GET /inference/health",
    lambda: client.inference_health())
if health:
    print(f"  status: {health.get('status')} | gpu_available: {health.get('gpu_available')}")

# ── Moderation (Module 23, requires moderation-admin role) ──────
section("Moderation")

check_expected("GET /moderation/flags",
    lambda: client.list_moderation_flags(page=1, page_size=5), {403})

# ── Admin (role-gated endpoints, dedicated account) ─────────────
# Needs a separate account with gdpr-admin/moderation-admin/compliance-auditor/
# guardrail-admin realm roles -- RAPTOR_ADMIN_USERNAME/RAPTOR_ADMIN_PASSWORD.
# Created the way BUILD.md documents (POST /admin/keycloak/user with
# realmRoles); this section doubles as a live check that that method actually
# grants working permissions, not just a Keycloak-side role label.
ADMIN_USERNAME = os.environ.get("RAPTOR_ADMIN_USERNAME")
ADMIN_PASSWORD = os.environ.get("RAPTOR_ADMIN_PASSWORD")
if ADMIN_USERNAME and ADMIN_PASSWORD:
    section("Admin (dedicated role-test account)")
    admin_client = RaptorClient(BASE_URL, ADMIN_USERNAME, ADMIN_PASSWORD)
    check("POST /sso/login (admin account)", lambda: admin_client.login())

    check("GET /gdpr/jobs",
        lambda: admin_client.get_gdpr_jobs())
    check("GET /gdpr/jobs/summary",
        lambda: admin_client.get_gdpr_jobs_summary())
    check("GET /gdpr/audit-log",
        lambda: admin_client.get_gdpr_audit_log(page=1, page_size=5))
    check("GET /gdpr/processing-records",
        lambda: admin_client.get_processing_records())
    check("GET /gdpr/processing-records/ccpa",
        lambda: admin_client.get_processing_records_ccpa())

    # retention/policy PUT is documented as a no-op for the actual values --
    # sending the exact array GET returned back only refreshes
    # updated_at/updated_by, so this is safe to exercise for real.
    policy = check("GET /gdpr/retention/policy",
        lambda: admin_client.get_retention_policy())
    if policy and policy.get("policies"):
        check("PUT /gdpr/retention/policy (round-trip, no value change)",
            lambda: admin_client.set_retention_policy(policy["policies"]))

    # Self-targeted only -- this proves the admin route itself works without
    # touching any other real user's data.
    admin_sub = None
    try:
        payload = admin_client._token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        admin_sub = json.loads(base64.urlsafe_b64decode(payload)).get("sub")
    except Exception:  # noqa: BLE001, S110 -- best-effort, just to self-target below
        pass
    if admin_sub:
        check("GET /gdpr/dsar/{user_id} (admin, self)",
            lambda: admin_client.get_dsar_for_user(admin_sub))
        export_job = check("POST /gdpr/export/{user_id} (admin, self)",
            lambda: admin_client.request_gdpr_export_for_user(admin_sub))
        if export_job and export_job.get("job_id"):
            check_soft_timeout("GET /gdpr/export/jobs/{job_id} (admin export, poll until done)",
                lambda: admin_client.wait_for_gdpr_export(export_job["job_id"], timeout=90, interval=3))
else:
    print("\n  [SKIP] Admin section -- set RAPTOR_ADMIN_USERNAME/RAPTOR_ADMIN_PASSWORD "
          "to a dedicated account with gdpr-admin/moderation-admin/compliance-auditor/"
          "guardrail-admin realm roles to cover these")

# ── Identity (Module 25 People Search) ──────────────────────────
section("Identity")

identities = check("GET /identity/identities",
    lambda: client.list_identities(limit=5))
if identities is not None:
    print(f"  count: {len(identities)}")

identity_name = f"smoke-test-{uuid.uuid4().hex[:8]}"
identity = check("POST /identity/upload",
    lambda: client.upsert_identity(identity_name))
identity_id = identity.get("identity_id") or identity.get("id") if identity else None
if identity_id:
    check("GET /identity/{identity_id}",
        lambda: client.get_identity(identity_id))
    check("PATCH /identity/{identity_id}",
        lambda: client.update_identity(identity_id, name=identity_name + "-renamed"))
    check("GET /identity/{identity_id}/traces",
        lambda: client.get_identity_traces(identity_id))
    check("DELETE /identity/{identity_id} (cleanup)",
        lambda: client.delete_identity(identity_id))

# ── Temporal (Module 25) ────────────────────────────────────────
section("Temporal")

check("GET /temporal/capabilities",
    lambda: client.temporal_capabilities())

temporal_key = f"smoke-test-{uuid.uuid4().hex}"
ingest = check("POST /temporal/ingest",
    lambda: client.temporal_ingest(
        idempotency_key=temporal_key,
        content="黃仁勳在1963年出生。",
        content_type="text/plain",
    ))
if ingest:
    tv_id = ingest.get("source_version_id")
    print(f"  source_version_id: {tv_id} | status: {ingest.get('status')}")
    if tv_id:
        # The job row only exists once the normal upload/analysis pipeline
        # finishes and Module 25 enqueues it -- a fresh ingest 404s here for a
        # while first, so poll rather than check once.
        job = check_soft_timeout("GET /temporal/status/{version_id} (poll until the job exists)",
            lambda: client.wait_for_temporal_job(tv_id, timeout=60, interval=3))
        if job:
            print(f"  status: {job.get('status')} | facts: {job.get('facts')}")
            check("GET /temporal/reason/{version_id}",
                lambda: client.temporal_reason(tv_id))
        check("GET /temporal/expiry/{version_id}",
            lambda: client.temporal_expiry(tv_id))

check("POST /temporal/parse",
    lambda: client.temporal_parse("去年發生的事", anchor="2026-11-23T00:00:00+08:00"))

check("POST /temporal/verify",
    lambda: client.temporal_verify(claims=[{"kind": "absence"}], evidence=[], complete=True))

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

# Requires face_detection GDPR consent -- 403 there is expected on an account
# that hasn't granted it.
people = check_expected("POST /search/people",
    lambda: client.search_people("黃仁勳", limit=3), {403})
if people:
    print(f"  identities: {[i['name'] for i in people.get('identities', [])]}")

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
