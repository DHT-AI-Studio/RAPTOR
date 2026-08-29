"""
Memory Service — 互動式測試 GUI
Run:
    streamlit run demo_gui.py
"""
import base64
import json
import streamlit as st
import httpx
from datetime import datetime


def _decode_jwt_payload(token: str) -> dict:
    """Decode JWT payload (base64url) without verifying signature."""
    try:
        payload_b64 = token.split(".")[1]
        # Add padding
        payload_b64 += "=" * (-len(payload_b64) % 4)
        return json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:
        return {}

st.set_page_config(page_title="Memory Service Demo", layout="wide")
st.title("Memory Service Demo")

# ── Sidebar：連線設定 + 登入 ──────────────────────────────────────────────────

with st.sidebar:
    st.header("連線設定")
    base_url     = st.text_input("Service URL",  value="http://localhost:8026")
    gateway_url  = st.text_input("Gateway URL",  value="http://localhost:8012")

    st.divider()
    if st.button("Health Check"):
        try:
            r = httpx.get(f"{base_url}/health", timeout=5)
            if r.json().get("status") == "ok":
                st.success("✅ 服務正常")
            else:
                st.error(f"異常回應：{r.text}")
        except Exception as e:
            st.error(f"連線失敗：{e}")

    st.divider()
    st.header("登入")
    username   = st.text_input("Username", value="test_basicuser")
    password   = st.text_input("Password", type="password", value="12345")
    realm_name = st.text_input("Realm",    value="dhtsolution")
    client_id  = st.text_input("Client",   value="raptor")

    if st.button("Login"):
        try:
            r = httpx.post(
                f"{gateway_url}/api/0.3/sso/login",
                data={
                    "username":   username,
                    "password":   password,
                    "realm_name": realm_name,
                    "client_id":  client_id,
                },
                timeout=10,
            )
            if r.status_code == 200:
                body = r.json()
                token = body.get("access_token") if isinstance(body, dict) else body.strip()
                st.session_state["token"] = token
                payload = _decode_jwt_payload(token)
                st.session_state["user_id"] = payload.get("sub", "")
                st.success("✅ 登入成功")
            else:
                st.error(f"登入失敗 {r.status_code}：{r.text}")
        except Exception as e:
            st.error(f"連線失敗：{e}")

    if "token" in st.session_state:
        st.caption(f"Token: ...{st.session_state['token'][-20:]}")
        uid = st.session_state.get("user_id", "")
        if uid:
            st.info(f"User ID (sub): `{uid}`")
    else:
        st.caption("尚未登入")


def headers():
    user_id = st.session_state.get("user_id", "")
    return {"X-User-ID": user_id, "Content-Type": "application/json"}


def show_response(r: httpx.Response):
    st.markdown(f"**HTTP {r.status_code}**")
    if r.content:
        try:
            st.json(r.json())
        except Exception:
            st.code(r.text)


# ── 兩個分頁 ──────────────────────────────────────────────────────────────────

tab_session, tab_longterm, tab_multimedia, tab_search, tab_manage, tab_extract, tab_compact = st.tabs(
    ["Session 記憶", "Long-term 記憶", "Multimedia 記憶", "搜尋 ", "Timeline / 管理", "記憶提取", "記憶壓縮"]
)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Session 記憶
# ════════════════════════════════════════════════════════════════════════════════

with tab_session:
    col1, col2 = st.columns(2)

    # ── 寫入一輪對話 ──────────────────────────────────────────────────────────
    with col1:
        st.subheader("寫入對話 (POST /turns)")
        s_id  = st.text_input("session_id", value="sess_001", key="s_write_id")
        u_msg = st.text_area("user_message", value="你好，我想查上次討論的項目", height=80, key="s_user")
        a_msg = st.text_area("assistant_response", value="根據之前的記錄，上次討論的是...", height=80, key="s_asst")

        if st.button("寫入", key="s_write"):
            try:
                r = httpx.post(
                    f"{base_url}/memory/sessions/{s_id}/turns",
                    headers=headers(),
                    json={"user_message": u_msg, "assistant_response": a_msg},
                    timeout=10,
                )
                show_response(r)
            except Exception as e:
                st.error(str(e))

        st.divider()

        # ── 列出所有 sessions ──────────────────────────────────────────────────
        st.subheader("列出所有 Sessions (GET /sessions)")
        if st.button("列出", key="s_list"):
            try:
                r = httpx.get(f"{base_url}/memory/sessions", headers=headers(), timeout=10)
                show_response(r)
            except Exception as e:
                st.error(str(e))

    # ── 查詢 / 刪除 ────────────────────────────────────────────────────────────
    with col2:
        st.subheader("查最近 N 輪 (GET /recent)")
        s_id_r  = st.text_input("session_id", value="sess_001", key="s_read_id")
        n_turns = st.number_input("n（最近幾輪）", min_value=1, max_value=100, value=5, key="s_n")

        if st.button("查詢", key="s_read"):
            try:
                r = httpx.get(
                    f"{base_url}/memory/sessions/{s_id_r}/recent",
                    headers=headers(),
                    params={"n": n_turns},
                    timeout=10,
                )
                show_response(r)
            except Exception as e:
                st.error(str(e))

        st.divider()

        # ── 刪除 session ───────────────────────────────────────────────────────
        st.subheader("刪除 Session (DELETE)")
        s_id_d = st.text_input("session_id", value="sess_001", key="s_del_id")

        if st.button("刪除", key="s_del", type="primary"):
            try:
                r = httpx.delete(
                    f"{base_url}/memory/sessions/{s_id_d}",
                    headers=headers(),
                    timeout=10,
                )
                if r.status_code == 204:
                    st.success("✅ 已刪除")
                else:
                    show_response(r)
            except Exception as e:
                st.error(str(e))


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Long-term 記憶
# ════════════════════════════════════════════════════════════════════════════════

with tab_longterm:
    col3, col4 = st.columns(2)

    # ── 寫入事實 ──────────────────────────────────────────────────────────────
    with col3:
        st.subheader("寫入事實 (POST /facts)")
        fact_text  = st.text_area("text", value="用戶偏好繁體中文回覆", height=80, key="lt_text")
        frame_type = st.selectbox("frame_type", ["preference", "fact", "entity", "conversation"], key="lt_type")
        lt_session = st.text_input("session_id（可選）", value="", key="lt_sess")

        if st.button("寫入", key="lt_write"):
            try:
                payload = {"text": fact_text, "frame_type": frame_type}
                if lt_session:
                    payload["session_id"] = lt_session
                r = httpx.post(
                    f"{base_url}/memory/longterm/facts",
                    headers=headers(),
                    json=payload,
                    timeout=10,
                )
                show_response(r)
            except Exception as e:
                st.error(str(e))

        st.divider()

        # ── 列出所有 preference + fact ─────────────────────────────────────────
        st.subheader("列出 preference + fact + entity (GET /facts)")
        if st.button("列出", key="lt_list"):
            try:
                r = httpx.get(f"{base_url}/memory/longterm/facts", headers=headers(), timeout=10)
                show_response(r)
            except Exception as e:
                st.error(str(e))

        st.divider()

        # ── 清空 long-term ─────────────────────────────────────────────────────
        st.subheader("清空 Long-term 記憶 (DELETE)")
        if st.button("清空", key="lt_del", type="primary"):
            try:
                r = httpx.delete(f"{base_url}/memory/longterm", headers=headers(), timeout=10)
                if r.status_code == 204:
                    st.success("✅ 已清空")
                else:
                    show_response(r)
            except Exception as e:
                st.error(str(e))

    # ── 語意搜尋 ──────────────────────────────────────────────────────────────
    with col4:
        st.subheader("語意搜尋 (POST /search)")
        query = st.text_input("query", value="語言偏好", key="lt_query")
        top_k = st.number_input("top_k", min_value=1, max_value=50, value=5, key="lt_topk")

        use_date = st.checkbox("加日期過濾", key="lt_use_date")
        from_date, to_date = None, None
        if use_date:
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                from_d    = st.date_input("from_date", key="lt_from")
                from_date = datetime(from_d.year, from_d.month, from_d.day).timestamp()
            with d_col2:
                to_d    = st.date_input("to_date", key="lt_to")
                to_date = datetime(to_d.year, to_d.month, to_d.day, 23, 59, 59).timestamp()

        if st.button("搜尋", key="lt_search"):
            try:
                payload: dict = {"query": query, "top_k": top_k}
                if use_date:
                    payload["from_date"] = from_date
                    payload["to_date"]   = to_date
                r = httpx.post(
                    f"{base_url}/memory/longterm/search",
                    headers=headers(),
                    json=payload,
                    timeout=15,
                )
                data = r.json()
                st.markdown(f"**HTTP {r.status_code}** — {len(data)} 筆結果")
                for i, hit in enumerate(data, 1):
                    score = hit.get("score", 0)
                    color = "🟢" if score >= 0.8 else "🟡" if score >= 0.5 else "🔴"
                    with st.expander(f"{color} [{i}] score={score:.3f}  [{hit.get('frame_type')}]"):
                        st.write(f"**text:** {hit.get('text')}")
                        st.write(f"**session_id:** {hit.get('session_id') or '—'}")
                        ts = hit.get("timestamp", 0)
                        if ts:
                            st.write(f"**timestamp:** {datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception as e:
                st.error(str(e))


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Multimedia 記憶
# ════════════════════════════════════════════════════════════════════════════════

with tab_multimedia:
    col5, col6 = st.columns(2)

    # ── 左欄：索引 ────────────────────────────────────────────────────────────
    with col5:
        media_type_write = st.selectbox("媒體類型", ["video", "audio", "image"], key="mm_type")
        st.subheader(f"索引 {media_type_write.capitalize()} (POST /{media_type_write})")

        mm_asset = st.text_input("asset_path", value=f"media/sample.{media_type_write[:3]}", key="mm_asset")
        mm_ver   = st.text_input("version_id", value="v1", key="mm_ver")
        mm_sess  = st.text_input("session_id（可選）", value="", key="mm_sess")
        mm_ctx   = st.text_input("context_query（可選）", value="", key="mm_ctx")

        if media_type_write in ("video", "audio"):
            mm_start = st.number_input("start_sec", value=0.0, step=0.1, key="mm_start")
            mm_end   = st.number_input("end_sec",   value=30.0, step=0.1, key="mm_end")
            mm_trans = st.text_area("transcription", value="", height=80, key="mm_trans")
            if media_type_write == "audio":
                st.caption("transcription 留空則自動呼叫 Module 07 Whisper ASR")

        if media_type_write == "image":
            mm_desc = st.text_area("description", value="", height=60, key="mm_desc")
            mm_ocr  = st.text_area("ocr_text",    value="", height=60, key="mm_ocr")

        if st.button("索引", key="mm_write"):
            try:
                if media_type_write == "video":
                    payload = {
                        "asset_path": mm_asset, "version_id": mm_ver,
                        "start_sec": mm_start, "end_sec": mm_end,
                        "transcription": mm_trans,
                        "session_id": mm_sess, "context_query": mm_ctx,
                    }
                elif media_type_write == "audio":
                    payload = {
                        "asset_path": mm_asset, "version_id": mm_ver,
                        "start_sec": mm_start, "end_sec": mm_end,
                        "session_id": mm_sess, "context_query": mm_ctx,
                    }
                    if mm_trans:
                        payload["transcription"] = mm_trans
                else:
                    payload = {
                        "asset_path": mm_asset, "version_id": mm_ver,
                        "description": mm_desc, "ocr_text": mm_ocr,
                        "session_id": mm_sess,
                    }
                r = httpx.post(
                    f"{base_url}/memory/multimedia/{media_type_write}",
                    headers=headers(),
                    json=payload,
                    timeout=120,
                )
                show_response(r)
            except Exception as e:
                st.error(str(e))

    # ── 右欄：搜尋 ────────────────────────────────────────────────────────────
    with col6:
        st.subheader("跨媒體搜尋 (POST /search)")
        mm_query  = st.text_input("query", value="cooling system", key="mm_query")
        mm_topk   = st.number_input("top_k", min_value=1, max_value=50, value=5, key="mm_topk")
        mm_filter = st.selectbox("media_type 過濾（可選）", ["（全部）", "video", "audio", "image"], key="mm_filter")

        if st.button("搜尋", key="mm_search"):
            try:
                payload2: dict = {"query": mm_query, "top_k": mm_topk}
                if mm_filter != "（全部）":
                    payload2["media_type"] = mm_filter
                r = httpx.post(
                    f"{base_url}/memory/multimedia/search",
                    headers=headers(),
                    json=payload2,
                    timeout=15,
                )
                data = r.json()
                st.markdown(f"**HTTP {r.status_code}** — {len(data)} 筆結果")
                for i, hit in enumerate(data, 1):
                    score = hit.get("score", 0)
                    color = "🟢" if score >= 0.8 else "🟡" if score >= 0.5 else "🔴"
                    media = hit.get("media_type", "")
                    icon  = {"video": "🎬", "audio": "🎵", "image": "🖼️"}.get(media, "📄")
                    with st.expander(f"{color} {icon} [{i}] score={score:.3f}  [{media}]"):
                        st.write(f"**asset_path:** {hit.get('asset_path')}")
                        st.write(f"**version_id:** {hit.get('version_id')}")
                        if hit.get("start_sec") is not None:
                            st.write(f"**time:** {hit.get('start_sec'):.1f}s — {hit.get('end_sec'):.1f}s")
                        st.write(f"**text:** {hit.get('text_snippet')}")
                        if hit.get("session_id"):
                            st.write(f"**session_id:** {hit.get('session_id')}")
            except Exception as e:
                st.error(str(e))


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — 搜尋 (MV-7): Session 搜尋 + 全域搜尋
# ════════════════════════════════════════════════════════════════════════════════

with tab_search:
    col_s1, col_s2 = st.columns(2)

    # ── 左欄：Session 搜尋 ────────────────────────────────────────────────────
    with col_s1:
        st.subheader("Session 搜尋 (POST /sessions/{id}/search)")
        ss_id    = st.text_input("session_id", value="sess_001", key="ss_id")
        ss_query = st.text_input("query", value="冷卻系統", key="ss_query")
        ss_topk  = st.number_input("top_k", min_value=1, max_value=50, value=5, key="ss_topk")

        ss_use_date = st.checkbox("加日期過濾 (ISO 8601)", key="ss_use_date")
        ss_from, ss_to = None, None
        if ss_use_date:
            d1, d2 = st.columns(2)
            with d1:
                from_d  = st.date_input("from_date", key="ss_from_d")
                ss_from = datetime(from_d.year, from_d.month, from_d.day).isoformat() + "Z"
            with d2:
                to_d  = st.date_input("to_date", key="ss_to_d")
                ss_to = datetime(to_d.year, to_d.month, to_d.day, 23, 59, 59).isoformat() + "Z"

        if st.button("搜尋 Session", key="ss_search"):
            try:
                payload: dict = {"query": ss_query, "top_k": ss_topk}
                if ss_use_date:
                    payload["from_date"] = ss_from
                    payload["to_date"]   = ss_to
                r = httpx.post(
                    f"{base_url}/memory/sessions/{ss_id}/search",
                    headers=headers(),
                    json=payload,
                    timeout=15,
                )
                body = r.json()
                total_searched = body.get("total_frames_searched", 0)
                hits = body.get("hits", [])
                st.markdown(f"**HTTP {r.status_code}** — {len(hits)} 筆結果 / 共搜尋 {total_searched} 個 frames")
                for i, hit in enumerate(hits, 1):
                    score = hit.get("score", 0)
                    color = "🟢" if score >= 0.8 else "🟡" if score >= 0.5 else "🔴"
                    label = f"{color} [{i}] score={score:.3f}  turn #{hit.get('turn_index')}  {hit.get('timestamp', '')[:19]}"
                    with st.expander(label):
                        st.write(f"**User:** {hit.get('user_message')}")
                        st.write(f"**Assistant:** {hit.get('assistant_response')}")
                        st.caption(f"text: {hit.get('text')}")
            except Exception as e:
                st.error(str(e))

    # ── 右欄：全域搜尋 ────────────────────────────────────────────────────────
    with col_s2:
        st.subheader("全域搜尋 (POST /memory/search)")
        gs_query = st.text_input("query", value="引擎冷卻", key="gs_query")
        gs_topk  = st.number_input("top_k", min_value=1, max_value=50, value=5, key="gs_topk")
        gs_scope = st.multiselect(
            "scope",
            ["sessions", "longterm", "multimedia"],
            default=["sessions", "longterm", "multimedia"],
            key="gs_scope",
        )

        if st.button("全域搜尋", key="gs_search"):
            try:
                r = httpx.post(
                    f"{base_url}/memory/search",
                    headers=headers(),
                    json={"query": gs_query, "top_k": gs_topk, "scope": gs_scope},
                    timeout=20,
                )
                body = r.json()
                total = body.get("total_frames_searched", 0)
                st.markdown(f"**HTTP {r.status_code}** — 共搜尋 {total} 個 frames")

                # Sessions scope results
                s_hits = body.get("sessions", [])
                if "sessions" in gs_scope:
                    st.markdown(f"**Sessions** — {len(s_hits)} 筆")
                    for i, h in enumerate(s_hits, 1):
                        score = h.get("score", 0)
                        color = "🟢" if score >= 0.8 else "🟡" if score >= 0.5 else "🔴"
                        with st.expander(f"{color} [{i}] score={score:.3f}  session={h.get('session_id')}  turn #{h.get('turn_index')}"):
                            st.write(f"**User:** {h.get('user_message')}")
                            st.write(f"**Assistant:** {h.get('assistant_response')}")
                            st.caption(f"timestamp: {h.get('timestamp', '')[:19]}")

                # Longterm scope results
                lt_hits = body.get("longterm", [])
                if "longterm" in gs_scope:
                    st.markdown(f"**Long-term** — {len(lt_hits)} 筆")
                    for i, h in enumerate(lt_hits, 1):
                        score = h.get("score", 0)
                        color = "🟢" if score >= 0.8 else "🟡" if score >= 0.5 else "🔴"
                        with st.expander(f"{color} [{i}] score={score:.3f}  [{h.get('frame_type')}]"):
                            st.write(f"**text:** {h.get('text')}")
                            st.caption(f"timestamp: {h.get('timestamp', '')[:19]}")

                # Multimedia scope results
                mm_hits = body.get("multimedia", [])
                if "multimedia" in gs_scope:
                    st.markdown(f"**Multimedia** — {len(mm_hits)} 筆")
                    for i, h in enumerate(mm_hits, 1):
                        score = h.get("score", 0)
                        color = "🟢" if score >= 0.8 else "🟡" if score >= 0.5 else "🔴"
                        icon  = {"video": "🎬", "audio": "🎵", "image": "🖼️"}.get(h.get("media_type", ""), "📄")
                        with st.expander(f"{color} {icon} [{i}] score={score:.3f}  {h.get('asset_path')}"):
                            st.write(f"**text:** {h.get('text')}")
                            if h.get("start_sec") is not None:
                                st.write(f"**time:** {h.get('start_sec'):.1f}s — {h.get('end_sec'):.1f}s")
                            st.caption(f"timestamp: {h.get('timestamp', '')[:19]}")

            except Exception as e:
                st.error(str(e))


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — Timeline / 管理 (MV-8)
# ════════════════════════════════════════════════════════════════════════════════

with tab_manage:
    # ── Timeline + Time-Travel ────────────────────────────────────────────────
    st.subheader("Timeline (GET /sessions/{id}/timeline)")
    tm_col1, tm_col2 = st.columns([2, 1])
    with tm_col1:
        tm_id        = st.text_input("session_id", value="sess_001", key="tm_id")
        tm_page      = st.number_input("page", min_value=1, value=1, key="tm_page")
        tm_page_size = st.number_input("page_size", min_value=1, max_value=100, value=20, key="tm_page_size")
    with tm_col2:
        tm_use_at = st.checkbox("Time-Travel (at=)", key="tm_use_at")
        tm_at = None
        if tm_use_at:
            tm_at_date = st.date_input("日期", key="tm_at_date")
            tm_at_time = st.time_input("時間", key="tm_at_time")
            tm_at = datetime(
                tm_at_date.year, tm_at_date.month, tm_at_date.day,
                tm_at_time.hour, tm_at_time.minute, tm_at_time.second,
            ).isoformat() + "Z"
            st.caption(f"at={tm_at}")

    if st.button("取得 Timeline", key="tm_fetch"):
        try:
            params: dict = {"page": tm_page, "page_size": tm_page_size}
            if tm_use_at and tm_at:
                params["at"] = tm_at
            r = httpx.get(
                f"{base_url}/memory/sessions/{tm_id}/timeline",
                headers=headers(),
                params=params,
                timeout=15,
            )
            body = r.json()
            total     = body.get("total", 0)
            has_next  = body.get("has_next", False)
            entries   = body.get("entries", [])
            st.markdown(f"**HTTP {r.status_code}** — 共 {total} 輪｜第 {tm_page} 頁 / {'有下一頁' if has_next else '最後一頁'}")
            for e in entries:
                ts_str = e.get("timestamp", "")[:19].replace("T", " ")
                label  = f"Turn #{e.get('turn_index')}  {ts_str}"
                with st.expander(label):
                    st.write(f"**User:** {e.get('user_message')}")
                    st.write(f"**Assistant:** {e.get('assistant_response')}")
                    if e.get("tool_calls"):
                        st.caption(f"tool_calls: {e.get('tool_calls')}")
                    if e.get("media_refs"):
                        st.caption(f"media_refs: {len(e.get('media_refs'))} 筆")
        except Exception as e:
            st.error(str(e))

    st.divider()

    # ── Stats + Export ────────────────────────────────────────────────────────
    mgmt_col1, mgmt_col2 = st.columns(2)

    with mgmt_col1:
        st.subheader("記憶統計 (GET /memory/stats)")
        if st.button("取得統計", key="stats_fetch"):
            try:
                r = httpx.get(f"{base_url}/memory/stats", headers=headers(), timeout=10)
                body = r.json()
                st.markdown(f"**HTTP {r.status_code}**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Sessions", body.get("session_count", 0))
                c2.metric("總對話輪數", body.get("total_turns", 0))
                c3.metric("媒體記憶", body.get("total_media_items", 0))
                c4, c5 = st.columns(2)
                c4.metric("Long-term frames", body.get("long_term_frame_count", 0))
                bytes_used = body.get("storage_bytes_used", 0)
                c5.metric("儲存空間", f"{bytes_used / 1024:.1f} KB" if bytes_used < 1024 * 1024 else f"{bytes_used / 1024 / 1024:.2f} MB")
            except Exception as e:
                st.error(str(e))

    with mgmt_col2:
        st.subheader("匯出記憶 (GET /memory/export)")
        if st.button("下載匯出 JSON", key="export_fetch"):
            try:
                r = httpx.get(f"{base_url}/memory/export", headers=headers(), timeout=60)
                doc = r.json()
                schema_ver = doc.get("export_schema_version", "?")
                n_sessions = len(doc.get("sessions", []))
                n_lt       = len(doc.get("longterm", []))
                n_mm       = len(doc.get("multimedia", []))
                st.success(f"schema v{schema_ver}｜{n_sessions} sessions｜{n_lt} long-term frames｜{n_mm} 媒體記錄")
                st.download_button(
                    label="⬇ 儲存為 JSON",
                    data=json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name=f"memory_export_{doc.get('exported_at', 'now')[:10]}.json",
                    mime="application/json",
                    key="export_dl",
                )
            except Exception as e:
                st.error(str(e))

    st.divider()

    # ── GDPR 清除所有記憶 ─────────────────────────────────────────────────────
    st.subheader("清除所有記憶 (DELETE /memory) — GDPR 抹除")
    st.warning("⚠️ 此操作將永久刪除當前用戶的所有 Session、Long-term 及 Multimedia 記憶，無法還原。")
    confirm_delete = st.checkbox("我確認要清除所有記憶", key="del_confirm")
    if st.button("清除所有記憶", key="del_all", type="primary", disabled=not confirm_delete):
        try:
            r = httpx.delete(f"{base_url}/memory", headers=headers(), timeout=15)
            if r.status_code == 204:
                st.success("✅ 所有記憶已清除")
            else:
                show_response(r)
        except Exception as e:
            st.error(str(e))


# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — 記憶提取
# ════════════════════════════════════════════════════════════════════════════════

with tab_extract:
    ex_col1, ex_col2 = st.columns(2)

    # ── 左欄：手動觸發提取 ────────────────────────────────────────────────────
    with ex_col1:
        st.subheader("手動觸發提取 (POST /longterm/extract)")
        st.caption("內部端點，直接呼叫服務，不需經過 Gateway")

        uid_display = st.session_state.get("user_id", "")
        if uid_display:
            st.info(f"User ID（來自 JWT）: `{uid_display}`")
        else:
            st.warning("尚未登入，請先於側欄登入以取得 user_id")

        ex_sess_id = st.text_input("session_id", value="sess_001", key="ex_sess_id")
        ex_user_msg = st.text_area("user_message", height=80, key="ex_user_msg",
                                   value="我偏好用繁體中文")
        ex_asst_msg = st.text_area("assistant_response", height=80, key="ex_asst_msg",
                                   value="好的，我已記下您的偏好。")

        if st.button("觸發提取", key="ex_trigger"):
            if not st.session_state.get("token"):
                st.warning("請先登入")
            else:
                try:
                    r = httpx.post(
                        f"{base_url}/memory/longterm/extract",
                        headers=headers(),
                        json={
                            "session_id": ex_sess_id,
                            "turn": {
                                "user_message": ex_user_msg,
                                "assistant_response": ex_asst_msg,
                            },
                        },
                        timeout=60,
                    )
                    if r.status_code == 200:
                        stored = r.json().get("stored", 0)
                        if stored:
                            st.success(f"✅ 提取完成，新增 {stored} 筆記憶")
                        else:
                            st.info("提取完成，無新增項目（LLM 判斷無需更新）")
                    else:
                        show_response(r)
                except Exception as e:
                    st.error(str(e))

        st.divider()

        # ── 自動提取說明 ───────────────────────────────────────────────────────
        st.subheader("自動提取機制")
        st.info(
            "每次呼叫 **POST /sessions/{id}/turns** 寫入對話後，"
            "系統會自動在背景（fire-and-forget）對該輪進行提取。\n\n"
            "提取流程：\n"
            "1. 取出用戶現有 long-term facts（最多 10 筆）\n"
            "2. 連同當前這一輪傳給 LLM（`qwen2.5:7b`）\n"
            "3. LLM 輸出 ADD / DELETE / UPDATE 操作\n"
            "4. 執行操作，舊偏好自動被新偏好取代"
        )

    # ── 右欄：查看現有 Long-term Facts ────────────────────────────────────────
    with ex_col2:
        st.subheader("現有 Long-term Facts (GET /longterm/facts)")
        st.caption("顯示 preference + fact + entity 類型的記憶，含 frame_id 供識別")

        if st.button("重新載入", key="ex_reload"):
            st.session_state["ex_facts_loaded"] = True

        if st.session_state.get("ex_facts_loaded"):
            try:
                r = httpx.get(f"{base_url}/memory/longterm/facts", headers=headers(), timeout=10)
                if r.status_code == 200:
                    facts = r.json()
                    if not facts:
                        st.info("目前沒有任何 long-term facts")
                    else:
                        st.markdown(f"共 **{len(facts)}** 筆")
                        for i, f in enumerate(facts):
                            ftype = f.get("frame_type") or f.get("label", "")
                            fid   = f.get("frame_id", "?")
                            text  = f.get("text", "")
                            ts    = f.get("timestamp", 0)
                            ts_str = datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M") if ts else ""
                            icon = {"preference": "💡", "fact": "📌", "entity": "🏷️"}.get(ftype, "📄")
                            with st.expander(f"{icon} [{ftype}]  {text[:50]}"):
                                st.write(f"**全文：** {text}")
                                st.caption(f"frame_id: {fid}  |  session: {f.get('session_id') or '—'}  |  {ts_str}")
                                if st.button("刪除此筆", key=f"ex_del_{i}_{fid}"):
                                    try:
                                        dr = httpx.delete(
                                            f"{base_url}/memory/longterm/facts/{fid}",
                                            headers=headers(),
                                            timeout=10,
                                        )
                                        if dr.status_code == 204:
                                            st.success("✅ 已刪除，請重新載入")
                                            st.session_state["ex_facts_loaded"] = False
                                        else:
                                            show_response(dr)
                                    except Exception as e:
                                        st.error(str(e))
                else:
                    show_response(r)
            except Exception as e:
                st.error(str(e))
        else:
            st.caption("按「重新載入」查看")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 7 — 記憶壓縮 (MV-12)
# ════════════════════════════════════════════════════════════════════════════════

with tab_compact:
    st.caption("壓縮 session 歷史，將超過 context window 的對話摘要為結構化記憶。")

    # ── 快速產生測試資料 ──────────────────────────────────────────────────────────
    with st.expander("🛠️ 快速產生測試資料（讓 session 達到壓縮門檻）"):
        st.caption(
            "每輪訊息約 3000 tokens。寫入 N 輪後，session 的總 token 數會超過 "
            "max_tail_tokens（預設 40000），讓壓縮功能實際觸發。"
        )
        with st.expander("📄 測試資料內容預覽（點擊展開）"):
            st.markdown("**User 訊息模板：**")
            st.code(
                "渦輪引擎冷卻系統技術討論：材料選用 316L 不鏽鋼（ASTM A240），"
                "工作溫度 -40°C 至 +180°C，壓力等級 PN25，水泵流量需達額定值 105%，"
                "冷卻液乙二醇比例 30:70，pH 7.5-8.5，散熱器安全係數 1.25-1.3。",
                language=None,
            )
            st.markdown("**Assistant 回應模板：**")
            st.code(
                "根據技術規格，建議採用板式熱交換器搭配變頻水泵：傳熱效率高 20-30%，"
                "節省電耗 30-40%。熱負荷計算 Q=mCpΔT，翅片密度 12-16 fin/inch，"
                "材料銅鋁複合結構，焊縫 100% 超音波探傷（AWS D1.1）。",
                language=None,
            )
            st.caption("實際寫入時會將此模板重複拼接至目標 token 數。")
        seed_col1, seed_col2, seed_col3 = st.columns(3)
        with seed_col1:
            seed_session = st.text_input("session_id", value="sess_compact_test", key="seed_sess")
        with seed_col2:
            seed_turns = st.number_input("寫入幾輪", min_value=1, max_value=50, value=20, key="seed_turns")
        with seed_col3:
            seed_tokens = st.number_input("每輪約多少 tokens", min_value=500, max_value=5000, value=3000, key="seed_tokens")

        if st.button("產生測試資料", key="seed_generate"):
            if not st.session_state.get("token"):
                st.warning("請先登入")
            else:
                # chars = tokens × 4
                chars_per_msg = seed_tokens * 2  # 一半給 user，一半給 assistant
                user_msg = (
                    "渦輪引擎冷卻系統技術討論：材料選用 316L 不鏽鋼（ASTM A240），"
                    "工作溫度 -40°C 至 +180°C，壓力等級 PN25，水泵流量需達額定值 105%，"
                    "冷卻液乙二醇比例 30:70，pH 7.5-8.5，散熱器安全係數 1.25-1.3。"
                )
                asst_msg = (
                    "根據技術規格，建議採用板式熱交換器搭配變頻水泵：傳熱效率高 20-30%，"
                    "節省電耗 30-40%。熱負荷計算 Q=mCpΔT，翅片密度 12-16 fin/inch，"
                    "材料銅鋁複合結構，焊縫 100% 超音波探傷（AWS D1.1）。"
                )
                # repeat until we hit target char count
                user_msg  = (user_msg  * (chars_per_msg // len(user_msg)  + 1))[:chars_per_msg]
                asst_msg  = (asst_msg  * (chars_per_msg // len(asst_msg)  + 1))[:chars_per_msg]

                progress = st.progress(0, text="寫入中...")
                errors = 0
                for i in range(int(seed_turns)):
                    try:
                        r = httpx.post(
                            f"{base_url}/memory/sessions/{seed_session}/turns",
                            headers=headers(),
                            json={"user_message": user_msg, "assistant_response": asst_msg},
                            timeout=15,
                        )
                        if r.status_code != 201:
                            errors += 1
                    except Exception:
                        errors += 1
                    progress.progress((i + 1) / int(seed_turns), text=f"寫入中... {i+1}/{int(seed_turns)}")

                total_tokens_est = int(seed_turns) * seed_tokens
                if errors == 0:
                    over = total_tokens_est > 40000
                    st.success(
                        f"✅ 完成！session `{seed_session}` 已寫入 {int(seed_turns)} 輪，"
                        f"估算共 ~{total_tokens_est:,} tokens。"
                    )
                    if over:
                        st.info(
                            "已超過 max_tail_tokens（40,000），可在下方填入相同 session_id，"
                            "設定 context_window ≤ 80,000，然後按「執行壓縮」觸發壓縮。"
                        )
                    else:
                        st.warning("token 數尚未超過 40,000，建議增加輪數或每輪 tokens 再試。")
                else:
                    st.warning(f"完成，但有 {errors} 輪寫入失敗（可能未登入或服務異常）")

    st.divider()
    cp_col1, cp_col2 = st.columns(2)

    # ── 左欄：Evaluate + Compact ─────────────────────────────────────────────
    with cp_col1:

        # ── Evaluate ──────────────────────────────────────────────────────────
        st.subheader("估算 Token 用量 (POST /compact/evaluate)")
        st.caption("輸入任意 messages 列表，回傳是否需要壓縮及超出多少 tokens。")

        cp_ev_cw     = st.number_input("context_window", value=128000, step=1000, key="cp_ev_cw")
        cp_ev_extra  = st.number_input("extra_tokens（當前 turn 預估）", value=0, step=500, key="cp_ev_extra")
        cp_ev_msgs   = st.text_area(
            "messages（JSON 陣列）",
            value='[{"role":"user","content":"你好"},{"role":"assistant","content":"你好！"}]',
            height=100,
            key="cp_ev_msgs",
        )

        if st.button("估算", key="cp_evaluate"):
            try:
                msgs = json.loads(cp_ev_msgs)
                r = httpx.post(
                    f"{base_url}/memory/compact/evaluate",
                    headers=headers(),
                    json={"messages": msgs, "context_window": cp_ev_cw, "extra_tokens": cp_ev_extra},
                    timeout=10,
                )
                body = r.json()
                st.markdown(f"**HTTP {r.status_code}**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Token 數量", body.get("token_count", 0))
                c2.metric("Threshold", body.get("auto_compact_threshold", 0))
                c3.metric("超出", body.get("tokens_over_threshold", 0))
                if body.get("should_compact"):
                    st.error("⚠️ 需要壓縮")
                else:
                    st.success("✅ 尚在 budget 內")
            except json.JSONDecodeError:
                st.error("messages 格式錯誤，請輸入合法 JSON 陣列")
            except Exception as e:
                st.error(str(e))

        st.divider()

        # ── Compact ───────────────────────────────────────────────────────────
        st.subheader("壓縮 Session (POST /sessions/{id}/compact)")

        cp_sess_id  = st.text_input("session_id", value="sess_001", key="cp_sess_id")
        cp_cw       = st.number_input("context_window", value=128000, step=1000, key="cp_cw")
        cp_trigger  = st.selectbox("trigger", ["auto", "manual"], key="cp_trigger")
        cp_dry_run  = st.checkbox("dry_run（不實際寫入摘要，只回傳估算）", value=True, key="cp_dry_run")
        cp_custom   = st.text_area(
            "custom_instructions（可選，額外摘要指示）",
            value="",
            height=60,
            key="cp_custom",
        )

        if st.button("壓縮", key="cp_compact"):
            try:
                payload: dict = {
                    "context_window": cp_cw,
                    "trigger": cp_trigger,
                    "dry_run": cp_dry_run,
                }
                if cp_custom.strip():
                    payload["custom_instructions"] = cp_custom.strip()
                r = httpx.post(
                    f"{base_url}/memory/sessions/{cp_sess_id}/compact",
                    headers=headers(),
                    json=payload,
                    timeout=120,
                )
                body = r.json()
                st.markdown(f"**HTTP {r.status_code}**")

                if body.get("compacted"):
                    st.success(f"✅ 壓縮完成  source={body.get('source')}  fallback={body.get('fallback_used')}")
                else:
                    st.info(f"ℹ️ 未壓縮：{body.get('source')}")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("壓縮前 tokens", body.get("pre_compact_tokens", 0))
                c2.metric("壓縮後 tokens", body.get("post_compact_tokens", 0))
                c3.metric("已摘要 turns", body.get("turns_compacted", 0))
                c4.metric("保留 turns",   body.get("turns_kept", 0))

                if body.get("summary_id"):
                    st.caption(f"summary_id: `{body.get('summary_id')}`")
                if body.get("summary_frame_written"):
                    st.caption("📝 Summary frame 已寫入 MemVID")
            except Exception as e:
                st.error(str(e))

    # ── 右欄：列出 / 刪除 Summary ─────────────────────────────────────────────
    with cp_col2:
        st.subheader("查看 Summaries (GET /sessions/{id}/summaries)")

        cp_list_id = st.text_input("session_id", value="sess_001", key="cp_list_id")

        if st.button("載入 Summaries", key="cp_list"):
            try:
                r = httpx.get(
                    f"{base_url}/memory/sessions/{cp_list_id}/summaries",
                    headers=headers(),
                    timeout=10,
                )
                summaries = r.json()
                st.markdown(f"**HTTP {r.status_code}** — {len(summaries)} 個 summary frame")
                st.session_state["cp_summaries"] = summaries
                st.session_state["cp_list_session"] = cp_list_id
            except Exception as e:
                st.error(str(e))

        summaries = st.session_state.get("cp_summaries", [])
        if summaries:
            for i, s in enumerate(summaries, 1):
                tr = s.get("source_turn_range", {})
                label = (
                    f"[{i}] source={s.get('source')}  "
                    f"turns {tr.get('from')}→{tr.get('to')}  "
                    f"{s.get('token_count_before')}→{s.get('token_count_after')} tokens"
                )
                with st.expander(label):
                    st.write(f"**summary_id:** `{s.get('summary_id')}`")
                    st.write(f"**created_at:** {s.get('created_at', '')[:19]}")
                    st.write(f"**preserved tail from turn:** {s.get('preserved_tail_from_turn')}")
                    st.markdown("**摘要內容：**")
                    st.text(s.get("text", "（空）"))

                    if st.button(f"刪除此 Summary", key=f"cp_del_{i}"):
                        try:
                            sess = st.session_state.get("cp_list_session", cp_list_id)
                            r = httpx.delete(
                                f"{base_url}/memory/sessions/{sess}/summaries/{s.get('summary_id')}",
                                headers=headers(),
                                timeout=10,
                            )
                            if r.status_code == 204:
                                st.success("✅ 已刪除，請重新載入")
                                st.session_state.pop("cp_summaries", None)
                            else:
                                show_response(r)
                        except Exception as e:
                            st.error(str(e))
        elif "cp_summaries" in st.session_state:
            st.info("此 session 尚無 summary frame")
