import streamlit as st
import requests

st.set_page_config(page_title="RAPTOR Vision Demo", layout="wide")

st.title(" RAPTOR Vision Analysis Demo")

with st.sidebar:
    st.header("設定")
    api_url = st.text_input("API URL", "http://localhost:8000/analyze/video")
    uploaded_file = st.file_uploader("上傳影片", type=["mp4", "avi", "mov"])

    st.markdown("---")
    st.markdown("### ⏱️ 時間同步設定")
    start_min = st.number_input("片段開始時間 (分)", min_value=0, value=0)
    start_sec = st.number_input("片段開始時間 (秒)", min_value=0, max_value=59, value=0)
    
    total_start_seconds = (start_min * 60) + start_sec
    st.caption(f"系統將以此時間 ( {start_min:02d}:{start_sec:02d} ) 作為基準進行校正")

    custom_prompt = st.text_area(
        "自訂提示詞 (Prompt)",
        height=150,
        value="You are an expert video analyst. Report in Traditional Chinese:\n1. Key Objects\n2. Micro-Action Breakdown\n3. Summary\n4. Causal Analysis"
    )
    
    analyze_btn = st.button("開始分析", type="primary", disabled=not uploaded_file)

col1, col2 = st.columns([1, 1])

if uploaded_file:
    with col1:
        st.video(uploaded_file)
        st.info(f"影片名稱: {uploaded_file.name}")

if analyze_btn and uploaded_file:
    with col2:
        st.markdown("### 分析報告")
        with st.spinner("正在進行視覺推理與時序校正..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {
                    "start_time": total_start_seconds 
                }
                
                response = requests.post(api_url, files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("status") == "success":
                        st.markdown(result["analysis"])
                        st.success(f"分析完成！(基準時間: {result['meta']['clip_start_time']})")
                    else:
                        st.error(f"分析失敗: {result.get('message')}")
                else:
                    st.error(f"API 連線錯誤: {response.status_code}")
            except Exception as e:
                st.error(f"發生錯誤: {e}")