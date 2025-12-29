import os
import streamlit as st
st.set_page_config(layout="wide", page_title="ZENith - Live MVP")

from streamlit_webrtc import webrtc_streamer, WebRtcMode
from zenith_core import ZenithCore

# SINGLETON CORE
if 'core' not in st.session_state:
    st.session_state['core'] = ZenithCore()
core = st.session_state['core']

# ---------- UI ----------
st.title("ZENith $ZEN^{ith}$ - The Dream (Cycle 18)")
st.write("Generative Latent Space Active.")

col1, col2, col3, col4, col5 = st.columns(5)
metrics = core.session.get_current_summary()
life_metrics = core.session.get_lifetime_summary()

col1.metric("Avg Flow", metrics["Avg Flow"])
col2.metric("In The Zone", metrics["Zone Time"])
col3.metric("Locks", metrics["Stability Events"])
col4.metric("Top Pose", metrics["Top Pose"])
col5.metric("Total Practice", life_metrics["Total Time"])

# CONTROLS
st.sidebar.header("Generative Controls")
use_dream = st.sidebar.checkbox("Enable Dream Mode", value=False) # NEW

st.sidebar.header("Standard Controls")
use_vae  = st.sidebar.checkbox("Enable VAE Quality Score", value=True)
use_ghost = st.sidebar.checkbox("Enable The Ghost (VAE Overlay)", value=True)
use_tts  = st.sidebar.checkbox("Enable Voice Coach", value=True)
use_ar   = st.sidebar.checkbox("Enable Visual Whispers (AR)", value=True)
use_grid = st.sidebar.checkbox("Enable Holo-Deck (Grid)", value=True)
use_gamification = st.sidebar.checkbox("Enable Stability Engine", value=True)
use_flow = st.sidebar.checkbox("Enable Flow Score", value=True)
use_seq  = st.sidebar.checkbox("Enable Sequencer", value=True)
use_data = st.sidebar.checkbox("Enable Data Harvest", value=True)
show_dbg = st.sidebar.checkbox("Show debug logs", value=False)
st.sidebar.header("Status / Debug")

if st.sidebar.button("End Session & Save"):
    core.session.save_session()
    st.sidebar.success("Saved to Vault.")

# --- THE SAGE ---
st.sidebar.markdown("---")
st.sidebar.header("The Sage")
if st.sidebar.button("Analyze Latest Frame"):
    if core.sage and core.latest_frame is not None:
        with st.sidebar.spinner("The Sage is analyzing..."):
            advice = core.sage.analyze_frame(core.latest_frame)
            st.sidebar.info(advice)
    else:
        st.sidebar.warning("No frame available yet.")

# --- GALLERY ---
st.sidebar.markdown("---")
st.sidebar.header("Session Gallery")
if st.sidebar.button("Refresh Gallery"):
    images = []
    dataset_root = "dataset"
    if os.path.exists(dataset_root):
        for pose_dir in os.listdir(dataset_root):
            pd = os.path.join(dataset_root, pose_dir)
            if os.path.isdir(pd):
                for f in os.listdir(pd):
                    if f.endswith(".jpg"):
                         images.append(os.path.join(pd, f))
    images.sort(key=os.path.getmtime, reverse=True)
    for img_path in images[:5]:
        st.sidebar.image(img_path, caption=os.path.basename(img_path), use_column_width=True)


def process_callback(frame):
    opts = {
        'use_vae': use_vae,
        'use_ghost': use_ghost,
        'use_tts': use_tts,
        'use_ar': use_ar,
        'use_grid': use_grid,
        'use_gamification': use_gamification,
        'use_flow': use_flow,
        'use_seq': use_seq,
        'use_data': use_data,
        'use_dream': use_dream # NEW
    }
    return core.process_frame(frame, opts)

webrtc_streamer(
    key="zenith-mvp",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=process_callback,
    async_processing=True,
    rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302","stun:stun1.l.google.com:19302"]}]},
    media_stream_constraints={"video":{"width":{"ideal":640},"height":{"ideal":480},"frameRate":{"ideal":30}}, "audio":False},
)
