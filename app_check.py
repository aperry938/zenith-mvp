import streamlit as st
st.set_page_config(layout="wide", page_title="ZENith CHECK")

import av, cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode

def process_frame(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    # ALWAYS draw a big banner so we can see the callback is active
    cv2.rectangle(img, (20, 20), (520, 120), (0, 100, 255), -1)
    cv2.putText(img, "CALLBACK OK", (40, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255,255,255), 3, cv2.LINE_AA)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Callback Sanity Check")
webrtc_streamer(
    key="check",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=process_frame,
    rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video":{"width":{"ideal":640},"height":{"ideal":480},"frameRate":{"ideal":30}}, "audio":False},
)
