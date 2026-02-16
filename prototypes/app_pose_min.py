import streamlit as st
st.set_page_config(layout="wide", page_title="ZENith Pose MIN")

import av, cv2, mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def process_frame(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if res.pose_landmarks:
        mp_draw.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # Always draw HUD so we know callback is running
    cv2.rectangle(img, (20, 20), (520, 120), (245,117,66), -1)
    cv2.putText(img, "POSE: (min)", (35, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(img, "QUALITY: --", (35, 97), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2, cv2.LINE_AA)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("ZENith – Pose MIN")
webrtc_streamer(
    key="pose-min",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=process_frame,
    rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video":{"width":{"ideal":640},"height":{"ideal":480},"frameRate":{"ideal":30}}, "audio":False},
)
