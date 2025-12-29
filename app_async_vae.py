import os, time
from collections import deque, Counter
import concurrent.futures as cf
import threading, queue

import streamlit as st
st.set_page_config(layout="wide", page_title="ZENith - Live MVP")

import av, cv2, numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import mediapipe as mp
import joblib

# --- TensorFlow (VAE for quality) ---
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- INTERNAL IMPORTS ---
try:
    from pose_foundations import PoseHeuristics
except ImportError:
    PoseHeuristics = None  # Graceful fallback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    # Avoid Metal/GPU stalls on some Macs
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass
# Keep TF lightweight (prevents contention with WebRTC thread)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# ---------- UI ----------
st.title("ZENith $ZEN^{ith}$ - Visual Whispers (AR Beta)")
st.write("Perform a yoga flow. The AI will provide audio AND visual corrections.")
use_vae  = st.sidebar.checkbox("Enable VAE Quality Score", value=True)
use_tts  = st.sidebar.checkbox("Enable Voice Coach", value=True)
use_ar   = st.sidebar.checkbox("Enable Visual Whispers (AR)", value=True)
show_dbg = st.sidebar.checkbox("Show debug logs", value=False)
st.sidebar.header("Status / Debug")

# ---------- Files ----------
CLF_PATH   = "zenith_pose_classifier.pkl"
ENC_W_PATH = "zenith_encoder_weights.weights.h5"
DEC_W_PATH = "zenith_decoder_weights.weights.h5"
POSE_NAMES = ["Chair","Downward Dog","Extended Side Angle","High Lunge","Mountain Pose","Plank","Tree","Triangle","Warrior II"]
INT_TO_POSE = {i:n for i,n in enumerate(POSE_NAMES)}

# ---------- TTS ENGINE (Thread-Safe) ----------
tts_queue = queue.Queue()

def tts_worker():
    """
    Dedicated worker thread for text-to-speech.
    Initializes the engine locally to avoid 'run loop' issues on main thread.
    """
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 160) # Slightly faster coaching voice
        while True:
            text = tts_queue.get()
            if text is None: break # Poison pill
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")
            tts_queue.task_done()
    except ImportError:
        print("pyttsx3 not installed. Voice disabled.")

# Start TTS Thread
threading.Thread(target=tts_worker, daemon=True).start()

# Debounce State
last_spoken_time = 0
DEBOUNCE_SECONDS = 5.0 
active_correction = None # Store current correction for drawing

# ---------- Classifier ----------
pose_classifier, clf_ok = None, False
try:
    if os.path.exists(CLF_PATH):
        pose_classifier = joblib.load(CLF_PATH)
        clf_ok = True
        st.sidebar.success(f"Loaded classifier: {CLF_PATH}")
    else:
        st.sidebar.warning(f"Classifier not found: {CLF_PATH}")
except Exception as e:
    st.sidebar.error(f"Classifier load error: {e}")

# ---------- VAE (non-blocking) ----------
encoder = decoder = None
vae_ok = False

def build_vae(input_dim=132, latent_dim=16):
    class Sampling(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            eps = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], tf.shape(z_mean)[1]))
            return z_mean + tf.exp(0.5 * z_log_var) * eps
    enc_in = keras.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(enc_in)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    enc = keras.Model(enc_in, [z_mean, z_log_var, z], name="encoder")
    z_in = keras.Input(shape=(latent_dim,))
    x = layers.Dense(64, activation="relu")(z_in)
    dec_out = layers.Dense(input_dim, activation="sigmoid")(x)
    dec = keras.Model(z_in, dec_out, name="decoder")
    return enc, dec

if use_vae:
    try:
        if os.path.exists(ENC_W_PATH) and os.path.exists(DEC_W_PATH):
            encoder, decoder = build_vae()
            encoder.load_weights(ENC_W_PATH)
            decoder.load_weights(DEC_W_PATH)
            vae_ok = True
            st.sidebar.success("Loaded VAE weights")
        else:
            st.sidebar.warning("VAE weights not found; disable VAE to avoid stalls.")
    except Exception as e:
        st.sidebar.error(f"VAE load error: {e}")

def vae_quality(flat_keypoints, low=0.0005, high=0.006):
    # Runs INSIDE a worker thread
    z_mean, _, z = encoder.predict(flat_keypoints, verbose=0)
    recon = decoder.predict(z, verbose=0)
    mse = float(np.mean((flat_keypoints - recon) ** 2))
    q = 100.0 * (1.0 - (mse - low) / (high - low))
    return float(np.clip(q, 0, 100))

# Single background worker for VAE
vae_pool = cf.ThreadPoolExecutor(max_workers=1)
vae_future = None
last_q = None
alpha = 0.25  # EMA

# ---------- MediaPipe ----------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def landmarks_to_flat(lms):
    arr = np.array([[lm.x,lm.y,lm.z,lm.visibility] for lm in lms.landmark], dtype=np.float32)
    return arr.flatten()[None,:]  # (1,132)

# ---------- State ----------
hist = deque(maxlen=15)
i = 0
fps = 0.0
t0 = time.time()
last_label = None
last_ok_ts = 0.0
TTL = 1.5

def majority(dq): return Counter(dq).most_common(1)[0][0] if dq else None

def draw_hud(img, label, q, fps):
    # Minimalist HUD - Replacing emojis with text/shapes as per Audit
    cv2.rectangle(img,(20,20),(520,120),(65,109,255),-1)
    cv2.putText(img,f"POSE: {label or '—'}",(35,60),cv2.FONT_HERSHEY_SIMPLEX,0.95,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(img,"QUALITY:",(35,97),cv2.FONT_HERSHEY_SIMPLEX,0.85,(255,255,255),2,cv2.LINE_AA)
    qv = 0 if q is None else int(round(q))
    bar_x,bar_y,bar_w,bar_h = 160,78,280,20
    cv2.rectangle(img,(bar_x,bar_y),(bar_x+bar_w,bar_y+bar_h),(20,20,20),-1)
    color = (0,170,0) if qv>85 else ((0,200,255) if qv>60 else (0,0,255))
    cv2.rectangle(img,(bar_x,bar_y),(bar_x+int(bar_w*qv/100),bar_y+bar_h),color,-1)
    cv2.putText(img,f"{qv:3d}",(bar_x+bar_w+12,bar_y+16),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(img,f"{fps:.1f} FPS",(img.shape[1]-140,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(230,230,230),2,cv2.LINE_AA)

def draw_ar_arrow(img, start_norm, end_norm, color=(255, 255, 0), thickness=4):
    """
    Draws a correction arrow on the image.
    Inputs are normalized coordinates (0.0 - 1.0).
    Color should be BGR.
    """
    h, w, _ = img.shape
    start_px = (int(start_norm[0] * w), int(start_norm[1] * h))
    end_px   = (int(end_norm[0] * w), int(end_norm[1] * h))
    
    cv2.arrowedLine(img, start_px, end_px, color, thickness, tipLength=0.3)
    # Draw a Glow bloom? (Simple circle at tip)
    cv2.circle(img, end_px, 6, (255,255,255), -1)

def process_frame(frame: av.VideoFrame) -> av.VideoFrame:
    global i, fps, t0, last_label, last_ok_ts, vae_future, last_q, last_spoken_time, active_correction
    img = frame.to_ndarray(format="bgr24")

    # FPS
    t1 = time.time(); dt = t1 - t0; t0 = t1
    if dt > 0: fps = 0.9*fps + 0.1*(1.0/dt) if fps else (1.0/dt)

    # Pose detect & draw
    res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if res.pose_landmarks:
        mp_draw.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                               mp_draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

    label = None
    q_disp = last_q

    try:
        # Prediction & Logic Loop (Every 3 frames)
        if res.pose_landmarks and (i % 3 == 0):
            feats = landmarks_to_flat(res.pose_landmarks)

            if clf_ok and pose_classifier is not None:
                pred = pose_classifier.predict(feats)
                label = INT_TO_POSE.get(int(pred[0]), "Unknown")
                hist.append(label)
                label = majority(hist)

                # --- COACHING LOGIC ---
                if PoseHeuristics and label:
                    lms = {lm_id: [lm.x, lm.y] for lm_id, lm in enumerate(res.pose_landmarks.landmark)}
                    correction = PoseHeuristics.evaluate(label, lms)
                    
                    if correction:
                        # AUDIO
                        if use_tts:
                            now = time.time()
                            if (now - last_spoken_time) > DEBOUNCE_SECONDS:
                                tts_queue.put(correction['text'])
                                last_spoken_time = now
                        
                        # VISUAL (Store for drawing every frame)
                        active_correction = correction
                    else:
                        active_correction = None

            # Non-blocking VAE
            if use_vae and vae_ok:
                if (i % 6 == 0) and (vae_future is None or vae_future.done()):
                    vae_future = vae_pool.submit(vae_quality, feats.copy())
                if vae_future is not None and vae_future.done():
                    q = vae_future.result()
                    last_q = q if last_q is None else (alpha*q + (1-alpha)*last_q)
                    q_disp = last_q

            if label or q_disp is not None:
                last_label = label or last_label
                last_ok_ts = time.time()
    except Exception as e:
        if show_dbg: st.sidebar.error(f"Processor error: {e}")

    i += 1

    # Keep HUD stable briefly if nothing new
    if (time.time() - last_ok_ts) < TTL:
        label = label or last_label

    # --- RENDER VISUAL WHISPERS ---
    if use_ar and active_correction:
        # Ensure correction has a vector
        if 'vector' in active_correction:
            start_pt, end_pt = active_correction['vector']
            # Color is RGB in dict (usually), convert to BGR for OpenCV
            # Our Foundations file returns (0, 255, 255) which is Yellow/Cyan.
            # Let's assume the Foundations file returns BGR or we fix it here.
            # Foundations returns (0, 255, 255) -> Yellow in RGB. BGR would be (255, 255, 0)
            col = active_correction.get('color', (0, 255, 255))
            # Fix if needed. Let's just use Cyan (255, 255, 0 in BGR)
            draw_ar_arrow(img, start_pt, end_pt, color=(255, 255, 0))

    draw_hud(img, label, q_disp, fps)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="zenith-mvp",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=process_frame,
    async_processing=True,
    rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302","stun:stun1.l.google.com:19302"]}]},
    media_stream_constraints={"video":{"width":{"ideal":640},"height":{"ideal":480},"frameRate":{"ideal":30}}, "audio":False},
)
