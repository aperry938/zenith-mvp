import os, time, sys
from collections import deque, Counter
import concurrent.futures as cf
import threading, queue
import random
import atexit

import streamlit as st
st.set_page_config(layout="wide", page_title="ZENith - Live MVP")

import av, cv2, numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import mediapipe as mp
import joblib

# --- TensorFlow ---
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- INTERNAL IMPORTS ---
try:
    from pose_foundations import PoseHeuristics
    from pose_sequencer import PoseSequencer
    from ui_overlay import ZenithUI
    from session_manager import SessionManager
    from data_harvester import DataHarvester # NEW
except ImportError:
    PoseHeuristics = None  
    PoseSequencer = None
    ZenithUI = None
    SessionManager = None
    DataHarvester = None

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# ---------- SESSION ----------
if 'session' not in st.session_state:
    if SessionManager: st.session_state['session'] = SessionManager()
    else:
        # Dummy
        class DummySession:
            def update(self, *args): pass
            def log_stability_event(self): pass
            def get_current_summary(self): return {"Duration":"0s", "Avg Flow":"0", "Stability Events":0, "Zone Time":"0s", "Top Pose":"-"}
            def get_lifetime_summary(self): return {"Total Time":"0h 0m", "Sessions":0}
            def save_session(self): pass
        st.session_state['session'] = DummySession()
session = st.session_state['session']

if 'sequencer' not in st.session_state:
    if PoseSequencer: st.session_state['sequencer'] = PoseSequencer()
    else: st.session_state['sequencer'] = None
sequencer = st.session_state['sequencer']

if 'ui' not in st.session_state:
    if ZenithUI: st.session_state['ui'] = ZenithUI()
    else: st.session_state['ui'] = None
ui = st.session_state['ui']

if 'harvester' not in st.session_state:
    if DataHarvester: st.session_state['harvester'] = DataHarvester()
    else: st.session_state['harvester'] = None
harvester = st.session_state['harvester']


# ---------- UI ----------
st.title("ZENith $ZEN^{ith}$ - The Ecosystem (Beta)")
st.write("Data Harvesting Active: Building the 'Yoga-Diverse' Dataset.")

col1, col2, col3, col4, col5 = st.columns(5)
metrics = session.get_current_summary()
life_metrics = session.get_lifetime_summary()

col1.metric("Avg Flow", metrics["Avg Flow"])
col2.metric("In The Zone", metrics["Zone Time"])
col3.metric("Locks", metrics["Stability Events"])
col4.metric("Top Pose", metrics["Top Pose"])
col5.metric("Total Practice", life_metrics["Total Time"])

use_vae  = st.sidebar.checkbox("Enable VAE Quality Score", value=True)
use_tts  = st.sidebar.checkbox("Enable Voice Coach", value=True)
use_ar   = st.sidebar.checkbox("Enable Visual Whispers (AR)", value=True)
use_gamification = st.sidebar.checkbox("Enable Stability Engine", value=True)
use_flow = st.sidebar.checkbox("Enable Flow Score", value=True)
use_seq  = st.sidebar.checkbox("Enable Sequencer", value=True)
use_data = st.sidebar.checkbox("Enable Data Harvest", value=True) # NEW
show_dbg = st.sidebar.checkbox("Show debug logs", value=False)
st.sidebar.header("Status / Debug")

if st.sidebar.button("End Session & Save"):
    session.save_session()
    st.sidebar.success("Saved to Vault.")

CLF_PATH   = "zenith_pose_classifier.pkl"
ENC_W_PATH = "zenith_encoder_weights.weights.h5"
DEC_W_PATH = "zenith_decoder_weights.weights.h5"
POSE_NAMES = ["Chair","Downward Dog","Extended Side Angle","High Lunge","Mountain Pose","Plank","Tree","Triangle","Warrior II"]
INT_TO_POSE = {i:n for i,n in enumerate(POSE_NAMES)}

tts_queue = queue.Queue()
def tts_worker():
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        while True:
            text = tts_queue.get()
            if text is None: break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")
            tts_queue.task_done()
    except ImportError:
        print("pyttsx3 not installed. Voice disabled.")
threading.Thread(target=tts_worker, daemon=True).start()

last_spoken_time = 0
DEBOUNCE_SECONDS = 5.0 
active_correction = None 

stability_start_time = None
is_locked = False
STABILITY_THRESHOLD = 3.0
LOCKED_PHRASES = ["Locked.", "Solid.", "Perfect.", "Holding strong.", "That's it."]

prev_landmarks_array = None
flow_history = deque(maxlen=50)
current_flow_score = 100.0
prev_velocity = 0.0

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
    z_mean, _, z = encoder.predict(flat_keypoints, verbose=0)
    recon = decoder.predict(z, verbose=0)
    mse = float(np.mean((flat_keypoints - recon) ** 2))
    q = 100.0 * (1.0 - (mse - low) / (high - low))
    return float(np.clip(q, 0, 100))
vae_pool = cf.ThreadPoolExecutor(max_workers=1)
vae_future = None
last_q = None
alpha = 0.25 

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def landmarks_to_flat(lms):
    arr = np.array([[lm.x,lm.y,lm.z,lm.visibility] for lm in lms.landmark], dtype=np.float32)
    return arr.flatten()[None,:] 

hist = deque(maxlen=15)
i = 0
fps = 0.0
t0 = time.time()
last_label = None
last_ok_ts = 0.0
TTL = 1.5
def majority(dq): return Counter(dq).most_common(1)[0][0] if dq else None

def calculate_flow(curr_flat, prev_flat, dt, prev_v):
    if prev_flat is None or dt <= 0: return 100.0, 0.0
    diff = curr_flat - prev_flat
    dist = np.linalg.norm(diff)
    velocity = dist / dt
    acceleration = abs(velocity - prev_v) / dt
    punishment = acceleration * 15.0 
    score = 100.0 - punishment
    return float(np.clip(score, 0, 100)), velocity

# --- PROCESSING LOOP ---
def process_frame(frame: av.VideoFrame) -> av.VideoFrame:
    global i, fps, t0, last_label, last_ok_ts, vae_future, last_q, last_spoken_time, active_correction
    global stability_start_time, is_locked, STABILITY_THRESHOLD
    global prev_landmarks_array, current_flow_score, prev_velocity, flow_history
    global session, sequencer, ui, harvester
    
    img = frame.to_ndarray(format="bgr24")

    # FPS
    t1 = time.time(); dt = t1 - t0; t0 = t1
    if dt > 0: fps = 0.9*fps + 0.1*(1.0/dt) if fps else (1.0/dt)

    res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Flow
    if use_flow and res.pose_landmarks:
        curr_flat = landmarks_to_flat(res.pose_landmarks)
        if prev_landmarks_array is not None:
             score, velocity = calculate_flow(curr_flat, prev_landmarks_array, dt, prev_velocity)
             alpha_flow = 0.15
             current_flow_score = (alpha_flow * score) + ((1-alpha_flow) * current_flow_score)
             prev_velocity = velocity
        prev_landmarks_array = curr_flat
    
    # Skel
    if res.pose_landmarks:
        mp_draw.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                               mp_draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

    label = None
    q_disp = last_q

    try:
        if res.pose_landmarks and (i % 3 == 0):
            feats = landmarks_to_flat(res.pose_landmarks)

            if clf_ok and pose_classifier is not None:
                pred = pose_classifier.predict(feats)
                label = INT_TO_POSE.get(int(pred[0]), "Unknown")
                hist.append(label)
                label = majority(hist)

                # Sequencer
                if use_seq and sequencer and label:
                    seq_status = sequencer.update(label, is_stable=False) 
                    if seq_status == "Advance":
                        if use_tts: tts_queue.put(f"Good. Next is {sequencer.get_next_goal()}.")

                # Coaching
                if PoseHeuristics and label:
                    lms = {lm_id: [lm.x, lm.y] for lm_id, lm in enumerate(res.pose_landmarks.landmark)}
                    correction = PoseHeuristics.evaluate(label, lms)
                    
                    if correction:
                        active_correction = correction
                        stability_start_time = None
                        is_locked = False
                        
                        text_data = correction['text']
                        hud_text = text_data
                        spoken_text = text_data
                        if isinstance(text_data, tuple):
                            hud_text, spoken_text = text_data
                            active_correction['hud_text'] = hud_text

                        if use_tts:
                            now = time.time()
                            if (now - last_spoken_time) > DEBOUNCE_SECONDS:
                                tts_queue.put(spoken_text)
                                last_spoken_time = now
                    else:
                        active_correction = None
                        if use_gamification:
                            if stability_start_time is None: stability_start_time = time.time()
                            duration = time.time() - stability_start_time
                            if duration > STABILITY_THRESHOLD and not is_locked:
                                is_locked = True
                                session.log_stability_event() 
                                
                                # --- HARVEST ---
                                if use_data and harvester and label:
                                    # Save frame if locked (stable) and high quality
                                    q_val = last_q if last_q is not None else 0
                                    harvester.save_frame(img, label, q_val)
                                    
                                if use_tts: 
                                    reward_phrase = random.choice(LOCKED_PHRASES)
                                    tts_queue.put(reward_phrase)

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
    if (time.time() - last_ok_ts) < TTL: label = label or last_label

    # Session
    is_stable_now = (stability_start_time is not None)
    session.update(label, current_flow_score if use_flow else None, is_stable_now, fps)

    # --- RENDER (ZENITH UI) ---
    if ui:
        if use_gamification and stability_start_time is not None and res.pose_landmarks:
            duration = time.time() - stability_start_time
            progress = min(duration / STABILITY_THRESHOLD, 1.0)
            ui.draw_halo(img, res.pose_landmarks, progress, is_locked)
        
        if use_ar and active_correction:
            if 'vector' in active_correction:
                start_pt, end_pt = active_correction['vector']
                ar_text = active_correction.get('hud_text', active_correction.get('text'))
                if isinstance(ar_text, tuple): ar_text = ar_text[0]
                ui.draw_arrow(img, start_pt, end_pt, text=ar_text)
                
        if use_flow:
            ui.draw_flow_bar(img, current_flow_score)
            
        if use_seq and sequencer:
            ui.draw_sequencer(img, sequencer.get_current_goal(), sequencer.get_next_goal(), sequencer.get_progress())

        ui.draw_hud(img, label, q_disp, fps)
        
        # Draw Harvest Indicator
        if use_data and harvester:
            count = harvester.get_stats()
            cv2.putText(img, f"DATA: {count}", (10, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="zenith-mvp",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=process_frame,
    async_processing=True,
    rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302","stun:stun1.l.google.com:19302"]}]},
    media_stream_constraints={"video":{"width":{"ideal":640},"height":{"ideal":480},"frameRate":{"ideal":30}}, "audio":False},
)
