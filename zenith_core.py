import os, time, sys, random
from collections import deque, Counter
import threading, queue

import cv2
import numpy as np
import av
import mediapipe as mp

# --- INTERNAL IMPORTS ---
try:
    from pose_foundations import PoseHeuristics
    from pose_sequencer import PoseSequencer
    from ui_overlay import ZenithUI
    from session_manager import SessionManager
    from data_harvester import DataHarvester
    from vision_client import VisionClient
    from zenith_brain import ZenithBrain
except ImportError:
    print("Warning: Internal modules not found.")


class ZenithCore:
    def __init__(self):
        self.stats_file = "zenith_stats.json"
        
        # --- STATE ---
        self.fps = 0.0
        self.t0 = time.time()
        
        self.last_label = None
        self.last_ok_ts = 0.0
        self.TTL = 1.5
        
        self.active_correction = None
        self.stability_start_time = None
        self.is_locked = False
        self.STABILITY_THRESHOLD = 3.0
        self.LOCKED_PHRASES = ["Locked.", "Solid.", "Perfect.", "Holding strong.", "That's it."]
        
        # flow (Brain manages raw calc, Core manages TTS trigger)
        self.current_flow_score = 100.0
        self.last_flow_msg_time = 0
        self.FLOW_MSG_DEBOUNCE = 10.0
        
        # tts
        self.tts_queue = queue.Queue()
        self.last_spoken_time = 0
        self.DEBOUNCE_SECONDS = 5.0
        self.start_tts_worker()
        
        # --- BRAIN (ASYNC) ---
        self.brain = ZenithBrain()
        self.last_brain_result = None # Cache
        
        # dream state
        self.dream_z_current = None
        self.dream_z_target = None
        self.dream_t = 0.0
        self.DREAM_SPEED = 0.02
        
        # modules
        self.ui = ZenithUI()
        self.session = SessionManager()
        self.sequencer = PoseSequencer()
        self.harvester = DataHarvester()
        self.sage = VisionClient()
        
        self.latest_frame = None
        self.last_recon = None
        self.last_q = None
        
        # Hist for smoothing label
        self.hist = deque(maxlen=5)

    def start_tts_worker(self):
        def tts_worker():
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', 160)
                while True:
                    text = self.tts_queue.get()
                    if text is None: break
                    try:
                        engine.say(text)
                        engine.runAndWait()
                    except Exception as e:
                        print(f"TTS Error: {e}")
                    self.tts_queue.task_done()
            except ImportError:
                 pass
        threading.Thread(target=tts_worker, daemon=True).start()

    def majority(self, dq):
        return Counter(dq).most_common(1)[0][0] if dq else None

    # --- CYCLE 21: THE MIRROR ---
    def apply_cinematic_look(self, img):
        img = cv2.convertScaleAbs(img, alpha=1.1, beta=0) 
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # s = cv2.multiply(s, 1.2) # Removed saturation overkill
        # s = np.clip(s, 0, 255).astype(hsv.dtype)
        # hsv = cv2.merge([h, s, v])
        # img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img

    def process_frame(self, frame_obj, options):
        img = frame_obj.to_ndarray(format="bgr24")
        
        # Cinematic Look
        img = self.apply_cinematic_look(img)
        
        if options.get('use_tts'):
            self.latest_frame = img.copy()

        t1 = time.time(); dt = t1 - self.t0; self.t0 = t1
        if dt > 0: self.fps = 0.9*self.fps + 0.1*(1.0/dt) if self.fps else (1.0/dt)

        # Send to Brain (Async) with timestamp
        self.brain.process_async(img.copy(), t1)
        
        # Check for new result
        new_res = self.brain.get_latest_result()
        if new_res:
             self.last_brain_result = new_res
             
        # Use Cached Result (or None)
        res = self.last_brain_result
        
        # --- LOGIC HANDLING (Using Cached/New Result) ---
        label = None
        q_disp = self.last_q
        pose_landmarks = None
        velocity = 0.0
        
        if res:
            pose_landmarks = res['pose_landmarks']
            raw_label = res['label']
            self.last_recon = res.get('vae_recon')
            q_disp = res.get('vae_q')
            if q_disp: self.last_q = q_disp
            
            # Flow from Brain
            if res.get('flow_score'):
                self.current_flow_score = res['flow_score']
            if res.get('velocity'):
                velocity = res['velocity']
            
            if raw_label:
                self.hist.append(raw_label)
                label = self.majority(self.hist)

            if label:
                self.last_label = label
                self.last_ok_ts = time.time()
                
                # Sequencer update
                if options.get('use_seq') and self.sequencer:
                    seq_status = self.sequencer.update(label, is_stable=False)
                    if seq_status == "Advance" and options.get('use_tts'):
                         self.tts_queue.put(f"Good. Next is {self.sequencer.get_next_goal()}.")
                
                # Heuristics
                if PoseHeuristics and pose_landmarks:
                     lms = {lm_id: [lm.x, lm.y] for lm_id, lm in enumerate(pose_landmarks.landmark)}
                     correction = PoseHeuristics.evaluate(label, lms)
                     
                     if correction:
                         self.active_correction = correction
                         self.stability_start_time = None
                         self.is_locked = False
                         text_data = correction.get('text', "")
                         if isinstance(text_data, tuple): text_data = text_data[1]
                         if options.get('use_tts') and (time.time() - self.last_spoken_time > self.DEBOUNCE_SECONDS):
                             self.tts_queue.put(text_data)
                             self.last_spoken_time = time.time()
                     else:
                         self.active_correction = None
                         if options.get('use_gamification'):
                             if self.stability_start_time is None: self.stability_start_time = time.time()
                             if (time.time() - self.stability_start_time) > self.STABILITY_THRESHOLD and not self.is_locked:
                                 self.is_locked = True
                                 self.session.log_stability_event()
                                 if options.get('use_data') and self.harvester:
                                     self.harvester.save_frame(img, label, self.last_q or 0)
                                 if options.get('use_tts'):
                                     self.tts_queue.put(random.choice(self.LOCKED_PHRASES))
            
            # Flow TTS Logic (Brain calculated score, Core decides to speak)
            if options.get('use_flow') and options.get('use_tts') and (time.time() - self.last_flow_msg_time > self.FLOW_MSG_DEBOUNCE):
                if self.current_flow_score < 40:
                    self.tts_queue.put("Smooth it out.")
                    self.last_flow_msg_time = time.time()
                elif self.current_flow_score > 90 and velocity > 0.1:
                    self.tts_queue.put("Excellent flow.")
                    self.last_flow_msg_time = time.time()

            # Debug Draw
            if pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(img, pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                       mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                       mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))


        if (time.time() - self.last_ok_ts) < self.TTL: label = label or self.last_label

        is_stable_now = (self.stability_start_time is not None)
        self.session.update(label, self.current_flow_score if options.get('use_flow') else None, is_stable_now, self.fps)

        # --- DRAWING (UI) ---
        if self.ui:
            if options.get('use_grid'): self.ui.draw_grid(img)
            
            # Smart Ghost Alpha
            ghost_alpha = 0.3
            if q_disp is not None:
                if q_disp > 85: ghost_alpha = 0.1
                elif q_disp < 60: ghost_alpha = 0.5
            
            if options.get('use_ghost') and self.last_recon is not None:
                self.ui.draw_ghost(img, self.last_recon, alpha=ghost_alpha)

            if options.get('use_gamification') and self.stability_start_time is not None and pose_landmarks:
                duration = time.time() - self.stability_start_time
                progress = min(duration / self.STABILITY_THRESHOLD, 1.0)
                self.ui.draw_halo(img, pose_landmarks, progress, self.is_locked)

            if options.get('use_ar') and self.active_correction and 'vector' in self.active_correction:
                s, e = self.active_correction['vector']
                txt = self.active_correction.get('hud_text', "")
                if isinstance(txt, tuple): txt = txt[0]
                self.ui.draw_arrow(img, s, e, text=txt)

            if options.get('use_flow'):
                self.ui.draw_flow_bar(img, self.current_flow_score)

            if options.get('use_seq') and self.sequencer:
                self.ui.draw_sequencer(img, self.sequencer.get_current_goal(), self.sequencer.get_next_goal(), self.sequencer.get_progress())

            self.ui.draw_hud(img, label, q_disp, self.fps)
            
            if options.get('use_data') and self.harvester:
                 count = self.harvester.get_stats()
                 cv2.putText(img, f"DATA: {count}", (10, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")
