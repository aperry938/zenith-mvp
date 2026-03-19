import cv2
import numpy as np
import asyncio
import json
import time
import sys

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("FastAPI/Uvicorn not installed. Please install with: pip install fastapi uvicorn")
    sys.exit(1)

from config import setup_logging, CORS_ORIGINS, WS_HOST, WS_PORT
from zenith_brain import ZenithBrain
from vision_client import VisionClient
from session_manager import SessionManager
from data_harvester import DataHarvester
from pose_foundations import PoseHeuristics, VALID_PERSONAS
from pose_sequencer import PoseSequencer
from streak_tracker import StreakTracker
from progress_tracker import ProgressTracker

logger = setup_logging("zenith.server")

# --- SERVER SETUP ---
app = FastAPI(title="ZENith API", version="2.6")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SINGLETONS ---
brain_instance = ZenithBrain()
vision_client = VisionClient()
session_mgr = SessionManager()
harvester = DataHarvester()

latest_frame_cache = None
latest_landmarks_cache = None
# Note: auto_analysis state is per-connection (see websocket_endpoint)

@app.on_event("startup")
async def startup():
    logger.info(f"ZENith API v2.6 starting on {WS_HOST}:{WS_PORT}")
    logger.info(f"CORS origins: {CORS_ORIGINS}")
    logger.info(f"Brain models: clf={'OK' if brain_instance.clf_ok else 'MISSING'}, vae={'OK' if brain_instance.vae_ok else 'MISSING'}")

@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down — saving session data")
    brain_instance.running = False
    if session_mgr.recording:
        session_mgr.save_session()

@app.get("/")
async def root():
    return {"status": "ZENith API Online", "version": "2.5"}

@app.get("/api/sessions")
async def get_sessions():
    return SessionManager.load_sessions()

@app.get("/api/streaks")
async def get_streaks():
    tracker = StreakTracker()
    return tracker.get_stats()

@app.get("/api/progress")
async def get_progress():
    tracker = ProgressTracker()
    return tracker.get_progress()

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")
    
    global latest_frame_cache, latest_landmarks_cache

    # Per-connection auto-analysis state
    auto_analysis = {"last_label": None, "hold_start": 0.0, "triggered": False}
    # Per-connection heuristic debounce state
    heuristic_state = {"last_text": None, "last_speak_time": 0.0}
    # Per-connection positive feedback debounce (less frequent)
    positive_state = {"last_pose": None, "last_time": 0.0}
    # Per-connection pose sequencer (None until started)
    sequencer = None
    # Per-connection intensity level (1=gentle, 2=standard, 3=intense)
    intensity = 2
    # Per-connection coach persona
    persona = 'default'
    
    try:
        while True:
            # 1. Receive Data (Bytes or Text JSON)
            message = await websocket.receive()
            
            if "bytes" in message:
                data = message["bytes"]
                # Decode
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    latest_frame_cache = frame
                    ts = time.time()
                    brain_instance.process_async(frame, ts)
                    
                    # Poll Result (wait up to 100ms for brain to finish)
                    result = None
                    for _ in range(20):
                        res = brain_instance.get_latest_result()
                        if res:
                            result = res
                            break
                        await asyncio.sleep(0.005)
                    
                    response = {"has_result": False}
                    if result:
                        label = result.get("label")
                        response["has_result"] = True
                        response["label"] = label
                        response["confidence"] = result.get("confidence")
                        response["flow"] = result.get("flow_score")
                        response["velocity"] = result.get("velocity")
                        response["q"] = result.get("vae_q")
                        # Form assessment based on bio quality
                        bio_q = result.get("bio_quality")
                        if bio_q is not None and label:
                            response["form_assessment"] = "Correct" if bio_q >= 80 else "Incorrect"
                        
                        # Pack Landmarks + cache for persistence
                        if result.get("pose_landmarks"):
                           latest_landmarks_cache = result["pose_landmarks"]
                           lms =  result["pose_landmarks"].landmark
                           response["landmarks"] = [
                               {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                               for lm in lms
                           ]
                           
                        # Pack Ghost
                        if result.get("vae_recon") is not None:
                            response["ghost"] = result["vae_recon"].flatten().tolist()

                        # Pack Biomechanical Features
                        if result.get("bio_features") is not None:
                            response["bio_features"] = result["bio_features"].flatten().tolist()
                        if result.get("bio_quality") is not None:
                            response["bio_quality"] = float(result["bio_quality"])
                        if result.get("bio_deviations") is not None:
                            # Convert numpy types to native Python for JSON serialization
                            devs = []
                            for d in result["bio_deviations"][:3]:
                                devs.append({
                                    "feature": str(d["feature"]),
                                    "feature_idx": int(d["feature_idx"]),
                                    "value": float(d["value"]),
                                    "ideal_lo": float(d["ideal_lo"]),
                                    "ideal_hi": float(d["ideal_hi"]),
                                    "deviation": float(d["deviation"]),
                                    "direction": str(d["direction"]),
                                })
                            response["bio_deviations"] = devs

                        # --- HEURISTIC COACHING ---
                        if label and result.get("pose_landmarks"):
                            lms = result["pose_landmarks"].landmark
                            lm_dict = {i: [lms[i].x, lms[i].y] for i in range(len(lms))}
                            correction = PoseHeuristics.evaluate(label, lm_dict, intensity, persona)
                            if correction:
                                is_positive = correction.get("positive", False)
                                hud_text, spoken_text = correction["text"]
                                now = time.time()

                                if is_positive:
                                    # Positive feedback: once per pose, max every 15s
                                    should_speak = (
                                        label != positive_state["last_pose"]
                                        or (now - positive_state["last_time"]) > 15.0
                                    )
                                    if should_speak:
                                        response["heuristic"] = {
                                            "hud": hud_text,
                                            "spoken": spoken_text,
                                            "speak": True,
                                            "positive": True,
                                        }
                                        positive_state["last_pose"] = label
                                        positive_state["last_time"] = now
                                else:
                                    # Correction: debounce by text change or 5s
                                    should_speak = (
                                        hud_text != heuristic_state["last_text"]
                                        or (now - heuristic_state["last_speak_time"]) > 5.0
                                    )
                                    vec_data = {}
                                    if correction.get("vector"):
                                        vec_data = {
                                            "vector": {
                                                "start": list(correction["vector"][0]),
                                                "end": list(correction["vector"][1]),
                                            },
                                            "color": list(correction["color"]),
                                        }
                                    response["heuristic"] = {
                                        "hud": hud_text,
                                        "spoken": spoken_text,
                                        "speak": should_speak,
                                        **vec_data,
                                    }
                                    if should_speak:
                                        heuristic_state["last_text"] = hud_text
                                        heuristic_state["last_speak_time"] = now

                        # --- LOGIC UPDATES ---

                        is_stable = False
                        if result.get("flow_score") is not None:
                            vel = result.get("velocity", 1.0)
                            if vel < 0.05:
                                is_stable = True

                        # 1. Auto-analysis: trigger coach feedback after ~5s hold
                        if label:
                            if label != auto_analysis["last_label"]:
                                auto_analysis["last_label"] = label
                                auto_analysis["hold_start"] = time.time()
                                auto_analysis["triggered"] = False
                            elif not auto_analysis["triggered"] and (time.time() - auto_analysis["hold_start"]) > 5.0:
                                auto_analysis["triggered"] = True
                                logger.info(f"Auto-analysis triggered (held {label} for 5s)")
                                if latest_frame_cache is not None:
                                    await websocket.send_text(json.dumps({"type": "analysis_started"}))
                                    asyncio.create_task(run_coach_analysis(
                                        websocket, latest_frame_cache,
                                        pose_label=label,
                                        bio_quality=result.get("bio_quality"),
                                        deviations=response.get("bio_deviations"),
                                    ))
                        else:
                            auto_analysis["last_label"] = None
                            auto_analysis["hold_start"] = 0.0
                            auto_analysis["triggered"] = False

                        # 2. Session Tracking (only when recording)
                        if session_mgr.recording and label and result.get("flow_score") is not None:
                             session_mgr.update(
                                 pose_label=label,
                                 flow_score=result["flow_score"],
                                 is_stable=is_stable,
                                 fps=30,
                                 bio_quality=result.get("bio_quality"),
                             )
                             if response.get("heuristic") and response["heuristic"].get("speak"):
                                 session_mgr.log_heuristic_correction()

                        # 3. Data Harvesting
                        if getattr(harvester, 'harvesting', False) and label:
                             harvester.save_frame(frame, label, result.get("vae_q", 0))

                        # 4. Pose Sequencer
                        if sequencer and not sequencer.completed and label:
                            sequencer.update(label, is_stable)
                            # Oracle trigger: auto-analyze during sequence
                            if sequencer.check_oracle_trigger() and latest_frame_cache is not None:
                                await websocket.send_text(json.dumps({"type": "analysis_started"}))
                                asyncio.create_task(run_coach_analysis(
                                    websocket, latest_frame_cache,
                                    pose_label=label,
                                    bio_quality=result.get("bio_quality"),
                                    deviations=response.get("bio_deviations"),
                                ))

                    # Pack sequencer state
                    if sequencer:
                        seq_data = {
                            "name": sequencer.sequence_name,
                            "current_goal": sequencer.get_current_goal(),
                            "next_goal": sequencer.get_next_goal(),
                            "progress": sequencer.get_progress(),
                            "completed": sequencer.completed,
                            "hold_seconds": round(sequencer.get_hold_elapsed(), 1),
                            "hold_target": sequencer.HOLD_DURATION,
                            "breath_cue": sequencer.get_breath_cue(),
                        }
                        if sequencer.has_announcement():
                            seq_data["announcement"] = sequencer.get_announcement()
                        response["sequence"] = seq_data

                    # Always pack cached landmarks so skeleton persists between brain polls
                    if "landmarks" not in response and latest_landmarks_cache is not None:
                        lms = latest_landmarks_cache.landmark
                        response["landmarks"] = [
                            {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                            for lm in lms
                        ]

                    # Send Status Flags back to Client
                    response["is_recording"] = session_mgr.recording
                    response["is_harvesting"] = getattr(harvester, 'harvesting', False)

                    await websocket.send_text(json.dumps(response))

            elif "text" in message:
                try:
                    cmd = json.loads(message["text"])
                    action = cmd.get("action")
                    
                    if action == "analyze":
                        # Manual Trigger
                        logger.info("Analysis requested (manual)")
                        if latest_frame_cache is not None:
                            await websocket.send_text(json.dumps({"type": "analysis_started"}))
                            asyncio.create_task(run_coach_analysis(websocket, latest_frame_cache))
                                
                    elif action == "toggle_record":
                        if not session_mgr.recording:
                            session_mgr.reset()
                            session_mgr.recording = True
                            logger.info("Recording started")
                        else:
                            session_mgr.recording = False
                            logger.info("Recording paused")
                            
                    elif action == "toggle_harvest":
                        harvester.harvesting = not getattr(harvester, 'harvesting', False)
                        logger.info(f"Harvesting: {harvester.harvesting}")
                    
                    elif action == "start_sequence":
                        seq_key = cmd.get("sequence", "strength_flow")
                        sanskrit = cmd.get("sanskrit_count", False)
                        sequencer = PoseSequencer(sequence_key=seq_key, sanskrit_count=sanskrit)
                        logger.info(f"Sequence started: {sequencer.sequence_name}")

                    elif action == "stop_sequence":
                        sequencer = None
                        logger.info("Pose sequence stopped")

                    elif action == "set_intensity":
                        intensity = max(1, min(3, int(cmd.get("intensity", 2))))
                        logger.info(f"Intensity set to {intensity}")

                    elif action == "set_persona":
                        requested = cmd.get("persona", "default")
                        if requested in VALID_PERSONAS:
                            persona = requested
                            logger.info(f"Persona set to '{persona}'")
                        else:
                            logger.warning(f"Invalid persona '{requested}', keeping '{persona}'")

                    elif action == "end_session":
                         logger.info("Ending session")
                         session_mgr.recording = False

                         # Generate Report
                         stats = session_mgr.get_current_summary()
                         session_mgr.save_session()
                         await websocket.send_text(json.dumps({
                             "type": "session_report",
                             "stats": stats
                         }))
                         # Reset session for next round
                         session_mgr.reset()

                except Exception as e:
                    logger.error(f"Command error: {e}")
            
    except WebSocketDisconnect:
        logger.info("Client disconnected")
        if session_mgr.recording:
            session_mgr.recording = False
            session_mgr.save_session()
            logger.info("Session auto-saved on disconnect")
    except Exception as e:
        import traceback
        logger.error(f"WebSocket fatal error: {e}")
        traceback.print_exc()

async def run_coach_analysis(websocket: WebSocket, frame, pose_label=None, bio_quality=None, deviations=None):
    """
    Runs the vision analysis in a separate thread/executor to avoid blocking the WS loop.
    Sends the result directly to the socket.
    """
    try:
        loop = asyncio.get_running_loop()
        advice_text, source = await loop.run_in_executor(
            None, vision_client.analyze_frame, frame, pose_label, bio_quality, deviations
        )

        await websocket.send_text(json.dumps({
            "type": "advice",
            "text": advice_text,
            "source": source,
        }))
    except Exception as e:
        logger.error(f"Coach analysis error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host=WS_HOST, port=WS_PORT)
