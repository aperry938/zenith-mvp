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

from zenith_brain import ZenithBrain
from vision_client import VisionClient
from session_manager import SessionManager
from data_harvester import DataHarvester

# --- SERVER SETUP ---
app = FastAPI(title="ZENith API", version="2.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

@app.get("/")
async def root():
    return {"status": "ZENith API Online", "version": "2.2"}

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client Connected to Zenith Stream")
    
    global latest_frame_cache, latest_landmarks_cache

    # Per-connection auto-analysis state
    auto_analysis = {"last_label": None, "hold_start": 0.0, "triggered": False}
    
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
                                print(f"Auto-analysis triggered (held {label} for 5s)")
                                if latest_frame_cache is not None:
                                    asyncio.create_task(run_coach_analysis(websocket, latest_frame_cache))
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
                                 fps=30
                             )

                        # 3. Data Harvesting
                        if getattr(harvester, 'harvesting', False) and label:
                             harvester.save_frame(frame, label, result.get("vae_q", 0))

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
                        print("Analysis Requested (Manual)...")
                        if latest_frame_cache is not None:
                            asyncio.create_task(run_coach_analysis(websocket, latest_frame_cache))
                                
                    elif action == "toggle_record":
                        if not session_mgr.recording:
                            session_mgr.reset()
                            session_mgr.recording = True
                            print("Recording started (session reset)")
                        else:
                            session_mgr.recording = False
                            print("Recording paused")
                            
                    elif action == "toggle_harvest":
                        harvester.harvesting = not getattr(harvester, 'harvesting', False)
                        print(f"Harvesting: {harvester.harvesting}")
                    
                    elif action == "end_session":
                         print("Ending Session...")
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
                    print(f"Command Error: {e}")
            
    except WebSocketDisconnect:
        print("Client Disconnected")
    except Exception as e:
        import traceback
        print(f"WebSocket Fatal Error: {e}")
        traceback.print_exc()

async def run_coach_analysis(websocket: WebSocket, frame):
    """
    Runs the vision analysis in a separate thread/executor to avoid blocking the WS loop.
    Sends the result directly to the socket.
    """
    try:
        success, buffer = cv2.imencode('.jpg', frame)
        if success:
             loop = asyncio.get_running_loop()
             # Note: analyze_frame might be blocking, so we run in executor
             advice_text = await loop.run_in_executor(None, vision_client.analyze_frame, frame)
             
             # Send as 'advice' type (which triggers TTS on client)
             await websocket.send_text(json.dumps({
                 "type": "advice",
                 "text": advice_text 
             }))
    except Exception as e:
        print(f"Coach Analysis Error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
