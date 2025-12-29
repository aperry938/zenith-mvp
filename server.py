import cv2
import numpy as np
import asyncio
import base64
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
from pose_sequencer import PoseSequencer

# --- SERVER SETUP ---
app = FastAPI(title="Zenith AI Gateway", version="Cycle 41 (Reflection)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SINGLETONS ---
brain_instance = ZenithBrain()
sage_instance = VisionClient()
session_mgr = SessionManager()
harvester = DataHarvester()
sequencer = PoseSequencer()

latest_frame_cache = None

@app.get("/")
async def root():
    return {"status": "Zenith AI Gateway Online", "cycle": 41}

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client Connected to Zenith Stream")
    
    global latest_frame_cache
    
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
                    
                    # Poll Result
                    result = None
                    for _ in range(5): 
                        res = brain_instance.get_latest_result()
                        if res:
                            result = res
                            break
                        await asyncio.sleep(0.005)
                    
                    response = {}
                    if result:
                        label = result.get("label")
                        response["label"] = label
                        response["flow"] = result.get("flow_score")
                        response["velocity"] = result.get("velocity")
                        response["q"] = result.get("vae_q")
                        
                        # Pack Landmarks
                        if result.get("pose_landmarks"):
                           lms =  result["pose_landmarks"].landmark
                           response["landmarks"] = [
                               {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility} 
                               for lm in lms
                           ]
                           
                        # Pack Ghost
                        if result.get("vae_recon") is not None:
                            response["ghost"] = result["vae_recon"].flatten().tolist()

                        # --- LOGIC UPDATES ---
                        
                        # 1. Sequencer Update
                        # We need 'is_stable' logic, usually derived from velocity/flow
                        is_stable = False
                        if result.get("flow_score") is not None:
                            # Heuristic: High flow score (>90) usually means stable/smooth
                            # Or low velocity. Let's use velocity if available, else flow.
                            # Brain output actually has velocity.
                            vel = result.get("velocity", 1.0)
                            if vel < 0.05: # Very low motion
                                is_stable = True
                                
                        seq_status = sequencer.update(label, is_stable)
                        
                        response["sequence_state"] = {
                            "target_pose": sequencer.get_current_goal(),
                            "next_pose": sequencer.get_next_goal(),
                            "status": seq_status, # "Holding", "Advance"
                            "progress": sequencer.get_progress()
                        }
                        
                        # 2. Voice/Announcement Check
                        if sequencer.has_announcement():
                            announcement = sequencer.get_announcement()
                            if announcement:
                                response["voice_message"] = announcement

                        # 3. Oracle Check (Proactive Analysis)
                        if sequencer.check_oracle_trigger():
                             print("Meaningful Moment Detected. Triggering Oracle...")
                             if latest_frame_cache is not None:
                                 asyncio.create_task(run_oracle_analysis(websocket, latest_frame_cache))

                        # 4. Session Recording
                        if session_mgr.is_recording and label and result.get("flow_score") is not None:
                             session_mgr.add_frame_data(
                                 timestamp=ts,
                                 pose_label=label,
                                 flow_score=result["flow_score"],
                                 vae_q=result.get("vae_q", 0)
                             )
                        
                        # 5. Data Harvesting
                        if harvester.is_active and result.get("pose_landmarks"):
                             harvester.process_frame(frame, result["pose_landmarks"])

                    # Send Status Flags back to Client
                    response["is_recording"] = session_mgr.is_recording
                    response["is_harvesting"] = harvester.is_active

                    await websocket.send_text(json.dumps(response))

            elif "text" in message:
                try:
                    cmd = json.loads(message["text"])
                    action = cmd.get("action")
                    
                    if action == "analyze":
                        # Manual Trigger
                        print("Analysis Requested (Manual)...")
                        if latest_frame_cache is not None:
                            asyncio.create_task(run_oracle_analysis(websocket, latest_frame_cache))
                                
                    elif action == "toggle_record":
                        if session_mgr.is_recording:
                            session_mgr.stop_recording()
                        else:
                            session_mgr.start_recording()
                            
                    elif action == "toggle_harvest":
                        harvester.toggle()
                        print(f"Harvesting Toggled to {harvester.is_active}")
                    
                    elif action == "end_session":
                         print("Ending Session...")
                         if session_mgr.is_recording:
                             session_mgr.stop_recording()
                         
                         # Generate Report
                         stats = session_mgr.get_current_summary()
                         await websocket.send_text(json.dumps({
                             "type": "session_report",
                             "stats": stats
                         }))

                except Exception as e:
                    print(f"Command Error: {e}")
            
    except WebSocketDisconnect:
        print("Client Disconnected")
        if session_mgr.is_recording:
             session_mgr.stop_recording()
    except Exception:
        pass 

async def run_oracle_analysis(websocket: WebSocket, frame):
    """
    Runs the vision analysis in a separate thread/executor to avoid blocking the WS loop.
    Sends the result directly to the socket.
    """
    try:
        success, buffer = cv2.imencode('.jpg', frame)
        if success:
             loop = asyncio.get_running_loop()
             # Note: analyze_frame might be blocking, so we run in executor
             advice_text = await loop.run_in_executor(None, sage_instance.analyze_frame, frame)
             
             # Send as 'advice' type (which triggers TTS on client)
             await websocket.send_text(json.dumps({
                 "type": "advice",
                 "text": f"Oracle Insight: {advice_text}" 
             }))
    except Exception as e:
        print(f"Oracle Error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
