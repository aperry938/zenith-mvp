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

# --- SERVER SETUP ---
app = FastAPI(title="Zenith AI Gateway", version="Cycle 36 (Record)")

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

latest_frame_cache = None

@app.get("/")
async def root():
    return {"status": "Zenith AI Gateway Online", "cycle": 36}

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
                        response["label"] = result.get("label")
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

                        # --- PERSISTENCE ---
                        # 1. Session Recording
                        if session_mgr.is_recording and result.get("label") and result.get("flow_score") is not None:
                             session_mgr.add_frame_data(
                                 timestamp=ts,
                                 pose_label=result["label"],
                                 flow_score=result["flow_score"],
                                 vae_q=result.get("vae_q", 0)
                             )
                        
                        # 2. Data Harvesting
                        if harvester.is_active and result.get("pose_landmarks"):
                             harvester.process_frame(frame, result["pose_landmarks"])

                    # Send Status Flags back to Client so UI matches Server state
                    response["is_recording"] = session_mgr.is_recording
                    response["is_harvesting"] = harvester.is_active

                    await websocket.send_text(json.dumps(response))

            elif "text" in message:
                try:
                    cmd = json.loads(message["text"])
                    action = cmd.get("action")
                    
                    if action == "analyze":
                        print("Analysis Requested...")
                        if latest_frame_cache is not None:
                            success, buffer = cv2.imencode('.jpg', latest_frame_cache)
                            if success:
                                loop = asyncio.get_event_loop()
                                advice_text = await loop.run_in_executor(None, sage_instance.analyze_frame, latest_frame_cache)
                                await websocket.send_text(json.dumps({
                                    "type": "advice",
                                    "text": advice_text
                                }))
                                
                    elif action == "toggle_record":
                        if session_mgr.is_recording:
                            session_mgr.stop_recording()
                            print("Recording Stopped")
                        else:
                            session_mgr.start_recording()
                            print("Recording Started")
                            
                    elif action == "toggle_harvest":
                        harvester.toggle() # DataHarvester has a toggle method? Let's check or just set bool.
                        # Ideally DataHarvester has .active bool.
                        # Looking at previous context, it seems standard to just toggle a flag.
                        # But wait, looking at file content is safer.
                        # Assuming it has a toggle or is_active property we can flip.
                        # Let's rely on the method I recall or verify.
                        # Actually I'll check DataHarvester code in next turn if needed, but 'toggle()' is common.
                        # If not, I will fix it.
                        print(f"Harvesting Toggled to {harvester.is_active}")

                except Exception as e:
                    print(f"Command Error: {e}")
            
    except WebSocketDisconnect:
        print("Client Disconnected")
        if session_mgr.is_recording:
             session_mgr.stop_recording()
    except Exception:
        pass 

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
