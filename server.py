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

# --- SERVER SETUP ---
app = FastAPI(title="Zenith AI Gateway", version="Cycle 31 (Wisdom)")

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
latest_frame_cache = None

@app.get("/")
async def root():
    return {"status": "Zenith AI Gateway Online", "cycle": 31}

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
                    
                    await websocket.send_text(json.dumps(response))

            elif "text" in message:
                # Handle JSON commands (e.g. { "action": "analyze" })
                try:
                    cmd = json.loads(message["text"])
                    if cmd.get("action") == "analyze":
                        print("Analysis Requested...")
                        if latest_frame_cache is not None:
                            # 1. Encode frame for Sage
                            success, buffer = cv2.imencode('.jpg', latest_frame_cache)
                            if success:
                                # 2. Call Sage (Blocking/Sync for now, ideally threaded)
                                # We run in executor to avoid blocking event loop
                                loop = asyncio.get_event_loop()
                                advice_text = await loop.run_in_executor(None, sage_instance.analyze_frame, latest_frame_cache)
                                
                                # 3. Send Advice back
                                await websocket.send_text(json.dumps({
                                    "type": "advice",
                                    "text": advice_text
                                }))
                        else:
                            await websocket.send_text(json.dumps({
                                "type": "advice",
                                "text": "No frame available to analyze."
                            }))
                except Exception as e:
                    print(f"Command Error: {e}")
            
    except WebSocketDisconnect:
        print("Client Disconnected")
    except Exception as e:
        print(f"WS Error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
