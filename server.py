import cv2
import numpy as np
import asyncio
import base64
import json
import time

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("FastAPI/Uvicorn not installed. Please install with: pip install fastapi uvicorn")
    sys.exit(1)

from zenith_brain import ZenithBrain

# --- SERVER SETUP ---
app = FastAPI(title="Zenith AI Gateway", version="Cycle 24")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SINGLETON BRAIN ---
# In a real production environment, we might manage sessions here.
# For the MVP/Personal Coach, a single brain instance suffices.
brain_instance = ZenithBrain()

@app.get("/")
async def root():
    return {"status": "Zenith AI Gateway Online", "cycle": 24}

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client Connected to Zenith Stream")
    
    try:
        while True:
            # 1. Receive Frame (expecting bytes)
            data = await websocket.receive_bytes()
            
            # 2. Decode
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
                
            # 3. Process via Brain (Async/Threaded)
            # We inject timestamp for flow calc
            ts = time.time()
            brain_instance.process_async(frame, ts)
            
            # 4. Wait for Result (Non-blocking poll is tricky here, 
            # ideally Brain would use asyncio queues or we poll)
            result = None
            
            # Simple polling with timeout to keep WS alive
            # In a robust system, Brain would be fully async/await compatible.
            for _ in range(10): # Try for ~100ms
                res = brain_instance.get_latest_result()
                if res:
                    result = res
                    break
                await asyncio.sleep(0.01)
            
            # 5. Pack Response
            response = {}
            if result:
                response["label"] = result.get("label")
                response["flow"] = result.get("flow_score")
                response["velocity"] = result.get("velocity")
                response["q"] = result.get("vae_q")
                
                # We could encode landmarks here too, but for speed keeping it light
                if result.get("pose_landmarks"):
                    # Basic landmark extraction for frontend rendering?
                    # For now just send high-level metrics
                    pass
            
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        print("Client Disconnected")
    except Exception as e:
        print(f"WS Error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
