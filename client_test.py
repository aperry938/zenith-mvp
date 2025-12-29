import cv2
import asyncio
import websockets
import json
import time
import sys

# Usage: python client_test.py [image_path]
# If no image path provided, creates a blank test image.

async def test_zenith_gateway(image_path=None):
    uri = "ws://localhost:8000/ws/stream"
    print(f"Attempting connection to {uri}...")
    
    # 1. Prepare Image
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Could not load image.")
            return
    else:
        print("No image provided, generating blank test frame...")
        img = 255 * np.ones((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "TEST FRAME", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # Encode to JPG Bytes
    _, buffer = cv2.imencode('.jpg', img)
    img_bytes = buffer.tobytes()
    print(f"Image ready: {len(img_bytes)} bytes")

    try:
        async with websockets.connect(uri) as websocket:
            print("Connected! Sending frame...")
            
            # 2. Send Frame
            start_t = time.time()
            await websocket.send(img_bytes)
            
            # 3. Receive Response
            response_text = await websocket.recv()
            dt = time.time() - start_t
            
            print(f"\n--- SERVER RESPONSE ({dt*1000:.1f}ms) ---")
            data = json.loads(response_text)
            print(json.dumps(data, indent=2))
            print("---------------------------------------")
            
    except Exception as e:
        print(f"Connection Failed: {e}")
        print("Make sure server.py is running! (uvicorn server:app --reload)")

if __name__ == "__main__":
    try:
        import numpy as np
        img_arg = sys.argv[1] if len(sys.argv) > 1 else None
        asyncio.run(test_zenith_gateway(img_arg))
    except ImportError:
        print("Missing dependencies. Run: pip install websockets opencv-python numpy")
