import os
import time
import random
import cv2
from PIL import Image

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

class VisionClient:
    """
    Interface for Multimodal LLM Analysis (The Sage).
    Connects to Gemini API if key is present, else falls back to mock.
    """
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.model = None
        self.use_real_api = False
        
        self.mock_responses = [
            "Analysis: Your spinal alignment improves when you engage your core. Try to lengthen your neck.",
            "Analysis: Excellent stability in the lower body. Watch out for hyperextension in the elbows.",
            "Analysis: Good flow. Your transition into Warrior II was smooth, but check your front knee tracking.",
            "Analysis: You are holding tension in your shoulders. Exhale and let them drop.",
            "Analysis: Form looks solid. Try holding the pose for 5 more seconds to build endurance."
        ]
        
        if HAS_GENAI and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.use_real_api = True
                print("VisionClient: Connected to Gemini API.")
            except Exception as e:
                print(f"VisionClient Error: {e}")

    def analyze_frame(self, img_array):
        """
        Sends frame to Gemini (or Mock).
        Returns a concise coaching string.
        """
        if self.use_real_api and self.model:
            try:
                # Convert BGR (OpenCV) to RGB (PIL)
                rgb_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)
                
                # Prompt
                prompt = "You are an expert yoga instructor. Analyze this image of a practitioner. Identify the pose and provide ONE concise, actionable correction (max 2 sentences). Start with 'Analysis: '"
                
                response = self.model.generate_content([prompt, pil_img])
                return response.text.strip()
            except Exception as e:
                return f"Sage Error: {str(e)}"
        
        # Fallback
        time.sleep(1.5)
        return random.choice(self.mock_responses) + " (Simulated)"
