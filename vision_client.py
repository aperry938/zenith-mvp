import os
import time
import random
import cv2
from PIL import Image
from config import setup_logging, GEMINI_API_KEY

logger = setup_logging("zenith.vision")

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

class VisionClient:
    """
    Multimodal LLM analysis client for biomechanical coaching.
    Connects to Gemini API if key is present, else falls back to mock.
    """
    def __init__(self):
        self.api_key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        self.model = None
        self.is_mock = True

        self.mock_responses = [
            "Your spinal alignment improves when you engage your core. Try to lengthen your neck.",
            "Excellent stability in the lower body. Watch out for hyperextension in the elbows.",
            "Good flow. Your transition was smooth, but check your front knee tracking over the ankle.",
            "You are holding tension in your shoulders. Exhale and let them drop away from your ears.",
            "Form looks solid. Try holding for 5 more seconds to build endurance."
        ]

        if HAS_GENAI and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.is_mock = False
                logger.info("Connected to Gemini API")
            except Exception as e:
                logger.error(f"Gemini init error: {e}")

    def analyze_frame(self, img_array, pose_label=None, bio_quality=None, deviations=None):
        """
        Sends frame to Gemini (or Mock) with pose context.
        Returns (coaching_text, source) tuple.
        """
        if not self.is_mock and self.model:
            return self._analyze_real(img_array, pose_label, bio_quality, deviations)

        # Mock fallback
        time.sleep(1.0)
        return random.choice(self.mock_responses), "mock"

    def _analyze_real(self, img_array, pose_label, bio_quality, deviations):
        """Real Gemini API call with retry."""
        rgb_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        # Build context-rich prompt
        context = "You are an expert yoga instructor analyzing a live practitioner."
        if pose_label:
            context += f" They are performing {pose_label}."
        if bio_quality is not None:
            context += f" Biomechanical quality score: {int(bio_quality)}/100."
        if deviations:
            dev_strs = [f"{d['feature']} ({d['direction']} ideal)" for d in deviations[:3]]
            context += f" Top deviations: {', '.join(dev_strs)}."
        prompt = f"{context} Provide ONE concise, actionable correction (max 2 sentences)."

        for attempt in range(2):
            try:
                response = self.model.generate_content([prompt, pil_img])
                return response.text.strip(), "gemini"
            except Exception as e:
                logger.error(f"Gemini API error (attempt {attempt + 1}): {e}")
                if attempt == 0:
                    time.sleep(2)

        return "Unable to analyze at this time.", "error"
