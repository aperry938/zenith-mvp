import os
import cv2
import time
import numpy as np

DATASET_ROOT = "dataset"

class DataHarvester:
    """
    Automated Data Collection Engine.
    Saves high-quality frames to build a proprietary dataset.
    """
    def __init__(self):
        if not os.path.exists(DATASET_ROOT):
            os.makedirs(DATASET_ROOT)
        self.frame_count = 0
        self.last_save_time = 0
        self.min_interval = 2.0  # Seconds between saves per pose

    def save_frame(self, img, label, quality):
        """
        Saves the frame if quality is sufficient.
        Folder structure: dataset/{label}/{timestamp}.jpg
        """
        if not label or quality < 85: # Strict quality threshold
            return

        now = time.time()
        if (now - self.last_save_time) < self.min_interval:
            return

        # Sanitize label
        safe_label = label.replace(" ", "_").lower()
        label_dir = os.path.join(DATASET_ROOT, safe_label)
        
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        filename = f"{int(now)}_{int(quality)}.jpg"
        path = os.path.join(label_dir, filename)
        
        try:
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Ensure RGB save if needed, but imwrite expects BGR usually. Input img is BGR from main loop.
            # Wait, main loop frame is BGR?
            # app_async_vae.py: img = frame.to_ndarray(format="bgr24")
            # So cv2.imwrite wants BGR. Pass as is.
            cv2.imwrite(path, img) 
            
            self.frame_count += 1
            self.last_save_time = now
            # print(f"Harvested: {path}") # Debug
        except Exception as e:
            print(f"Harvest Error: {e}")

    def get_stats(self):
        return self.frame_count
