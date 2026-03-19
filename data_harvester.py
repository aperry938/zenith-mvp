import os
import cv2
import time
import threading
from config import setup_logging

logger = setup_logging("zenith.harvester")
DATASET_ROOT = "dataset"

class DataHarvester:
    """
    Automated Data Collection Engine.
    Saves high-quality frames to build a proprietary dataset. Thread-safe.
    """
    def __init__(self):
        if not os.path.exists(DATASET_ROOT):
            os.makedirs(DATASET_ROOT)
        self._lock = threading.Lock()
        self.harvesting = False
        self.frame_count = 0
        self.last_save_time = 0
        self.min_interval = 2.0  # Seconds between saves per pose

    def save_frame(self, img, label, quality):
        if not label or quality < 85:
            return

        with self._lock:
            now = time.time()
            if (now - self.last_save_time) < self.min_interval:
                return

            safe_label = label.replace(" ", "_").lower()
            label_dir = os.path.join(DATASET_ROOT, safe_label)

            if not os.path.exists(label_dir):
                os.makedirs(label_dir)

            filename = f"{int(now)}_{int(quality)}.jpg"
            path = os.path.join(label_dir, filename)

            try:
                cv2.imwrite(path, img)
                self.frame_count += 1
                self.last_save_time = now
                logger.debug(f"Harvested: {path}")
            except Exception as e:
                logger.error(f"Harvest error: {e}")

    def get_stats(self):
        return self.frame_count
