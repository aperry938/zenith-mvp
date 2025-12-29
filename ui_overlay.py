import cv2
import numpy as np

class ZenithUI:
    """
    Centralized Design System for ZENith.
    Handles all OpenCV drawing operations with a unified aesthetic (Neon Sci-Fi).
    """
    def __init__(self):
        # COLORS (BGR)
        self.C_BG_DARK = (20, 20, 20)
        self.C_TEXT = (255, 255, 255)
        self.C_TEXT_DIM = (200, 200, 200)
        
        self.C_NEON_BLUE = (255, 215, 0) # Cyan-ish
        self.C_NEON_PURPLE = (255, 66, 230)
        self.C_NEON_GREEN = (0, 255, 0)
        self.C_NEON_RED = (0, 0, 255)
        self.C_NEON_ORANGE = (0, 165, 255)
        
        # FONTS
        self.F_MAIN = cv2.FONT_HERSHEY_SIMPLEX

    def draw_hud(self, img, label, q, fps):
        """Draws the Main Status Box (Top Left)."""
        cv2.rectangle(img, (20, 20), (520, 120), (65, 109, 255), -1)
        
        # Label
        label_text = label if label else "—"
        cv2.putText(img, f"POSE: {label_text}", (35, 60), self.F_MAIN, 0.95, self.C_TEXT, 2, cv2.LINE_AA)
        
        # Quality
        cv2.putText(img, "QUALITY:", (35, 97), self.F_MAIN, 0.85, self.C_TEXT, 2, cv2.LINE_AA)
        
        qv = 0 if q is None else int(round(q))
        bar_x, bar_y, bar_w, bar_h = 160, 78, 280, 20
        
        # Bar BG
        cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), self.C_BG_DARK, -1)
        
        # Bar Fill
        color = (0, 170, 0) if qv > 85 else ((0, 200, 255) if qv > 60 else (0, 0, 255))
        fill_w = int(bar_w * qv / 100)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x+fill_w, bar_y+bar_h), color, -1)
        
        # Quality Text
        cv2.putText(img, f"{qv:3d}", (bar_x+bar_w+12, bar_y+16), self.F_MAIN, 0.7, self.C_TEXT, 2, cv2.LINE_AA)
        
        # FPS
        h, w, _ = img.shape
        cv2.putText(img, f"{fps:.1f} FPS", (w - 140, 30), self.F_MAIN, 0.7, self.C_TEXT_DIM, 2, cv2.LINE_AA)

    def draw_flow_bar(self, img, score):
        """Draws the Flow Meter (Bottom)."""
        h, w, _ = img.shape
        bar_h = 20
        bar_w = int(w * 0.8)
        x = int(w * 0.1)
        y = h - 60
        
        # BG
        cv2.rectangle(img, (x, y), (x+bar_w, y+bar_h), (30, 30, 30), -1)
        
        # Fill
        fill_w = int(bar_w * (score / 100.0))
        
        # Color Logic: Red < 50 < Purple < 80 < Blue
        color = (255, 50, 50) # Blue
        if score < 50: color = (0, 0, 255) # Red
        elif score < 80: color = (255, 0, 255) # Purple
        
        cv2.rectangle(img, (x, y), (x+fill_w, y+bar_h), color, -1)
        
        # Text
        cv2.putText(img, f"FLOW: {int(score)}", (x, y-10), self.F_MAIN, 0.7, color, 2, cv2.LINE_AA)

    def draw_sequencer(self, img, current, next_pose, progress):
        """Draws the Sequence HUD (Top Right)."""
        h, w, _ = img.shape
        w_box = 200
        h_box = 80
        x = w - w_box - 20
        y = 50 # Moved down slightly to avoid FPS overlap
        
        # BG
        cv2.rectangle(img, (x, y), (x+w_box, y+h_box), (50, 50, 50), -1)
        
        # Progress Bar
        bar_h = 6
        fill_w = int(w_box * progress)
        cv2.rectangle(img, (x, y), (x+fill_w, y+bar_h), self.C_NEON_GREEN, -1)
        
        # Text
        cv2.putText(img, f"NOW: {current}", (x+10, y+35), self.F_MAIN, 0.55, self.C_TEXT, 1, cv2.LINE_AA)
        cv2.putText(img, f"NEXT: {next_pose}", (x+10, y+65), self.F_MAIN, 0.45, self.C_TEXT_DIM, 1, cv2.LINE_AA)

    def draw_arrow(self, img, start_norm, end_norm, text=None):
        """Draws AR Arrow (Visual Whispers)."""
        h, w, _ = img.shape
        start_px = (int(start_norm[0] * w), int(start_norm[1] * h))
        end_px   = (int(end_norm[0] * w), int(end_norm[1] * h))
        
        color = (0, 255, 255) # Yellow
        thickness = 4
        
        cv2.arrowedLine(img, start_px, end_px, color, thickness, tipLength=0.3)
        cv2.circle(img, end_px, 6, self.C_TEXT, -1)
        
        if text:
            # Add a small backdrop for text readability
            (tw, th), _ = cv2.getTextSize(text, self.F_MAIN, 0.6, 2)
            bg_pt1 = (end_px[0] + 8, end_px[1] - th - 4)
            bg_pt2 = (end_px[0] + 12 + tw, end_px[1] + 4)
            cv2.rectangle(img, bg_pt1, bg_pt2, (0,0,0), -1)
            cv2.putText(img, text, (end_px[0]+10, end_px[1]), self.F_MAIN, 0.6, self.C_TEXT, 2, cv2.LINE_AA)

    def draw_halo(self, img, landmarks, progress, is_locked):
        """Draws the Stability Halo."""
        h, w, _ = img.shape
        xs = [lm.x for lm in landmarks.landmark]
        ys = [lm.y for lm in landmarks.landmark]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        center_x = int(((min_x + max_x) / 2) * w)
        center_y = int(((min_y + max_y) / 2) * h)
        
        axis_x = int(((max_x - min_x) / 1.5) * w)
        axis_y = int(((max_y - min_y) / 1.5) * h)
        
        if is_locked:
            color = (0, 215, 255) # Gold/Orange
            thickness = 5
            # Double ring for Lock
            cv2.ellipse(img, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, color, thickness)
            cv2.ellipse(img, (center_x, center_y), (axis_x+12, axis_y+12), 0, 0, 360, (0, 165, 255), 2)
        else:
            color = (255, 120, 0) # Blue
            thickness = 2
            end_angle = int(360 * progress)
            cv2.ellipse(img, (center_x, center_y), (axis_x, axis_y), 0, 0, end_angle, color, thickness)
