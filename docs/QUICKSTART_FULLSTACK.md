# ZENith Fullstack Quickstart Guide

## Prerequisites
- Python 3.10+
- Node.js 18+
- A webcam
- (Optional) Google Gemini API key for AI coaching

## 1. Backend Setup

```bash
cd zenith-mvp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment (optional)
cp .env.example .env
# Edit .env to set GEMINI_API_KEY, CORS origins, log level, etc.

# Start server
python server.py
```

You should see structured log output:
```
HH:MM:SS | zenith.server      | INFO  | ZENith API v2.5 starting on 0.0.0.0:8000
HH:MM:SS | zenith.server      | INFO  | Brain models: clf=OK, vae=OK
```

## 2. Frontend Setup

```bash
cd zenith-web
npm install
npm run dev
```

Open `http://localhost:5173` and allow camera access.

## 3. Usage

- **Onboarding:** A welcome guide appears on first visit. Click "?" in the header to re-show it
- **Connection:** Status badge turns GREEN ("LIVE") when connected
- **HUD (right):** Real-time Pose, Flow (with bar), Quality, and Stability metrics
- **Bio Panel (left):** Biomechanical quality score, deviations, joint angles
- **Coaching:** Heuristic corrections for all 10 poses appear as cyan badges with spoken feedback and correction arrows on the skeleton overlay
- **AI Coach (bottom-right):** Click to request Gemini Vision analysis (or mock if no API key). Shows "Gemini" or "Demo" source badge
- **Guided Flows:** Choose from 3 sequences (Warrior, Balance, Strength). Hold timer shows live countdown. Coach auto-analyzes during holds
- **Session:** Click "Record Session" to track metrics, "End Session" for a summary report with pose timeline
- **History:** Click "History" in the header to view past sessions

## 4. Configuration

All settings are in `.env` (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ZENITH_HOST` | `0.0.0.0` | Server bind address |
| `ZENITH_PORT` | `8000` | Server port |
| `ZENITH_CORS_ORIGINS` | `http://localhost:5173,...` | Allowed CORS origins |
| `ZENITH_MODEL_DIR` | `.` | Directory containing model weights |
| `GEMINI_API_KEY` | (empty) | Gemini API key (falls back to mock without it) |
| `ZENITH_LOG_LEVEL` | `INFO` | Log verbosity (DEBUG, INFO, WARNING, ERROR) |

## Troubleshooting

- **Server error:** Ensure ports 8000 and 5173 are free
- **Camera denied:** Check browser permissions, reload page
- **No camera found:** VideoStage will display an error message
- **"OFFLINE" badge:** Backend not running or CORS origin mismatch
- **Gemini errors:** Check API key in `.env`; mock coaching works without it
