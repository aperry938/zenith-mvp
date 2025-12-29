# Zenith Fullstack Quickstart Guide

This guide will help you set up and run the Zenith AI Yoga Coach (Fullstack Version).

## Prerequisites
*   Python 3.10+
*   Node.js 18+
*   A Google Gemini API Key

## 1. Backend Setup (Server)

1.  **Environment:**
    Ensure you are in the root directory (`zenith-mvp-main`).
    ```bash
    # (Optional) Create venv
    python -m venv venv
    source venv/bin/activate
    ```

2.  **Dependencies:**
    Install the Python requirements.
    ```bash
    pip install -r requirements.txt
    pip install fastapi uvicorn websockets python-multipart
    ```

3.  **API Key:**
    Set your Gemini API Key.
    ```bash
    export GEMINI_API_KEY="your_api_key_here"
    ```

4.  **Run Server:**
    ```bash
    python server.py
    ```
    You should see `Uvicorn running on http://0.0.0.0:8000`.

## 2. Frontend Setup (Client)

1.  **Navigate:**
    ```bash
    cd zenith-web
    ```

2.  **Install:**
    ```bash
    npm install
    ```

3.  **Run:**
    ```bash
    npm run dev
    ```

4.  **Open:**
    Visit the URL shown (usually `http://localhost:5173`). Allow camera access.

## 3. Usage

*   **Status Indicator:** Should turn GREEN ("LIVE") when connected to the server.
*   **HUD:** You will see real-time metrics for Pose, Flow, and Quality.
*   **Ask Sage:** Click the button to have the AI analyze your current frame and give audible advice.

## Troubleshooting

*   **Server Error:** Ensure ports 8000 and 5173 are free.
*   **Camera:** Ensure your browser has permission to access the webcam.
*   **Gemini Error:** Check your API key.
