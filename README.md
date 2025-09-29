# ZENith: AI-Assisted Movement Coaching

**By: Anthony Perry** | **[linkedin.com/in/aperry938](https://www.linkedin.com/in/aperry938/)** | **anthonycperry21@gmail.com**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)

---

### Demo Video

*A short video demonstrating the live application can be found here: [![Watch the ZENith Demo Video](https://img.youtube.com/vi/lPnmOBwJfkE/0.jpg)](https://youtu.be/lPnmOBwJfkE)

---

### Abstract

This project presents a proof-of-concept for an AI-assisted yoga coach that provides real-time pose classification and kinesiology-based quality assessment. The system leverages a computer vision pipeline to extract 3D skeletal landmarks from a live webcam feed. This data is processed by two core models: a supervised classifier for pose identification and an unsupervised deep learning model (a Variational Autoencoder) for form quality analysis. The primary goal of this work is to demonstrate a feasible, low-cost architecture for creating personalized and data-driven feedback in at-home wellness and physical rehabilitation settings. This MVP serves as the foundation for future PhD research into mitigating algorithmic bias in AI coaching systems.

### Key Features
* **Real-Time Pose Classification:** Identifies **10 core yoga poses** from a live video stream.
* **AI Quality Assessment:** An unsupervised VAE scores form correctness based on a learned distribution of "correct" poses.
* **Dynamic Biomechanical Analysis:** Calculates and displays 8 different real-time joint angles (shoulders, elbows, hips, knees).
* **Interactive Web Demo:** A fully functional application built with Streamlit and WebRTC.

### How to Reproduce

This application can be run locally on macOS with Apple Silicon.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/aperry938/zenith-mvp](https://github.com/aperry938/zenith-mvp)
    ```
2.  **Navigate into the project directory:**
    ```bash
    cd zenith-mvp
    ```
3.  **Create and activate the Conda environment:**
    ```bash
    # Requires Miniconda to be installed
    conda create --name zenith python=3.11 -y
    conda activate zenith
    ```
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

https://colab.research.google.com/drive/1DSYxlitGTFTivI2nsCgHxrPoP5nJK-5Y
