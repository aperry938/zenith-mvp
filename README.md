# ZENith: AI-Assisted Movement Coaching

**By: Anthony Perry** | **[linkedin.com/in/aperry9-38](https://www.linkedin.com/in/aperry9-38/)** | **anthonycperry21@gmail.com**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)

---

### Demo Video

*A short video demonstrating the live application can be found here: [![Watch the ZENith Demo Video](https://img.youtube.com/vi/lPnmOBwJfkE/0.jpg)](https://youtu.be/lPnmOBwJfkE)*

---

### Abstract

This project presents a proof-of-concept for an AI-assisted yoga coach that provides **real-time pose classification**. The system leverages a computer vision pipeline to extract 3D skeletal landmarks from a live webcam feed. This data is processed by a supervised classifier to accurately identify 10 core yoga poses. The primary goal of this work is to demonstrate a feasible, low-cost architecture for creating data-driven feedback in at-home wellness settings. This MVP serves as the foundation for future PhD research into more advanced topics like quality assessment and mitigating algorithmic bias in AI coaching systems.

### Key Features
* **Real-Time Pose Classification:** Successfully identifies 10 core yoga poses from a live video stream.
* **Computer Vision Pipeline:** Utilizes MediaPipe to extract 33 body landmarks for analysis.
* **Interactive Web Demo:** A fully functional application built with Streamlit and WebRTC.

### Results & Evaluation

The pose classifier achieved **99% accuracy** on the test set. The confusion matrix below shows near-perfect classification with no confusion between the 10 distinct poses.

![Pose Classifier Confusion Matrix](Pose_Classifier_Confusion_Matrix_10_Poses.png)

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

[Model training was conducted in this Colab Notebook: https://colab.research.google.com/drive/1DSYxlitGTFTivI2nsCgHxrPoP5nJK-5Y](https://colab.research.google.com/drive/1DSYxlitGTFTivI2nsCgHxrPoP5nJK-5Y)