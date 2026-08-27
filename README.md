# VisionMouse: Virtual Gesture-Controlled Mouse

A real-time computer vision application that tracks your index finger and custom gestures for mouse control. Built using **Python**, **OpenCV**, **MediaPipe**, and **Scikit-Learn**.

---

## Key Features

* **Custom Machine Learning Clicks:** Uses a **K-Nearest Neighbors (KNN)** classifier trained on custom hand landmark coordinates to recognize left and right click gestures with confidence score filtering (75%+ thresholding) to eliminate false positives.
* **Ergonomic Active Zone:** Features a customized camera margin and boundary clamping to prevent your cursor getting stuck in the corner, and ensures comfortable range control across your display.

---

## Tech Stack & Libraries

* **OpenCV (`cv2`):** Webcam frame capture, image transformations, and real-time bounding box rendering.
* **MediaPipe:** High-performance 3D hand landmark detection and skeletal feature extraction.
* **Scikit-Learn:** K-Nearest Neighbors (KNN) model training and probability estimation
* **PyAutoGUI:** System-level hardware automation for cursor movement and click handling.

---

## Setup

**1. Install Dependencies:**
Ensure you have Python installed, then run: pip install opencv-python mediapipe scikit-learn pyautogui joblib pandas

**2. Run the Control Center:**
Run main.py

**3. Options**

- Collect Data (Option 1): Record custom poses (for left and right clicks)
- Train Model (Option 2): Train the KNN classifier on your collected dataset.
- Run VisionMouse (Option 3):
- Test Accuracy (Option 4): Test the accuracy of your model predictions in real time before enabling mouse control.

- c -- centers your mouse
- q -- quits the application
