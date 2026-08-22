import csv
import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Configure MediaPipe to track up to 2 hands (left and right)
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)

# Standard MediaPipe hand bone connection index pairs for drawing lines
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # Thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # Index
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # Middle
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # Ring
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # Pinky
    (0, 17),  # Palm
]

cap = cv2.VideoCapture(0)

# Open or create a CSV file to save your training data
csv_file = open("gesture_data.csv", mode="a", newline="")
csv_writer = csv.writer(csv_file)

print("--- GESTURE DATA COLLECTOR ---")
print("Press '1' on your keyboard for: Thumbs_Up")
print("Press '2' on your keyboard for: Thumbs_Down")
print("Press 'q' to Quit")

current_label = None

try:
  with HandLandmarker.create_from_options(options) as landmarker:
    frame_timestamp_ms = 0

    while cap.isOpened():
      success, frame = cap.read()
      if not success:
        continue

      mp_image = mp.Image(
          image_format=mp.ImageFormat.SRGB,
          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
      )
      frame_timestamp_ms += int(1000 / 30)
      detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

      display_text = "Ready: Press '1' (Thumbs Up) or '2' (Thumbs Down)"

      if detection_result.hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in detection_result.hand_landmarks:
          # Normalize coordinates relative to the wrist (landmark 0)
          wrist_x = hand_landmarks[0].x
          wrist_y = hand_landmarks[0].y

          row_data = []
          pixel_landmarks = []
          for lm in hand_landmarks:
            row_data.append(lm.x - wrist_x)
            row_data.append(lm.y - wrist_y)
            pixel_landmarks.append((int(lm.x * w), int(lm.y * h)))

          # If you pressed 1 or 2, record this frame's data into the CSV
          if current_label is not None:
            row_data.insert(
                0, current_label
            )  # First column stores the label name
            csv_writer.writerow(row_data)
            display_text = f"SAVED: {current_label}"

          # Draw the white skeleton lines connecting the joints
          for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            cv2.line(
                frame,
                pixel_landmarks[start_idx],
                pixel_landmarks[end_idx],
                (255, 255, 255),
                2,
            )

          # Draw green joint dots over the finger joints
          for pt in pixel_landmarks:
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)

        # Reset the label trigger after capturing the frame
        current_label = None

      # Show instructions on the webcam feed
      cv2.putText(
          frame,
          display_text,
          (10, 40),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.7,
          (0, 255, 255),
          2,
      )
      cv2.imshow("Data Collector - Press 'q' to Quit", frame)

      # Key listening logic mapped to Thumbs Up and Thumbs Down
      key = cv2.waitKey(5) & 0xFF
      if key == ord("q"):
        break
      elif key == ord("1"):
        current_label = "Thumbs_Up"
      elif key == ord("2"):
        current_label = "Thumbs_Down"

except KeyboardInterrupt:
  pass

finally:
  csv_file.close()
  cap.release()
  cv2.destroyAllWindows()