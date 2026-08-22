import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,  # Stricter initial detection
    min_hand_presence_confidence=0.6,  # Stricter presence check
    min_tracking_confidence=0.6,  # Stricter tracking stability
)

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

try:
  with HandLandmarker.create_from_options(options) as landmarker:
    frame_timestamp_ms = 0

    while cap.isOpened():
      success, frame = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        continue

      mp_image = mp.Image(
          image_format=mp.ImageFormat.SRGB,
          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
      )
      frame_timestamp_ms += int(1000 / 30)
      detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

      if detection_result.hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in detection_result.hand_landmarks:
          pixel_landmarks = [
              (int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks
          ]

          for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            cv2.line(
                frame,
                pixel_landmarks[start_idx],
                pixel_landmarks[end_idx],
                (255, 255, 255),
                2,
            )

          for pt in pixel_landmarks:
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)

      cv2.imshow("Hand Tracking - Press 'q' to Quit", frame)

      if cv2.waitKey(5) & 0xFF == ord("q"):
        break

except KeyboardInterrupt:
  pass

finally:
  cap.release()
  cv2.destroyAllWindows()