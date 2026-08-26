import cv2
import mediapipe as mp


class HandTracker:

  def __init__(self, num_hands=1, model_path="hand_landmarker.task"):
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    self.options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    self.landmarker = HandLandmarker.create_from_options(self.options)
    self.timestamp_ms = 0

    self.connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (0, 17),
    ]

  def process_frame(self, frame):
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )
    self.timestamp_ms += int(1000 / 30)
    return self.landmarker.detect_for_video(mp_image, self.timestamp_ms)

  def extract_features(self, hand_landmarks):
    wrist_x = hand_landmarks[0].x
    wrist_y = hand_landmarks[0].y
    row_data = []
    for lm in hand_landmarks:
      row_data.append(lm.x - wrist_x)
      row_data.append(lm.y - wrist_y)
    return row_data

  def draw_landmarks(self, frame, hand_landmarks, color=(0, 255, 0)):
    h, w, _ = frame.shape
    pixel_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for start_idx, end_idx in self.connections:
      cv2.line(
          frame,
          pixel_landmarks[start_idx],
          pixel_landmarks[end_idx],
          (255, 255, 255),
          2,
      )
    for pt in pixel_landmarks:
      cv2.circle(frame, pt, 5, color, -1)

    return pixel_landmarks