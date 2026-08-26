import time
import cv2
import joblib
import pyautogui
from handtracker import HandTracker

pyautogui.FAILSAFE = False


class GestureRecognizer:

  def __init__(self, model_path="gesture_model.pkl", threshold=0.24):
    self.tracker = HandTracker(num_hands=1)
    self.model = joblib.load(model_path)
    self.threshold = threshold

    self.screen_w, self.screen_h = pyautogui.size()
    self.prev_x, self.prev_y = self.screen_w / 2, self.screen_h / 2

    self.frame_margin = 0.2
    self.deadzone = 2.5

    self.last_click_time = 0
    self.click_cooldown = 1.0

    self.confidence_threshold = 0.75

  def center_mouse(self):
    center_x = self.screen_w / 2
    center_y = self.screen_h / 2
    pyautogui.moveTo(center_x, center_y)
    self.prev_x, self.prev_y = center_x, center_y
    print("Cursor centered")

  def run(self):
    cap = cv2.VideoCapture(0)
    print("use index finger to move cursor")
    print("press c to center, q to quit")

    self.center_mouse()

    try:
      while cap.isOpened():
        success, frame = cap.read()
        if not success:
          continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        result = self.tracker.process_frame(frame)

        if result.hand_landmarks:
          for hand_landmarks in result.hand_landmarks:
            index_tip = hand_landmarks[8]

            x_normalized = max(
                0.0,
                min(
                    1.0,
                    (index_tip.x - self.frame_margin)
                    / (1.0 - 2 * self.frame_margin),
                ),
            )
            y_normalized = max(
                0.0,
                min(
                    1.0,
                    (index_tip.y - self.frame_margin)
                    / (1.0 - 2 * self.frame_margin),
                ),
            )

            target_x = x_normalized * self.screen_w
            target_y = y_normalized * self.screen_h

            distance_moved = (
                (target_x - self.prev_x) ** 2 + (target_y - self.prev_y) ** 2
            ) ** 0.5

            if distance_moved > self.deadzone:
              alpha = 0.4 if distance_moved > 40 else 0.25

              current_x = (alpha * target_x) + ((1 - alpha) * self.prev_x)
              current_y = (alpha * target_y) + ((1 - alpha) * self.prev_y)

              current_x = max(0, min(self.screen_w - 1, current_x))
              current_y = max(0, min(self.screen_h - 1, current_y))

              pyautogui.moveTo(current_x, current_y)
              self.prev_x, self.prev_y = current_x, current_y

            cv2.circle(
                frame,
                (int(index_tip.x * w), int(index_tip.y * h)),
                8,
                (255, 0, 0),
                -1,
            )

            features = self.tracker.extract_features(hand_landmarks)
            pixel_landmarks = self.tracker.draw_landmarks(
                frame, hand_landmarks, (0, 255, 0)
            )

            distances, _ = self.model.kneighbors([features])
            avg_distance = sum(distances[0]) / len(distances[0])

            probabilities = self.model.predict_proba([features])[0]
            max_confidence = max(probabilities)
            prediction = self.model.classes_[probabilities.argmax()]

            x_coords = [pt[0] for pt in pixel_landmarks]
            y_coords = [pt[1] for pt in pixel_landmarks]
            x_min, x_max = max(0, min(x_coords) - 25), min(w, max(x_coords) + 25)
            y_min, y_max = max(0, min(y_coords) - 25), min(h, max(y_coords) + 25)

            if (
                avg_distance > self.threshold
                or max_confidence < self.confidence_threshold
            ):
              box_color = (0, 0, 255)
              display_text = "Unknown"
            else:
              box_color = (0, 255, 0)
              display_text = f"{prediction} ({int(max_confidence * 100)}%)"

              current_time = time.time()
              if current_time - self.last_click_time > self.click_cooldown:
                if prediction == "Left_Click":
                  pyautogui.click()
                  print("left click")
                  self.last_click_time = current_time
                elif prediction == "Right_Click":
                  pyautogui.rightClick()
                  print("right click")
                  self.last_click_time = current_time

            cv2.putText(
                frame,
                display_text,
                (x_min, max(35, y_min - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                box_color,
                2,
            )
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)

        cv2.imshow("AirMouse", frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord("q"):
          break
        elif key == ord("c"):
          self.center_mouse()

    finally:
      cap.release()
      cv2.destroyAllWindows()


if __name__ == "__main__":
  recognizer = GestureRecognizer()
  recognizer.run()