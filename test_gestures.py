import cv2
import joblib
from handtracker import HandTracker


class GestureTester:

  def __init__(self, model_path="gesture_model.pkl", threshold=0.24):
    self.tracker = HandTracker(num_hands=1)
    self.model = joblib.load(model_path)
    self.threshold = threshold

  def run(self):
    cap = cv2.VideoCapture(0)
    print("testing ground for gesture accuracy...")
    print("press q to quit")

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
            pixel_landmarks = self.tracker.draw_landmarks(
                frame, hand_landmarks, (0, 255, 0)
            )
            features = self.tracker.extract_features(hand_landmarks)

            distances, _ = self.model.kneighbors([features])
            avg_distance = sum(distances[0]) / len(distances[0])

            x_coords = [pt[0] for pt in pixel_landmarks]
            y_coords = [pt[1] for pt in pixel_landmarks]
            x_min, x_max = max(0, min(x_coords) - 25), min(w, max(x_coords) + 25)
            y_min, y_max = max(0, min(y_coords) - 25), min(h, max(y_coords) + 25)

            if avg_distance > self.threshold:
              box_color = (0, 0, 255)
              display_text = "Unknown"
            else:
              prediction = self.model.predict([features])[0]
              box_color = (0, 255, 0)
              display_text = str(prediction)

            cv2.putText(
                frame,
                display_text,
                (x_min, max(35, y_min - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                box_color,
                2,
            )
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)

        cv2.imshow("Gesture Accuracy Tester", frame)

        if cv2.waitKey(5) & 0xFF == ord("q"):
          break
    finally:
      cap.release()
      cv2.destroyAllWindows()


if __name__ == "__main__":
  tester = GestureTester()
  tester.run()