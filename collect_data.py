import csv
import cv2
from handtracker import HandTracker


class collect_data:

  def __init__(self, csv_filename="gesture_data.csv"):
    self.tracker = HandTracker(num_hands=2)
    self.csv_file = open(csv_filename, mode="a", newline="")
    self.csv_writer = csv.writer(self.csv_file)

  def run(self):
    cap = cv2.VideoCapture(0)
    current_label = None

    print("Press 'q' to quit.")

    try:
      while cap.isOpened():
        success, frame = cap.read()
        if not success:
          continue

        result = self.tracker.process_frame(frame)
        display_text = "Ready..."

        if result.hand_landmarks:
          for hand_landmarks in result.hand_landmarks:
            features = self.tracker.extract_features(hand_landmarks)
            self.tracker.draw_landmarks(frame, hand_landmarks)

            if current_label is not None:
              row = [current_label] + features
              self.csv_writer.writerow(row)
              display_text = f"SAVED: {current_label}"

          current_label = None

        cv2.putText(
            frame,
            display_text,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.imshow("OOP Collector", frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord("q"):
          break
        elif key == ord("1"):
          current_label = "Thumbs_Up"
        elif key == ord("2"):
          current_label = "Thumbs_Down"
        elif key == ord("3"):
          current_label = "Peace"
    finally:
      cap.release()
      self.csv_file.close()
      cv2.destroyAllWindows()