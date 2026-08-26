from collect_data import GestureCollector
from recognize import GestureRecognizer
from train_model import train_gesture_model
from test_gestures import GestureTester


def main():
  print("1. Collect gesture data")
  print("2. Train or retrain model")
  print("3. Run gesture recognition")
  print("4. Test gesture recognition")
  choice = input("Enter 1, 2, 3, or 4: ").strip()

  if choice == "1":
    collector = GestureCollector()
    collector.run()
  elif choice == "2":
    print("Starting model training...")
    train_gesture_model()
  elif choice == "3":
    try:
      recognizer = GestureRecognizer()
      recognizer.run()
    except FileNotFoundError:
      print(
          "Error: collect data and train your model fist."
      )
  elif choice == "4":
    try:
      tester = GestureTester()
      tester.run()
    except FileNotFoundError:
      print("Error.")
  else:
    print("Invalid choice.")


if __name__ == "__main__":
  main()