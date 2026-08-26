from collect_data import collect_data
from recognize import GestureRecognizer
from train_model import train_gesture_model


def main():
  print("1. Collect gesture data")
  print("2. Train or retrain model")
  print("3. Run gesture recognition")
  choice = input("Enter 1, 2, or 3: ").strip()

  if choice == "1":
    collector = collect_data()
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
  else:
    print("Invalid choice.")


if __name__ == "__main__":
  main()