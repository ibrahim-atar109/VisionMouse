import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def train_gesture_model():
  try:
    df = pd.read_csv("gesture_data.csv", header=None)
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Increase neighbors to 5 and use distance weighting
    model = KNeighborsClassifier(n_neighbors=5, weights="distance")
    model.fit(X_train, y_train)

    print(f"Model Accuracy: {model.score(X_test, y_test) * 100:.2f}%")
    joblib.dump(model, "gesture_model.pkl")
    print("Model saved successfully!")

  except Exception as e:
    print("Error training model:", e)


if __name__ == "__main__":
  train_gesture_model()