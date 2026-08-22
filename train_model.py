import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 1. Load dataset
df = pd.read_csv("gesture_data.csv", header=None)
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train model (weights probabilities enabled by default)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

print(f"Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

# 4. Save model
joblib.dump(model, "gesture_model.pkl")
print("Model saved successfully!")