import pickle
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the data
with open(os.path.join("data", "hand_landmarks_data.pkl"), "rb") as f:
    dataset = pickle.load(f)

X = dataset["data"]
y = dataset["labels"]

# Encode labels if they are not numeric
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Initialize and train the XGBoost Classifier
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='mlogloss',
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Optional: Save the model and label encoder
joblib.dump(model, "xgb_hand_gesture_model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("\nModel and label encoder saved.")
