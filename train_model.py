# src/train_model.py
# Trains a Random Forest classifier and saves it

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
import joblib
import os

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Trains Random Forest, prints evaluation metrics, saves model.
    """

    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,   # 100 decision trees
        max_depth=20,       # prevent overfitting
        random_state=42,
        n_jobs=-1           # use all CPU cores
    )
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/threat_model.pkl")
    print("\nModel saved to models/threat_model.pkl")

    return model, y_pred, cm