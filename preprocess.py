# src/preprocess.py
# Cleans data, encodes categorical features, and scales numerics

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def preprocess(train_df, test_df):
    """
    - Encode categorical columns (protocol_type, service, flag)
    - Drop the raw attack_type string column
    - Scale numeric features
    """

    # Columns with text values that need encoding
    categorical_cols = ["protocol_type", "service", "flag"]

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on combined data so test set labels don't cause errors
        combined = pd.concat([train_df[col], test_df[col]])
        le.fit(combined)
        train_df[col] = le.transform(train_df[col])
        test_df[col]  = le.transform(test_df[col])
        encoders[col] = le

    # Drop the attack_type string column (we already have binary 'label')
    train_df.drop("attack_type", axis=1, inplace=True)
    test_df.drop("attack_type",  axis=1, inplace=True)

    # Separate features and labels
    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"]
    X_test  = test_df.drop("label", axis=1)
    y_test  = test_df["label"]

    # Scale features (zero mean, unit variance)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Save scaler and encoders for use in Flask API later
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler,   "models/scaler.pkl")
    joblib.dump(encoders, "models/encoders.pkl")

    print("Preprocessing complete. Scaler and encoders saved.")
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns.tolist()
