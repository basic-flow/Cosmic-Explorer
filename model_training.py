# model_training.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib


def train_exoplanet_model():
    # Load KOI data (you can download from NASA Exoplanet Archive)
    # For demonstration, I'll create synthetic data similar to KOI features
    print("Training XGBoost model for exoplanet detection...")

    # Example features from Kepler Object of Interest data
    features = [
        'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration',
        'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol',
        'koi_model_snr', 'koi_steff', 'koi_slogg', 'koi_srad',
        'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'
    ]

    # Create synthetic training data (replace with real KOI data)
    np.random.seed(42)
    n_samples = 5000

    X = pd.DataFrame({
        'koi_period': np.random.uniform(0.5, 500, n_samples),  # orbital period in days
        'koi_time0bk': np.random.uniform(100, 2000, n_samples),  # transit epoch
        'koi_impact': np.random.uniform(0, 1, n_samples),  # impact parameter
        'koi_duration': np.random.uniform(1, 20, n_samples),  # transit duration in hours
        'koi_depth': np.random.uniform(50, 5000, n_samples),  # transit depth in ppm
        'koi_prad': np.random.uniform(0.5, 20, n_samples),  # planetary radius (Earth radii)
        'koi_teq': np.random.uniform(300, 3000, n_samples),  # equilibrium temperature (K)
        'koi_insol': np.random.uniform(0.1, 100, n_samples),  # insolation flux (Earth flux)
        'koi_model_snr': np.random.uniform(5, 100, n_samples),  # signal-to-noise ratio
        'koi_steff': np.random.uniform(3000, 7000, n_samples),  # stellar temperature (K)
        'koi_slogg': np.random.uniform(3.5, 5.0, n_samples),  # stellar surface gravity
        'koi_srad': np.random.uniform(0.5, 2.0, n_samples),  # stellar radius (Solar radii)
        'koi_fpflag_nt': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),  # not transit-like flag
        'koi_fpflag_ss': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),  # stellar eclipse flag
        'koi_fpflag_co': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),  # centroid offset flag
        'koi_fpflag_ec': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),  # ephemeris match flag
    })

    # Create target variable (exoplanet = 1, non-exoplanet = 0)
    # Real logic would be more complex - this is simplified
    y = (
            (X['koi_model_snr'] > 15) &
            (X['koi_fpflag_nt'] == 0) &
            (X['koi_depth'] > 100) &
            (np.random.random(n_samples) > 0.3)
    ).astype(int)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model trained with accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    # Save model and scaler
    joblib.dump(model, 'exoplanet_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(features, 'feature_names.pkl')

    print("Model saved successfully!")
    return model, scaler, features


if __name__ == "__main__":
    train_exoplanet_model()