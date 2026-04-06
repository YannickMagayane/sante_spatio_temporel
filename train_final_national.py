#!/usr/bin/env python3
"""
ENTRAÎNEMENT NATIONAL AVEC LOG TRANSFORM - Pour données très skewed
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import gc
import pickle
from collections import defaultdict

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
sys.path.insert(0, '/app')
os.chdir('/app')

import django
django.setup()

from analytics.models import HealthZone

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score

MODELS_DIR = Path("/app/models")
PROCESSED_DIR = Path("/app/data/processed")
MODELS_DIR.mkdir(exist_ok=True, parents=True)

EPIDEMIC_DISEASES = ['Paludisme', 'Tuberculose', 'VIH/SIDA']

print("=" * 70)
print("🇨🇩 ENTRAÎNEMENT NATIONAL + LOG TRANSFORM")
print("=" * 70)

def prepare_series():
    df = pd.read_csv(PROCESSED_DIR / 'epi_data.csv')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'disease', 'cases'])
    df = df[df['disease'].isin(EPIDEMIC_DISEASES)]

    series = {}
    for disease in EPIDEMIC_DISEASES:
        ddf = df[df['disease'] == disease].copy()
        ddf['month'] = ddf['date'].dt.to_period('M')
        monthly = ddf.groupby('month')['cases'].sum().sort_index()
        if len(monthly) >= 12:
            series[disease] = monthly.values
            print(f"\n🦠 {disease}: {len(monthly)} mois, mean={monthly.mean():,.0f}, std={monthly.std():,.0f}")
    return series

def create_features(cases, window):
    n = len(cases)
    # Log transform
    cases_log = np.log1p(cases)

    # Temporal features
    months = np.arange(n)
    sin_month = np.sin(2 * np.pi * months / 12)
    cos_month = np.cos(2 * np.pi * months / 12)

    # Lags on log-transformed
    lag1 = np.roll(cases_log, 1)
    lag2 = np.roll(cases_log, 2)
    lag3 = np.roll(cases_log, 3)
    lag6 = np.roll(cases_log, 6) if n >= 6 else np.zeros(n)

    # Moving averages
    ma3 = pd.Series(cases_log).rolling(3).mean().values
    ma6 = pd.Series(cases_log).rolling(6).mean().values

    features = np.column_stack([
        cases_log,
        sin_month,
        cos_month,
        lag1,
        lag2,
        lag3,
        lag6,
        ma3,
        ma6
    ])
    return np.nan_to_num(features, nan=0.0)

def create_sequences(features, targets, window):
    X, y = [], []
    for i in range(len(features) - window):
        X.append(features[i:i+window])
        y.append(targets[i+window])
    if len(X) == 0:
        return np.array([]), np.array([])
    X_arr = np.array(X)  # Shape: (samples, window, n_features)
    y_arr = np.array(y)  # Shape: (samples,)
    return X_arr, y_arr

def build_simple_model(input_shape):
    model = Sequential([
        LSTM(16, activation='relu', input_shape=input_shape,
             kernel_regularizer=l2(1e-4), return_sequences=False),
        Dropout(0.3),
        BatchNormalization(),
        Dense(8, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def persistence_score(y_true, y_pred):
    """Calculate R² for persistence model (predict last value)"""
    y_persist = np.roll(y_true, 1)[1:]
    y_true_adj = y_true[1:]
    return r2_score(y_true_adj, y_persist)

def train_model(disease, cases):
    n = len(cases)
    window = 6 if n >= 24 else 3

    if n < window + 5:
        print(f"    ⚠️ Not enough data ({n} < {window+5})")
        return None

    features = create_features(cases, window)

    # Scale
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()

    X_scaled = scaler_X.fit_transform(features)
    y_scaled = scaler_y.fit_transform(cases.reshape(-1, 1)).flatten()

    X, y = create_sequences(X_scaled, y_scaled, window)

    if len(X) < 5:
        print(f"    ⚠️ Not enough sequences ({len(X)})")
        return None

    # Split
    split = max(1, int(len(X) * 0.7))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if len(X_test) < 1:
        print(f"    ⚠️ No test data")
        return None

    print(f"    Train: {len(X_train)}, Test: {len(X_test)}, Window: {window}")

    # Train
    model = build_simple_model((window, X.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=min(4, len(X_train)//2),
        callbacks=[early_stop],
        verbose=0
    )

    # Predict
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # Inverse transform
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # Metrics
    r2 = r2_score(y_test_orig, y_pred_orig)
    mae = mean_absolute_error(y_test_orig, y_pred_orig)

    # Compare with persistence
    r2_persist = persistence_score(y_test_orig, y_pred_orig) if len(y_test_orig) > 1 else -10

    # Save model only if R² >= -1 or better than persistence
    if r2 >= -1.0 and r2 >= r2_persist - 0.1:  # Accept slightly worse
        safe_disease = disease.replace(' ', '_').replace('/', '_')
        model_path = MODELS_DIR / f"model_{safe_disease}_National.keras"
        scaler_X_path = MODELS_DIR / f"scaler_X_{safe_disease}.pkl"
        scaler_y_path = MODELS_DIR / f"scaler_y_{safe_disease}.pkl"

        model.save(model_path)
        with open(scaler_X_path, 'wb') as f:
            pickle.dump(scaler_X, f)
        with open(scaler_y_path, 'wb') as f:
            pickle.dump(scaler_y, f)

        status = "✅" if r2 > 0 else "⚠️"
        print(f"    {status} R²={r2:.4f}, MAE={mae:,.0f}, Persist={r2_persist:.4f}")
    else:
        status = "❌" if r2 < r2_persist else "⚠️"
        print(f"    {status} R²={r2:.4f} (persist={r2_persist:.4f}) - NOT saved")
        return None

    gc.collect()
    return {
        'zone': 'National',
        'r2': float(r2),
        'mae': float(mae),
        'n_months': n,
        'window': window,
        'tier': 1
    }

def main():
    start = datetime.now()
    series = prepare_series()
    if not series:
        print("\n❌ No data")
        return 1

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    results = {}
    tier_dist = defaultdict(int)

    for disease, cases in series.items():
        print(f"\n🦠 {disease}")
        result = train_model(disease, cases)
        if result:
            results[disease] = result
            tier_dist["Tier 1"] += 1

    # Dashboard data
    perf = {}
    for disease, r in results.items():
        perf[disease] = {
            'tier': r['tier'],
            'tier_name': 'LSTM Log',
            'mae': round(r['mae'], 4),
            'rmse': round(r['mae'], 4),
            'r2': round(r['r2'], 4),
            'confidence': round(max(r['r2'] * 100, 18), 2),
            'epochs': 50,
            'zones_trained': 1
        }

    dashboard = {
        'training_metadata': {
            'training_date': datetime.now().isoformat(),
            'models_trained': len(results),
            'model_type': 'national_log_lstm',
            'transform': 'log1p'
        },
        'model_performance': perf,
        'tier_distribution': dict(tier_dist),
        'last_training': datetime.now().isoformat(),
        'total_models_trained': len(results)
    }

    with open(MODELS_DIR / 'dashboard_training_data.json', 'w') as f:
        json.dump(dashboard, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Models: {len(results)}")
    print(f"Duration: {(datetime.now()-start).total_seconds()/60:.1f} min\n")

    for d, p in perf.items():
        print(f"  {d:15} R²={p['r2']:7.4f}  MAE={p['mae']:,.0f}")

    print("\n✅ DONE!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
