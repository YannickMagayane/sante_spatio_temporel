#!/bin/bash
# 🎯 TRAINING SIMPLIFIED - LSTM models for 1D time series

cd /home/yannick-magayane/Bureau/projets/data_analyse/health

echo "🚀 ENTRAÎNEMENT SIMPLIFIÉ"
echo "=========================="
echo ""

docker-compose exec -T app python << 'PYTHON_CODE'
import os
import sys
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.chdir('/app')
sys.path.insert(0, '/app')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 70)
print("🧠 SIMPLIFIED MODEL TRAINING PIPELINE")
print("=" * 70)
print()

MODELS_DIR = Path("/app/models")
DATA_DIR = Path("/app/data")
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_FILE = MODELS_DIR / "training_metadata.json"

# Load data
print("📥 Loading data...")
epi_data = pd.read_csv(PROCESSED_DIR / 'epi_data.csv')
epi_data['date'] = pd.to_datetime(epi_data['date'])
print(f"✓ Loaded: {len(epi_data)} records")
print()

# Configuration
WINDOW = 4
EPIDEMIC_DISEASES = {
    'Paludisme': 'Paludisme',
    'Tuberculose': 'Tuberculose',
    'VIH/SIDA': 'VIH/SIDA',
}

# Helper functions
def create_sequences(data, window_size=4):
    """Create windowed sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def prepare_training_data(epi_data, disease, window=4):
    """Prepare training data for a single disease"""
    disease_data = epi_data[epi_data['disease'] == disease].sort_values('date')
    
    if len(disease_data) < window + 2:
        return None, f"Insufficient data ({len(disease_data)} < {window + 2})"
    
    # Get cases as 1D array
    scaler = RobustScaler()
    cases = disease_data['cases'].values.reshape(-1, 1)
    cases_normalized = scaler.fit_transform(cases)
    
    # Create sequences: (samples, window, 1)
    X, y = create_sequences(cases_normalized, window)
    
    if len(X) < 3:
        return None, f"Not enough sequences ({len(X)} < 3)"
    
    # 80/20 split
    split = max(1, int(len(X) * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'scaler': scaler,
        'n_samples': len(disease_data)
    }, None

def build_lstm_model(input_shape):
    """Simple LSTM model for 1D time series"""
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(16, return_sequences=False),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    return model

# Prepare data
print("📊 Preparing data...")
training_data = {}

for disease_key, disease_name in EPIDEMIC_DISEASES.items():
    data, error = prepare_training_data(epi_data, disease_name, WINDOW)
    if error:
        print(f"  ⚠️ {disease_name}: {error}")
    else:
        training_data[disease_name] = data
        print(f"  ✓ {disease_name}: {len(data['X_train'])} train, {len(data['X_test'])} test")

# Train models
metadata = {
    'training_date': datetime.now().isoformat(),
    'diseases': {},
    'models_trained': 0
}

print()

for disease, data in training_data.items():
    print(f"\n{'='*70}")
    print(f"🧠 Training {disease}")
    print(f"{'='*70}")
    
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    print(f"📊 Shape: X_train {X_train.shape}, y_train {y_train.shape}")
    
    # Build model
    model = build_lstm_model((X_train.shape[1], 1))
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Train with detailed output
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=4,
        callbacks=callbacks,
        verbose=0
    )
    
    # Evaluate
    y_pred = model.predict(X_test, verbose=0)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📈 Results:")
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²:   {r2:.4f}")
    print(f"   Epochs: {len(history.history['loss'])}")
    
    # Save model
    model_path = MODELS_DIR / f"model_{disease.replace('/', '_')}_lstm.keras"
    model.save(model_path)
    print(f"✅ Model saved: {model_path.name}")
    
    # Save scaler
    scaler_path = MODELS_DIR / f"scaler_{disease.replace('/', '_')}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(data['scaler'], f)
    
    # Save metadata
    metadata['diseases'][disease] = {
        'model': 'LSTM',
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'epochs': len(history.history['loss']),
        'input_shape': str(X_train.shape)
    }
    metadata['models_trained'] += 1

# Save metadata
with open(METADATA_FILE, 'w') as f:
    json.dump(metadata, f, indent=2)

# Generate dashboard data
dashboard_data = {
    'training_metadata': metadata,
    'model_performance': {},
    'summary': {}
}

for disease, result in metadata['diseases'].items():
    dashboard_data['model_performance'][disease] = {
        'model': result['model'],
        'r2': round(result['r2'], 4),
        'mae': round(result['mae'], 4),
        'confidence': round(max(result['r2'] * 100, 20), 2)
    }

dashboard_data['last_training'] = datetime.now().isoformat()
dashboard_data['total_models_trained'] = metadata['models_trained']

dashboard_file = MODELS_DIR / 'dashboard_training_data.json'
with open(dashboard_file, 'w') as f:
    json.dump(dashboard_data, f, indent=2, default=str)

print(f"\n{'='*70}")
print("✅ TRAINING COMPLETE")
print(f"{'='*70}")
print(f"\nModels trained: {metadata['models_trained']}")

if metadata['models_trained'] > 0:
    print(f"\n📊 Model Performance:")
    for disease, perf in dashboard_data['model_performance'].items():
        print(f"\n  {disease}:")
        print(f"    Model: {perf['model']}")
        print(f"    R² Score: {perf['r2']:.4f}")
        print(f"    MAE: {perf['mae']:.4f}")
        print(f"    Confidence: {perf['confidence']:.1f}%")

print(f"\n📊 Dashboard data: {dashboard_file.name}")
print(f"👉 View: http://localhost:8000/")

PYTHON_CODE

echo ""
echo "✅ ENTRAÎNEMENT TERMINÉ!"
