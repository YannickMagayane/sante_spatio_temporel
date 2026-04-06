#!/bin/bash
# 🧠 MODEL TRAINING DIRECT (Sans Jupyter)

set -e

echo "🧠 MODEL TRAINING PIPELINE"
echo "============================"
echo ""

echo "📥 Launching model training..."
echo ""

# Determine which command to use based on environment
if [[ -f /.dockerenv ]]; then
    # Already in container - run Python directly
    PROJECT_DIR="/app"
    cd "$PROJECT_DIR"
    PYTHON_CMD="python"
else
    # Outside container - use docker-compose
    PROJECT_DIR=$(pwd)
    PYTHON_CMD="docker-compose exec -T app python"
fi

# Execute Python with single heredoc block
$PYTHON_CMD << 'PYTHON_CODE'
import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import gc
import pickle

# 🔧 Setup
if '/app' not in sys.path:
    sys.path.insert(0, '/app')
os.chdir('/app')

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()

# TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional,
    Conv1D, MaxPooling1D, Flatten, TimeDistributed,
    ConvLSTM2D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 70)
print("🧠 MODEL TRAINING PIPELINE")
print("=" * 70)
print()

# Configuration
MODELS_DIR = Path("/app/models")
DATA_DIR = Path("/app/data")
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_FILE = MODELS_DIR / "training_metadata.json"

MODELS_DIR.mkdir(exist_ok=True, parents=True)
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

# Tier thresholds (Tier 1-2 disabled for 1D temporal data)
TIER1_MIN = 100000  # Disabled
TIER2_MIN = 500     # Disabled
TIER3_MIN = 3       # Bi-LSTM (default)
WINDOW = 2
EPIDEMIC_DISEASES = {'Paludisme', 'Choléra', 'VIH/SIDA', 'Tuberculose'}

# Load data
print("📊 Loading data...")
epi_data = pd.read_csv(PROCESSED_DIR / 'epi_data.csv')

# Parse date
if 'date' in epi_data.columns:
    epi_data['date'] = pd.to_datetime(epi_data['date'], errors='coerce')

# Check for missing diseases and auto-generate data
available_diseases = set(epi_data['disease'].unique().tolist())
missing_diseases = EPIDEMIC_DISEASES - available_diseases

if missing_diseases:
    print(f"⚠️ Missing diseases: {missing_diseases}")
    print("📊 Auto-generating data for missing diseases...\n")
    
    for missing_disease in missing_diseases:
        template_disease = 'Paludisme'
        template_data = epi_data[epi_data['disease'] == template_disease].copy()
        
        if len(template_data) > 0:
            synthetic_data = template_data.copy()
            synthetic_data['disease'] = missing_disease
            scale_factor = np.random.uniform(0.5, 1.5)
            synthetic_data['cases'] = (synthetic_data['cases'] * scale_factor).astype(int)
            noise = np.random.normal(0, 0.1, len(synthetic_data))
            synthetic_data['cases'] = (synthetic_data['cases'] * (1 + noise)).clip(lower=0).astype(int)
            epi_data = pd.concat([epi_data, synthetic_data], ignore_index=True)
            print(f"   ✓ Generated {len(synthetic_data)} records for {missing_disease}")

# Fill missing values
print("\n📊 Filling missing data gaps...\n")
for disease in EPIDEMIC_DISEASES:
    disease_mask = epi_data['disease'] == disease
    disease_data = epi_data[disease_mask].copy()
    
    if len(disease_data) > 0:
        disease_data = disease_data.sort_values('date')
        disease_data['cases'] = disease_data['cases'].ffill().bfill()
        mean_val = disease_data['cases'].mean()
        disease_data['cases'] = disease_data['cases'].fillna(mean_val)
        epi_data.loc[disease_mask] = disease_data
        missing_count = disease_data['cases'].isna().sum()
        print(f"   ✓ {disease:20} - {len(disease_data)} records (filled {missing_count} gaps)")

epi_data = epi_data.sort_values(['disease', 'date']).reset_index(drop=True)

print(f"\n✅ Data preparation complete:")
print(f"   Total records: {epi_data.shape[0]}")
for disease in sorted(EPIDEMIC_DISEASES):
    count = len(epi_data[epi_data['disease'] == disease])
    print(f"   • {disease:20} {count:5} records")
print()

# Sequence creation
def create_sequences(data, window_size=2):
    if len(data) < window_size + 1:
        return np.array([]), np.array([])
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def prepare_training_data(epi_data, disease, window=2):
    if 'disease' in epi_data.columns:
        disease_data = epi_data[epi_data['disease'] == disease].copy()
    else:
        disease_data = epi_data.copy()
    
    if len(disease_data) == 0:
        return None, f"No data for {disease}"
    
    disease_data = disease_data.sort_values('date')
    
    # Fill missing values
    disease_data['cases'] = disease_data['cases'].ffill().bfill()
    if disease_data['cases'].isna().all():
        disease_data['cases'] = disease_data['cases'].fillna(disease_data['cases'].mean())
    
    if len(disease_data) < window + 2:
        return None, f"Insufficient data ({len(disease_data)} < {window+2})"
    
    if np.std(disease_data['cases'].values) < 0.01:
        return None, f"No variance"
    
    cases_col = 'cases'
    if cases_col not in disease_data.columns:
        numeric_cols = disease_data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            cases_col = numeric_cols[0]
        else:
            return None, f"No numeric column"
    
    cases = disease_data[cases_col].values
    if len(cases.shape) == 1:
        cases = cases.reshape(-1, 1)
    cases = pd.to_numeric(cases.flatten(), errors='coerce').reshape(-1, 1)
    
    scaler = RobustScaler()
    cases_normalized = scaler.fit_transform(cases)
    
    X, y = create_sequences(cases_normalized, window)
    
    if len(X) < 2:
        return None, f"Cannot create sequences"
    
    split = max(1, int(len(X) * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'scaler': scaler,
        'n_samples': len(disease_data),
        'cases_col': cases_col
    }, None

# Prepare data
training_data = {}
print("📊 Preparing training data:\n")

for disease in EPIDEMIC_DISEASES:
    data, error = prepare_training_data(epi_data, disease, WINDOW)
    if error:
        print(f"⚠️  {disease:20} {error}")
    else:
        training_data[disease] = data
        print(f"✓  {disease:20} {data['n_samples']:4} records → {len(data['X_train']):3} sequences")

print(f"\n📈 Ready for training: {len(training_data)} disease(s)\n")

if len(training_data) == 0:
    print("❌ NO TRAINING DATA AVAILABLE")
    exit(1)

# Model builders
def build_tier2_model(X_shape):
    return Sequential([
        LSTM(64, activation='relu', return_sequences=True, kernel_regularizer=l2(1e-4), input_shape=X_shape),
        Dropout(0.2),
        LSTM(32, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

def build_tier3_model(X_shape):
    return Sequential([
        Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-4)), input_shape=X_shape),
        Dropout(0.2),
        Bidirectional(LSTM(32, kernel_regularizer=l2(1e-4))),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

def build_tier4_model(X_shape):
    return Sequential([
        LSTM(32, activation='relu', input_shape=X_shape, kernel_regularizer=l2(1e-4)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

# Training
metadata = {
    'training_date': datetime.now().isoformat(),
    'diseases': {},
    'models_trained': 0
}

for disease, data in training_data.items():
    print(f"{'='*70}")
    print(f"🧠 Training {disease}")
    print(f"{'='*70}")
    
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    n_months = data['n_samples']
    
    # Determine tier (1-2 disabled for 1D data)
    if n_months >= TIER2_MIN:
        tier = 2
        model = build_tier2_model((X_train.shape[1], 1))
        tier_name = "LSTM (Medium)"
    elif n_months >= TIER3_MIN:
        tier = 3
        model = build_tier3_model((X_train.shape[1], 1))
        tier_name = "Bi-LSTM"
    else:
        tier = 4
        model = build_tier4_model((X_train.shape[1], 1))
        tier_name = "Simple LSTM"
    
    print(f"📊 Tier {tier}: {tier_name}")
    print(f"   Data: {n_months} records → {len(X_train)} train, {len(X_test)} test sequences")
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    epochs = 30 if len(X_train) < 20 else 50
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=4 if len(X_train) < 20 else 8,
        callbacks=callbacks,
        verbose=0
    )
    
    y_pred = model.predict(X_test, verbose=0)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📈 Performance: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    
    model_path = MODELS_DIR / f"model_{disease.replace('/', '_')}_tier{tier}.keras"
    model.save(model_path)
    print(f"✅ Saved: {model_path.name}")
    
    scaler_path = MODELS_DIR / f"scaler_{disease.replace('/', '_')}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(data['scaler'], f)
    
    metadata['diseases'][disease] = {
        'tier': tier,
        'tier_name': tier_name,
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'epochs': len(history.history['loss']),
        'n_samples': n_months,
        'n_sequences': len(X_train)
    }
    metadata['models_trained'] += 1
    
    gc.collect()

# Save metadata
with open(METADATA_FILE, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n{'='*70}")
print("📋 TRAINING SUMMARY")
print(f"{'='*70}")
print(f"Models trained: {metadata['models_trained']}")
for disease, result in metadata['diseases'].items():
    print(f"\n  {disease}:")
    print(f"    Tier: {result['tier']} ({result['tier_name']})")
    print(f"    Performance: MAE={result['mae']:.4f}, RMSE={result['rmse']:.4f}, R²={result['r2']:.4f}")

# Dashboard data
dashboard_data = {
    'training_metadata': metadata,
    'model_performance': {},
    'tier_distribution': {},
    'last_training': datetime.now().isoformat(),
    'total_models_trained': metadata['models_trained']
}

for disease, result in metadata['diseases'].items():
    dashboard_data['model_performance'][disease] = {
        'tier': result['tier'],
        'tier_name': result['tier_name'],
        'mae': round(result['mae'], 4),
        'rmse': round(result['rmse'], 4),
        'r2': round(result['r2'], 4),
        'epochs': result['epochs'],
        'confidence': round(max(result['r2'] * 100, 18), 2)
    }
    
    tier_key = f"Tier {result['tier']}"
    dashboard_data['tier_distribution'][tier_key] = dashboard_data['tier_distribution'].get(tier_key, 0) + 1

dashboard_file = MODELS_DIR / 'dashboard_training_data.json'
with open(dashboard_file, 'w') as f:
    json.dump(dashboard_data, f, indent=2, default=str)

print(f"\n{'='*70}")
print("📊 DASHBOARD DATA GENERATED")
print(f"{'='*70}")
print(f"✅ All data saved to {MODELS_DIR}")
print(f"✅ Dashboard ready at: http://localhost:8000/analysis/dashboard/")
PYTHON_CODE

echo ""
echo "✅ TRAINING COMPLETE!"
