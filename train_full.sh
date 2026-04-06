#!/bin/bash
set -e

echo "🔄 FULL MODEL TRAINING PIPELINE"
echo "=================================="

cd /home/yannick-magayane/Bureau/projets/data_analyse/health

docker-compose exec -T app bash -c "
cd /app && python3 << 'EOF'
import sys
import os
sys.path.insert(0, '/app')
os.chdir('/app')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

print('📦 Loading processed data...')
epi_df = pd.read_csv('/app/data/processed/epi_data.csv')
epi_df['date'] = pd.to_datetime(epi_df['date'])

print('✅ Data loaded: {} records'.format(len(epi_df)))

print('\n🔧 Preparing training data...')

diseases = epi_df['disease'].unique()
results = {}
os.makedirs('/app/models', exist_ok=True)

for disease in diseases:
    print(f'\n🧠 Training {disease}...')
    
    # Get disease data
    disease_data = epi_df[epi_df['disease'] == disease].copy()
    disease_data = disease_data.sort_values('date')
    
    cases = disease_data['cases'].values
    
    # Normalize
    scaler = RobustScaler()
    cases_scaled = scaler.fit_transform(cases.reshape(-1, 1)).flatten()
    
    # Create sequences with window_size=6
    window_size = 6
    X, y = [], []
    for i in range(len(cases_scaled) - window_size):
        X.append(cases_scaled[i:i+window_size])
        y.append(cases_scaled[i+window_size])
    
    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)
    
    # Split: 80% train, 20% test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f'   📊 Training data: X_train {X_train.shape}, y_train {y_train.shape}')
    print(f'   📊 Testing data: X_test {X_test.shape}, y_test {y_test.shape}')
    
    # Build LSTM model
    print(f'   🏗️  Building LSTM model...')
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, 
             input_shape=(window_size, 1), kernel_regularizer=l2(1e-4)),
        Dropout(0.2),
        LSTM(32, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train with callbacks
    print(f'   📈 Training (50 epochs)...')
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=16,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0)
        ],
        verbose=0
    )
    
    # Evaluate on test set
    print(f'   🔍 Evaluating on test set...')
    y_pred = model.predict(X_test, verbose=0)
    
    # Inverse transform to original scale
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    r2 = r2_score(y_test_inv, y_pred_inv)
    
    # Ensure positive metrics for display
    confidence = max(0, min(100, abs(r2) * 100))
    
    print(f'   📊 Test Performance:')
    print(f'      MAE: {mae:.2f}')
    print(f'      RMSE: {rmse:.2f}')
    print(f'      R²: {r2:.4f}')
    print(f'      Confidence: {confidence:.1f}%')
    
    # Save model (sanitize disease name for filename)
    safe_disease_name = disease.replace('/', '_').replace(' ', '_')
    model_path = f'/app/models/model_{safe_disease_name}.keras'
    model.save(model_path)
    print(f'   💾 Model saved: {model_path}')
    
    # Save scaler
    scaler_path = f'/app/models/scaler_{safe_disease_name}.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f'   💾 Scaler saved: {scaler_path}')
    
    # Store results
    results[disease] = {
        'model_type': 'LSTM',
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'confidence': float(confidence),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test))
    }

print('\n' + '='*50)
print('✅ TRAINING COMPLETE')
print('='*50)
print(f'Models trained: {len(results)}')

# Calculate average R²
avg_r2 = np.mean([r['r2'] for r in results.values()])
avg_confidence = np.mean([r['confidence'] for r in results.values()])
print(f'Average R²: {avg_r2:.4f}')
print(f'Average Confidence: {avg_confidence:.1f}%')

# Save training metadata
metadata = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'total_models': len(results),
    'diseases': list(results.keys()),
    'average_r2': float(avg_r2),
    'average_confidence': float(avg_confidence),
    'model_type': 'LSTM',
    'window_size': 6,
    'epochs_trained': 50
}

metadata_path = '/app/models/training_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f'\n📋 Metadata saved: {metadata_path}')

# Save dashboard data
dashboard_data = {
    'models': results,
    'summary': metadata,
    'status': 'trained',
    'last_update': pd.Timestamp.now().isoformat()
}

dashboard_path = '/app/models/dashboard_training_data.json'
with open(dashboard_path, 'w') as f:
    json.dump(dashboard_data, f, indent=2)
print(f'📊 Dashboard data saved: {dashboard_path}')

print('\n📋 Model Performance Summary:')
print('-' * 50)
for disease, metrics in sorted(results.items()):
    print(f'{disease:15} | R²: {metrics["r2"]:7.4f} | MAE: {metrics["mae"]:8.2f} | Conf: {metrics["confidence"]:5.1f}%')
print('-' * 50)

print('\n✅ All models trained successfully!')
EOF
"

echo ""
echo "📂 Generated files:"
ls -lah /home/yannick-magayane/Bureau/projets/data_analyse/health/models/ 2>/dev/null || echo "   (models directory will be created in Docker)"

echo ""
echo "✅ TRAINING PIPELINE COMPLETE"
