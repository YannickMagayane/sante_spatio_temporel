#!/bin/bash
# Script to make predictions using trained LSTM models

set -e

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
import pickle
import json
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print('🔮 DISEASE PREDICTION SYSTEM')
print('='*50)

# Define diseases and safe names
diseases = ['Paludisme', 'Tuberculose', 'Choléra', 'VIH/SIDA']
disease_names_safe = {
    'Paludisme': 'Paludisme',
    'Tuberculose': 'Tuberculose',
    'Choléra': 'Choléra',
    'VIH/SIDA': 'VIH_SIDA'
}

print('\n📦 Loading trained models...')
models = {}
scalers = {}

for disease in diseases:
    safe_name = disease_names_safe[disease]
    model_path = f'/app/models/model_{safe_name}.keras'
    scaler_path = f'/app/models/scaler_{safe_name}.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        models[disease] = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scalers[disease] = pickle.load(f)
        print(f'✅ Loaded: {disease}')
    else:
        print(f'❌ Missing: {disease}')

print(f'\n✅ Loaded {len(models)} models')

# Load processed data
print('\n📂 Loading processed data...')
epi_df = pd.read_csv('/app/data/processed/epi_data.csv')
epi_df['date'] = pd.to_datetime(epi_df['date'])

print(f'✅ Loaded {len(epi_df)} records')

# Make predictions for next 12 months
print('\n🔮 Generating predictions for next 12 months...')

predictions = {}
last_date = epi_df['date'].max()
window_size = 6

for disease in diseases:
    if disease not in models:
        continue
    
    print(f'\n📈 Predicting {disease}...')
    
    # Get last window of data for this disease
    disease_data = epi_df[epi_df['disease'] == disease].sort_values('date')
    cases = disease_data['cases'].values[-window_size:]
    
    # Normalize using the scaler
    scaler = scalers[disease]
    cases_scaled = scaler.transform(cases.reshape(-1, 1)).flatten()
    
    # Prepare for prediction
    X_pred = cases_scaled.reshape(1, window_size, 1)
    
    # Make predictions for next 12 months
    forecast = []
    pred_dates = []
    
    for i in range(12):
        next_month = last_date + timedelta(days=30*(i+1))
        
        # Predict next value
        next_scaled = models[disease].predict(X_pred, verbose=0)[0, 0]
        forecast.append(next_scaled)
        pred_dates.append(next_month)
        
        # Shift window for next prediction
        X_pred = np.append(X_pred[0, 1:, :], [[next_scaled]], axis=0)
        X_pred = X_pred.reshape(1, window_size, 1)
    
    # Inverse transform predictions
    forecast_inv = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    
    # Store predictions
    predictions[disease] = {
        'dates': [d.isoformat() for d in pred_dates],
        'forecasts': forecast_inv.tolist(),
        'confidence': 0.6
    }
    
    print(f'   ✅ Generated 12-month forecast')
    print(f'   Predicted range: {forecast_inv.min():.0f} - {forecast_inv.max():.0f} cases')

# Save predictions
predictions_path = '/app/models/predictions.json'
with open(predictions_path, 'w') as f:
    json.dump(predictions, f, indent=2)
print(f'\n💾 Predictions saved: {predictions_path}')

# Create summary
summary = {
    'prediction_date': last_date.isoformat(),
    'prediction_horizon': '12 months',
    'diseases': list(predictions.keys()),
    'total_predictions': sum(len(p['forecasts']) for p in predictions.values())
}

print('\n📊 Prediction Summary:')
print(f'   Prediction date: {last_date.date()}')
print(f'   Horizon: 12 months')
print(f'   Diseases: {len(predictions)}')
print(f'   Total forecasts: {summary["total_predictions"]}')

print('\n✅ PREDICTIONS COMPLETE')
EOF
"

echo ""
echo "✅ Prediction script completed"
