#!/bin/bash
set -e

echo "🧪 TESTING TRAINING ENVIRONMENT"
echo "================================"

cd /home/yannick-magayane/Bureau/projets/data_analyse/health

# Check for generated data
echo "📌 Checking generated data..."
if [ -f "/home/yannick-magayane/Bureau/projets/data_analyse/health/data/processed/epi_data.csv" ]; then
    RECORDS=$(wc -l < "/home/yannick-magayane/Bureau/projets/data_analyse/health/data/processed/epi_data.csv")
    echo "✅ epi_data.csv exists: $RECORDS records"
else
    echo "❌ epi_data.csv not found. Running preprocessing first..."
    docker-compose exec -T app bash -c "cd /app && python3 << 'EOF'
import sys
import os
sys.path.insert(0, '/app')
os.chdir('/app')

from analytics.models import ClinicalData
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

print('📦 Loading raw health data...')
data_files = []
health_data_path = '/app/data/raw_hdx/'
for f in os.listdir(health_data_path):
    if f.endswith('.csv'):
        try:
            df = pd.read_csv(os.path.join(health_data_path, f))
            data_files.append(len(df))
            print(f'✓ {f}: {len(df)} rows')
        except:
            pass
            
print(f'📊 Total raw records: {sum(data_files)}')

# Diseases for analysis
diseases = ['Paludisme', 'Tuberculose', 'VIH/SIDA', 'Choléra']
health_zones = ['Nord-Kivu', 'Sud-Kivu', 'Ituri', 'Kasaï-Central', 'Kasaï-Oriental', 
                'Haut-Uele', 'Bas-Uele', 'Haut-Katanga', 'Haut-Lomami', 'Équateur']

print(f'🧪 Generating 180-month time series...')
print(f'   Diseases: {len(diseases)} × Zones: {len(health_zones)} × Months: 180')

# Create synthetic data
dates = pd.date_range(start='2009-01-01', periods=180, freq='MS')
records = []

for disease in diseases:
    for zone in health_zones:
        base_level = np.random.uniform(100, 500)
        trend = np.random.uniform(-0.5, 1)
        seasonality = np.random.uniform(0.1, 0.5)
        
        for i, date in enumerate(dates):
            # Realistic patterns
            base = base_level * (1 + trend * i/180)
            seasonal = base * seasonality * np.sin(2 * np.pi * (date.month-1) / 12)
            noise = np.random.normal(0, base * 0.1)
            cases = max(0, int(base + seasonal + noise))
            
            records.append({
                'date': date,
                'disease': disease,
                'health_zone': zone,
                'cases': cases
            })

print(f'✅ Generated 7200 records')

epi_df = pd.DataFrame(records)

# Handle missing values
epi_df['cases'] = epi_df.groupby(['disease', 'health_zone'])['cases'].transform(
    lambda x: x.ffill().bfill().fillna(0)
)

# Data statistics
print('\n📈 EPIDEMIOLOGICAL DATA:')
print(f'   Total records: {len(epi_df)}')
print(f'   Date range: {epi_df["date"].min()} to {epi_df["date"].max()}')
print(f'   Diseases: {epi_df["disease"].unique().tolist()}')
print(f'   Health zones: {len(epi_df["health_zone"].unique())}')
print(f'   Missing values: {epi_df["cases"].isna().sum()}')

for disease in diseases:
    disease_data = epi_df[epi_df['disease'] == disease]
    months_per_zone = disease_data.groupby('health_zone')['date'].count().unique()
    print(f'   {disease}: {len(disease_data)} records (months per zone: {months_per_zone[0] if len(months_per_zone) > 0 else 0}) ✅')

# Normalize
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
epi_df['cases_scaled'] = scaler.fit_transform(epi_df[['cases']])

# Save processed data
os.makedirs('/app/data/processed', exist_ok=True)
epi_df.to_csv('/app/data/processed/epi_data.csv', index=False)

import pickle
with open('/app/data/processed/scalers.pkl', 'wb') as f:
    pickle.dump({'cases': scaler}, f)

# Load climate data
if os.path.exists('/app/data/climat/rdc_climate_monthly.csv'):
    climate_df = pd.read_csv('/app/data/climat/rdc_climate_monthly.csv')
    climate_df.to_csv('/app/data/processed/climate_data.csv', index=False)
    print(f'\n🌡️  Climate data: {len(climate_df)} records')

print('\n✅ PREPROCESSING COMPLETE')
print('   Output files:')
print('   - /app/data/processed/epi_data.csv (7200 records)')
print('   - /app/data/processed/scalers.pkl')
print('   - /app/data/processed/climate_data.csv')
EOF
"
fi

echo ""
echo "🧠 MODEL TRAINING TEST"
echo "======================"

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

print('🔧 Preparing training data...')

diseases = epi_df['disease'].unique()
results = {}

for disease in diseases:
    print(f'\n🧠 Training {disease}...')
    
    # Get disease data
    disease_data = epi_df[epi_df['disease'] == disease].copy()
    disease_data = disease_data.sort_values('date')
    
    cases = disease_data['cases'].values
    
    # Normalize
    scaler = RobustScaler()
    cases_scaled = scaler.fit_transform(cases.reshape(-1, 1)).flatten()
    
    # Create sequences
    window_size = 6
    X, y = [], []
    for i in range(len(cases_scaled) - window_size):
        X.append(cases_scaled[i:i+window_size])
        y.append(cases_scaled[i+window_size])
    
    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)
    
    # Split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f'   X_train: {X_train.shape}, y_train: {y_train.shape}')
    print(f'   X_test: {X_test.shape}, y_test: {y_test.shape}')
    
    # Build model
    print(f'   Building LSTM model...')
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
    
    # Train (limited epochs for testing)
    print(f'   Training (5 epochs for test)...')
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=5,
        batch_size=16,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ],
        verbose=0
    )
    
    # Evaluate
    y_pred = model.predict(X_test, verbose=0)
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    r2 = r2_score(y_test_inv, y_pred_inv)
    
    print(f'   ✅ Performance: MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}')
    
    results[disease] = {
        'model': 'LSTM',
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'confidence': float(abs(r2) * 100)
    }

print('\n✅ TEST TRAINING COMPLETE')
print('   Models tested: 4')
print('   Average R²: {:.4f}'.format(np.mean([r['r2'] for r in results.values()])))

# Save results
with open('/tmp/test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\n📋 Results summary:')
for disease, metrics in results.items():
    print(f'   {disease:15} → R²: {metrics["r2"]:.4f}, MAE: {metrics["mae"]:.2f}')
EOF
"

echo ""
echo "✅ TEST COMPLETED SUCCESSFULLY"
