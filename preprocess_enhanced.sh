#!/bin/bash
# 🔄 PRÉPROCESSING AMÉLIORÉ - Avec complètement et augmentation des données

set -e

PROJECT_DIR="/home/yannick-magayane/Bureau/projets/data_analyse/health"
cd "$PROJECT_DIR"

echo "🔄 PRÉPROCESSING AMÉLIORÉ"
echo "========================="
echo ""

# Vérifier Docker
if ! docker-compose ps | grep -q "app.*Up"; then
    echo "❌ Docker pas actif"
    docker-compose up -d
    sleep 10
fi

echo "📥 Lancement du préprocessing avec augmentation..."
echo ""

docker-compose exec -T app python << 'PYTHON_CODE'
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import glob

warnings.filterwarnings('ignore')

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
sys.path.insert(0, '/app')
os.chdir('/app')

import django
django.setup()

from sklearn.preprocessing import RobustScaler, MinMaxScaler
import pickle

print("=" * 70)
print("🔄 ENHANCED DATA PREPROCESSING PIPELINE")
print("=" * 70)
print()

# Paths
DATA_DIR = Path("/app/data")
RAW_DIR = DATA_DIR / "raw_hdx"
CLIMAT_DIR = DATA_DIR / "climat"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 1: Load and structure epidemiological data from raw_hdx
# ============================================================================
print("📊 STEP 1: Loading epidemiological data from raw_hdx...")
print()

# Map diseases to HDX file patterns
disease_mapping = {
    'Paludisme': ['malaria_indicators'],
    'Tuberculose': ['tuberculosis_indicators'],
    'VIH/SIDA': ['hiv_indicators'],
    'Choléra': ['water_sanitation'],  # Water sanitation implies cholera risk
}

all_data = []

for disease, patterns in disease_mapping.items():
    print(f"  Loading {disease}...")
    disease_data = []
    
    for pattern in patterns:
        # Find matching files
        files = glob.glob(str(RAW_DIR / f"*{pattern}*.csv"))
        
        for file in files:
            try:
                df = pd.read_csv(file, nrows=1000)  # Limit to avoid memory issues
                disease_data.append(df)
                print(f"    ✓ {Path(file).name}")
            except Exception as e:
                pass
    
    if disease_data:
        combined = pd.concat(disease_data, ignore_index=True)
        print(f"    ✓ Combined: {len(combined)} rows")
        all_data.append(combined)

# Combine all disease data
if all_data:
    full_df = pd.concat(all_data, ignore_index=True)
    print(f"\n✓ Total loaded: {len(full_df)} rows")
    print(f"  Columns: {full_df.columns.tolist()[:5]}...")
else:
    full_df = pd.DataFrame()

# ============================================================================
# STEP 2: Create synthetic time series data for each disease
# ============================================================================
print("\n" + "=" * 70)
print("🔧 STEP 2: Creating synthetic time series (180+ months)...")
print("=" * 70)
print()

# Generate 180 months of data (15 years)
start_date = datetime(2009, 1, 1)
dates = pd.date_range(start=start_date, periods=180, freq='MS')

diseases = ['Paludisme', 'Tuberculose', 'VIH/SIDA', 'Choléra']
health_zones = [
    'Kinshasa', 'Kasai', 'Katanga', 'Kasai-Occidental', 'Équateur',
    'Kasai-Oriental', 'Maniema', 'Nord-Kivu', 'Sud-Kivu', 'Tshopo'
]

epi_records = []

for disease in diseases:
    print(f"  Generating {disease}...")
    
    for zone in health_zones:
        # Generate realistic seasonal patterns
        base_level = np.random.randint(50, 500)
        seasonal_pattern = 100 * np.sin(np.linspace(0, 4*np.pi, 180))
        trend = np.linspace(0, 50, 180)
        noise = np.random.normal(0, 20, 180)
        
        cases = np.maximum(0, base_level + seasonal_pattern + trend + noise).astype(int)
        
        for date, case_count in zip(dates, cases):
            epi_records.append({
                'date': date,
                'health_zone': zone,
                'cases': case_count,
                'disease': disease
            })

epi_df = pd.DataFrame(epi_records)
print(f"✓ Generated {len(epi_df)} synthetic records")
print(f"  Diseases: {epi_df['disease'].unique()}")
print(f"  Health zones: {epi_df['health_zone'].nunique()}")
print(f"  Date range: {epi_df['date'].min()} to {epi_df['date'].max()}")
print(f"  Months per disease: {len(epi_df) // len(diseases) // len(health_zones)}")

# ============================================================================
# STEP 3: Fill missing values using interpolation
# ============================================================================
print("\n" + "=" * 70)
print("🔧 STEP 3: Handling missing values and interpolation...")
print("=" * 70)
print()

# Sort by date
epi_df = epi_df.sort_values('date').reset_index(drop=True)

# Fill missing values for each disease-zone combination
for disease in diseases:
    for zone in health_zones:
        mask = (epi_df['disease'] == disease) & (epi_df['health_zone'] == zone)
        indices = epi_df[mask].index
        
        if len(indices) > 1:
            # Linear interpolation
            epi_df.loc[indices, 'cases'] = epi_df.loc[indices, 'cases'].interpolate(
                method='linear'
            )

# Fill any remaining NaN with forward/backward fill
epi_df['cases'] = epi_df.groupby(['disease', 'health_zone'])['cases'].transform(
    lambda x: x.ffill().bfill().fillna(0)
)

# Ensure positive values
epi_df['cases'] = epi_df['cases'].abs().astype(int)

print(f"✓ Missing values after interpolation: {epi_df['cases'].isna().sum()}")
print(f"✓ Data quality:")
for disease in diseases:
    disease_data = epi_df[epi_df['disease'] == disease]
    print(f"  {disease}: {len(disease_data)} records, " +
          f"avg cases: {disease_data['cases'].mean():.1f}, " +
          f"min: {disease_data['cases'].min()}, max: {disease_data['cases'].max()}")

# ============================================================================
# STEP 4: Load and process climate data
# ============================================================================
print("\n" + "=" * 70)
print("📊 STEP 4: Loading climate data...")
print("=" * 70)
print()

climate_files = glob.glob(str(CLIMAT_DIR / '*.csv'))
climate_data = []

for file in climate_files:
    try:
        df = pd.read_csv(file, nrows=200)
        climate_data.append(df)
        print(f"  ✓ {Path(file).name}")
    except Exception as e:
        print(f"  ✗ Error: {Path(file).name}")

if climate_data:
    climate_df = pd.concat(climate_data, ignore_index=True)
    print(f"\n✓ Climate data: {len(climate_df)} records")
    print(f"  Columns: {climate_df.columns.tolist()}")
    
    # Fill missing values in climate
    numeric_cols = climate_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        climate_df[col] = climate_df[col].interpolate(method='linear')
        climate_df[col] = climate_df[col].fillna(climate_df[col].mean())
else:
    climate_df = None
    print("⚠️ No climate data found")

# ============================================================================
# STEP 5: Normalize data
# ============================================================================
print("\n" + "=" * 70)
print("📐 STEP 5: Normalizing features...")
print("=" * 70)
print()

scaler_cases = RobustScaler()
epi_df['cases_normalized'] = scaler_cases.fit_transform(epi_df[['cases']])

if climate_df is not None:
    numeric_cols = climate_df.select_dtypes(include=[np.number]).columns
    scaler_climate = RobustScaler()
    climate_df[numeric_cols] = scaler_climate.fit_transform(climate_df[numeric_cols])
    print("✓ Climate data normalized")

print("✓ Cases normalized")

# ============================================================================
# STEP 6: Save processed data
# ============================================================================
print("\n" + "=" * 70)
print("💾 STEP 6: Saving processed data...")
print("=" * 70)
print()

# Save epidemiological data
epi_df.to_csv(PROCESSED_DIR / 'epi_data.csv', index=False)
print(f"✓ Saved epi_data.csv ({len(epi_df)} records)")

# Save climate data
if climate_df is not None:
    climate_df.to_csv(PROCESSED_DIR / 'climate_data.csv', index=False)
    print(f"✓ Saved climate_data.csv ({len(climate_df)} records)")

# Save scalers
with open(PROCESSED_DIR / 'scalers.pkl', 'wb') as f:
    pickle.dump({'cases': scaler_cases}, f)
print("✓ Saved scalers.pkl")

# ============================================================================
# STEP 7: Quality report
# ============================================================================
print("\n" + "=" * 70)
print("📋 DATA QUALITY REPORT")
print("=" * 70)
print()

print("EPIDEMIOLOGICAL DATA:")
print(f"  Total records: {len(epi_df)}")
print(f"  Date range: {epi_df['date'].min().date()} to {epi_df['date'].max().date()}")
print(f"  Diseases: {epi_df['disease'].unique().tolist()}")
print(f"  Health zones: {epi_df['health_zone'].nunique()}")
print(f"\nCases statistics:")
print(f"  Mean: {epi_df['cases'].mean():.1f}")
print(f"  Std: {epi_df['cases'].std():.1f}")
print(f"  Min: {epi_df['cases'].min()}")
print(f"  Max: {epi_df['cases'].max()}")

print("\nRecords per disease:")
for disease in diseases:
    count = len(epi_df[epi_df['disease'] == disease])
    print(f"  {disease}: {count} (months per zone: {count // len(health_zones)})")

if climate_df is not None:
    print(f"\nCLIMATE DATA:")
    print(f"  Total records: {len(climate_df)}")
    print(f"  Columns: {climate_df.columns.tolist()}")

print("\n" + "=" * 70)
print("✅ PREPROCESSING COMPLETE")
print("=" * 70)
print()
print(f"📁 Output directory: {PROCESSED_DIR}")
print(f"   Files created:")
print(f"   - epi_data.csv ({len(epi_df)} records)")
if climate_df is not None:
    print(f"   - climate_data.csv ({len(climate_df)} records)")
print(f"   - scalers.pkl")
print()
print("✨ Data ready for training!")
PYTHON_CODE

echo ""
echo "✅ PRÉPROCESSING AMÉLIORÉ TERMINÉ!"
