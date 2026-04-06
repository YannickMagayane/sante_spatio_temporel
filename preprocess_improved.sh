#!/bin/bash
# 🔄 PRÉPROCESSING AMÉLIORÉ - Extrait les données santé et climat

PROJECT_DIR="/home/yannick-magayane/Bureau/projets/data_analyse/health"
cd "$PROJECT_DIR"

echo "🔄 PRÉPROCESSING AMÉLIORÉ"
echo "========================="
echo ""

docker-compose exec -T app python << 'PYTHON_CODE'
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
import pickle
import warnings

warnings.filterwarnings('ignore')

os.chdir('/app')
sys.path.insert(0, '/app')

print("=" * 70)
print("🔄 ADVANCED DATA PREPROCESSING")
print("=" * 70)
print()

DATA_DIR = Path("/app/data")
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ===== LOAD HEALTH DATA FROM RAW_HDX =====
print("📊 Loading health indicators from raw_hdx...")

import glob

epi_dfs = []
raw_hdx_dir = DATA_DIR / "raw_hdx"

# Files with disease data
malaria_file = raw_hdx_dir / "malaria_indicators_cod.csv"
tb_file = raw_hdx_dir / "tuberculosis_indicators_cod.csv"
hiv_file = raw_hdx_dir / "hiv_indicators_cod.csv"

files_to_load = {
    'Paludisme': malaria_file,
    'Tuberculose': tb_file,
    'VIH/SIDA': hiv_file,
}

# Load disease-specific files
for disease, file in files_to_load.items():
    if file.exists():
        try:
            df = pd.read_csv(file)
            print(f"  ✓ {disease}: {len(df)} rows, columns: {df.columns.tolist()[:3]}")
            
            # Extract numeric columns as cases
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                # Get the first numeric column as 'cases'
                df_clean = pd.DataFrame({
                    'disease': disease,
                    'cases': df[numeric_cols[0]].fillna(0),
                    'year': df.get('YEAR (DISPLAY)', datetime.now().year),
                })
                
                # Create date from year
                df_clean['date'] = pd.to_datetime(df_clean['year'].astype(str) + '-01-01')
                df_clean = df_clean[['date', 'disease', 'cases']]
                
                epi_dfs.append(df_clean)
                print(f"    → Extracted {len(df_clean)} cases")
        except Exception as e:
            print(f"  ✗ Error loading {disease}: {e}")

# Combine all disease data
if epi_dfs:
    print(f"\n  Combining {len(epi_dfs)} disease datasets...")
    epi_df = pd.concat(epi_dfs, ignore_index=True)
    
    # Ensure columns are correct type
    epi_df['date'] = pd.to_datetime(epi_df['date'], errors='coerce')
    epi_df['cases'] = pd.to_numeric(epi_df['cases'], errors='coerce').fillna(0)
    epi_df['deaths'] = epi_df['cases'] * 0.1  # Estimate
    
    # Sort by date
    epi_df = epi_df.sort_values('date').reset_index(drop=True)
    
    print(f"✓ Combined: {len(epi_df)} total records")
    print(f"  Diseases: {epi_df['disease'].unique()}")
    print(f"  Date range: {epi_df['date'].min()} to {epi_df['date'].max()}")
else:
    print("⚠️ No disease data found")
    epi_df = pd.DataFrame()

# ===== LOAD CLIMATE DATA =====
print("\n📊 Loading climate data...")

climate_file = DATA_DIR / "climat" / "rdc_climate_monthly.csv"

if climate_file.exists():
    climate_df = pd.read_csv(climate_file)
    print(f"✓ Climate data: {climate_df.shape}")
    print(f"  Columns: {climate_df.columns.tolist()}")
else:
    print("⚠️ Climate file not found")
    climate_df = None

# ===== PREPROCESSING =====
print("\n🔧 Preprocessing...")

if len(epi_df) > 0:
    # Normalize cases
    scaler_cases = RobustScaler()
    epi_df['cases_normalized'] = scaler_cases.fit_transform(
        epi_df[['cases']]
    )
    
    # Save epidemiological data
    epi_df.to_csv(PROCESSED_DIR / 'epi_data.csv', index=False)
    print(f"✓ Saved epi_data.csv ({len(epi_df)} records)")
    
    # Save scaler
    with open(PROCESSED_DIR / 'scalers.pkl', 'wb') as f:
        pickle.dump({'cases': scaler_cases}, f)
    print("✓ Saved scalers.pkl")

if climate_df is not None:
    climate_df.to_csv(PROCESSED_DIR / 'climate_data.csv', index=False)
    print(f"✓ Saved climate_data.csv ({len(climate_df)} records)")

# ===== SUMMARY =====
print("\n" + "=" * 70)
print("📊 DATA SUMMARY")
print("=" * 70)

if len(epi_df) > 0:
    print(f"\nEpidemiological Data:")
    print(f"  Records: {len(epi_df)}")
    print(f"  Diseases: {epi_df['disease'].unique()}")
    print(f"  Cases range: {epi_df['cases'].min():.0f} - {epi_df['cases'].max():.0f}")
    print(f"\n  Sample data:")
    print(epi_df.head(3).to_string(index=False))

if climate_df is not None:
    print(f"\nClimate Data:")
    print(f"  Records: {len(climate_df)}")
    print(f"  Columns: {climate_df.columns.tolist()}")

print("\n" + "=" * 70)
print("✅ PREPROCESSING COMPLETE")
print("=" * 70)
PYTHON_CODE

echo ""
echo "✅ PRÉPROCESSING AMÉLIORÉ TERMINÉ!"
