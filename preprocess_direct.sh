#!/bin/bash
# 🔄 PRÉPROCESSING DIRECT (Sans Jupyter)

set -e

PROJECT_DIR="/app"
cd "$PROJECT_DIR"

echo "🔄 PRÉPROCESSING DIRECT"
echo "======================="
echo ""

# Vérifier Docker (uniquement si exécuté localement)
if [[ ! -f /.dockerenv ]]; then
    if ! docker-compose ps | grep -q "app.*Up"; then
        echo "❌ Docker pas actif"
        docker-compose up -d
        sleep 10
    fi
fi

echo "📥 Lancement du préprocessing..."
echo ""

# Determine which command to use based on environment
if [[ -f /.dockerenv ]]; then
    # Already in container - run Python directly
    PYTHON_CMD="python"
else
    # Outside container - use docker-compose
    PYTHON_CMD="docker-compose exec -T app python"
fi

# Execute Python with single heredoc block
$PYTHON_CMD << 'PYTHON_CODE'
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
sys.path.insert(0, '/app')
os.chdir('/app')

import django
django.setup()

from analytics.models import HealthZone, ClinicalData, Province
from sklearn.preprocessing import RobustScaler
import pickle

print("=" * 70)
print("🔄 DATA PREPROCESSING PIPELINE")
print("=" * 70)
print()

# Paths
DATA_DIR = Path("/app/data")
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Load epidemiological data from raw_hdx folder
print("📊 Loading epidemiological data from raw_hdx...")
import glob

# Map file names to disease names
disease_mapping = {
    'malaria': 'Paludisme',
    'tuberculosis': 'Tuberculose',
    'hiv': 'VIH/SIDA',
    'cholera': 'Choléra'
}

epi_frames = []
data_dir = Path("/app/data/raw_hdx")

# Load health indicator CSVs
health_files = glob.glob(str(data_dir / "*_indicators_cod.csv"))
print(f"  Found {len(health_files)} health indicator files\n")

for file in health_files:
    try:
        filename = Path(file).name.lower()
        
        # Detect disease from filename
        disease_name = 'Health_Indicator'
        is_disease_specific = False
        for key, value in disease_mapping.items():
            if key in filename:
                disease_name = value
                is_disease_specific = True
                break
        
        # Load ALL rows for disease-specific files, or 5000 for generic
        if is_disease_specific:
            df = pd.read_csv(file)  # Load all rows for specific diseases
            print(f"  ✓ {filename[:50]:50} → {disease_name:15} (ALL rows: {len(df):6})")
        else:
            df = pd.read_csv(file, nrows=5000)  # Limit for generic
            print(f"  ✓ {filename[:50]:50} → {disease_name:15} (LIMITED: {len(df):6})")
        
        if len(df) > 0:
            df['disease'] = disease_name
            epi_frames.append(df)
        
    except Exception as e:
        print(f"  ✗ Error loading {filename}: {e}")

if epi_frames:
    epi_df = pd.concat(epi_frames, ignore_index=True)
    
    # Create date column
    if 'date' not in epi_df.columns:
        if 'YEAR (DISPLAY)' in epi_df.columns:
            epi_df['date'] = pd.to_datetime(epi_df['YEAR (DISPLAY)'].astype(str), format='%Y', errors='coerce')
        else:
            epi_df['date'] = pd.Timestamp.now()
    
    # Create cases column from Numeric (most reliable)
    if 'cases' not in epi_df.columns:
        if 'Numeric' in epi_df.columns:
            epi_df['cases'] = pd.to_numeric(epi_df['Numeric'], errors='coerce').fillna(0)
        elif 'Value' in epi_df.columns:
            # Clean Value column (remove spaces, brackets, etc)
            epi_df['cases'] = epi_df['Value'].str.extract('(\d+\.?\d*)', expand=False)
            epi_df['cases'] = pd.to_numeric(epi_df['cases'], errors='coerce').fillna(0)
        else:
            numeric_cols = epi_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                epi_df['cases'] = epi_df[numeric_cols[0]]
            else:
                epi_df['cases'] = 0
    
    print(f"\n✓ Loaded {len(epi_df)} total records")
    print(f"  Diseases: {epi_df['disease'].value_counts().to_dict()}")
else:
    print(f"⚠️ No health indicator files found")
    epi_df = pd.DataFrame()

# Load climate data if available
print("\n📊 Checking climate data...")
climate_dir = DATA_DIR / "climat"
climate_df = None

if climate_dir.exists():
    import glob
    files = glob.glob(str(climate_dir / '*.csv'))
    if files:
        frames = []
        for file in files[:5]:  # Limit to first 5
            try:
                df = pd.read_csv(file, nrows=1000)
                frames.append(df)
                print(f"  ✓ {Path(file).name}")
            except:
                pass
        if frames:
            climate_df = pd.concat(frames, ignore_index=True)
            print(f"✓ Climate data shape: {climate_df.shape}")
    else:
        print("  ℹ️ No climate CSV files found")
else:
    print("  ℹ️ Climate directory not found")

# Preprocess
print("\n🔧 Preprocessing...")

if len(epi_df) > 0:
    # Create missing columns if they don't exist
    if 'deaths' not in epi_df.columns:
        epi_df['deaths'] = epi_df['cases'] * 0.1
    
    # Ensure numeric values
    epi_df['cases'] = pd.to_numeric(epi_df['cases'], errors='coerce').fillna(0)
    epi_df['deaths'] = pd.to_numeric(epi_df['deaths'], errors='coerce').fillna(0)
    
    # Normalize
    scaler_cases = RobustScaler()
    epi_df['cases_normalized'] = scaler_cases.fit_transform(
        epi_df[['cases']]
    )
    
    # Save
    epi_df.to_csv(PROCESSED_DIR / 'epi_data.csv', index=False)
    print(f"✓ Saved epi_data.csv ({len(epi_df)} records)")
    
    # Save scaler
    with open(PROCESSED_DIR / 'scalers.pkl', 'wb') as f:
        pickle.dump({'cases': scaler_cases}, f)
    print("✓ Saved scalers.pkl")

if climate_df is not None:
    climate_df.to_csv(PROCESSED_DIR / 'climate_data.csv', index=False)
    print(f"✓ Saved climate_data.csv ({len(climate_df)} records)")

print("\n" + "=" * 70)
print("✅ PREPROCESSING COMPLETE")
print("=" * 70)
print(f"\n📁 Output directory: {PROCESSED_DIR}")
print(f"   Files created:")
print(f"   - epi_data.csv")
if climate_df is not None:
    print(f"   - climate_data.csv")
print(f"   - scalers.pkl")
print(f"\n👉 Next step: Run training")
print(f"   ./train_direct.sh")
PYTHON_CODE

echo ""
echo "✅ PRÉPROCESSING TERMINÉ!"
