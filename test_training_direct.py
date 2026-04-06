#!/usr/bin/env python
"""
🧪 Test Direct Training Script
Bypasses Jupyter notebooks - runs training directly from CLI
"""

import os
import sys
from pathlib import Path

# 🔧 FIX: Add /app to Python path BEFORE any relative imports
app_path = Path(__file__).parent.absolute()
if str(app_path) not in sys.path:
    sys.path.insert(0, str(app_path))

os.chdir(app_path)

print(f"📍 Working directory: {os.getcwd()}")
print(f"📍 App path: {app_path}")
print(f"📍 sys.path[0]: {sys.path[0]}")
print()

# ============================================================================
# STEP 1: Setup Django
# ============================================================================
print("=" * 70)
print("🔧 STEP 1: Setting up Django")
print("=" * 70)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

try:
    import django
    print(f"✓ Django {django.get_version()} imported from: {django.__file__}")
    
    django.setup()
    print("✓ Django setup completed")
except Exception as e:
    print(f"✗ Django setup failed: {e}")
    print(f"  Current workdir: {os.getcwd()}")
    print(f"  sys.path: {sys.path[:3]}")
    sys.exit(1)

# ============================================================================
# STEP 2: Import models
# ============================================================================
print("\n" + "=" * 70)
print("📦 STEP 2: Importing models")
print("=" * 70)

try:
    from analytics.models import HealthZone, ClinicalData, Prediction, Province
    print("✓ Models imported successfully")
    print(f"  - HealthZone: {HealthZone}")
    print(f"  - ClinicalData: {ClinicalData}")
except ImportError as e:
    print(f"✗ Cannot import models: {e}")
    sys.exit(1)

# ============================================================================
# STEP 3: Import ML libraries
# ============================================================================
print("\n" + "=" * 70)
print("🤖 STEP 3: Importing ML libraries")
print("=" * 70)

try:
    import numpy as np
    import pandas as pd
    import pickle
    import json
    from datetime import datetime
    import warnings
    
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    import tensorflow as tf
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    print("✓ All ML libraries imported")
    print(f"  - TensorFlow {tf.__version__}")
    print(f"  - NumPy {np.__version__}")
    print(f"  - Pandas {pd.__version__}")
except ImportError as e:
    print(f"✗ Cannot import ML libraries: {e}")
    sys.exit(1)

# ============================================================================
# STEP 4: Check data
# ============================================================================
print("\n" + "=" * 70)
print("📊 STEP 4: Checking preprocessed data")
print("=" * 70)

DATA_DIR = Path("/app/data/processed")
MODELS_DIR = Path("/app/models")

if not DATA_DIR.exists():
    print(f"⚠️  Data directory doesn't exist: {DATA_DIR}")
    print(f"   Creating: {DATA_DIR}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

if not MODELS_DIR.exists():
    print(f"⚠️  Models directory doesn't exist: {MODELS_DIR}")
    print(f"   Creating: {MODELS_DIR}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Check if preprocessed data exists
epi_data_file = DATA_DIR / 'epi_data.csv'
climate_data_file = DATA_DIR / 'climate_data.csv'

has_epi = epi_data_file.exists()
has_climate = climate_data_file.exists()

print(f"  Epidemiological data: {'✓' if has_epi else '✗'} {epi_data_file}")
print(f"  Climate data: {'✓' if has_climate else '✗'} {climate_data_file}")

if not has_epi:
    print("\n⚠️  PREPROCESSING REQUIRED!")
    print("   Please run: 01_data_preprocessing.ipynb first")
    print("   https://localhost:8888 → 01_data_preprocessing.ipynb → Run All Cells")
    sys.exit(1)

# ============================================================================
# STEP 5: Load data
# ============================================================================
print("\n" + "=" * 70)
print("📥 STEP 5: Loading data")
print("=" * 70)

try:
    epi_data = pd.read_csv(epi_data_file)
    epi_data['date'] = pd.to_datetime(epi_data['date'])
    print(f"✓ Loaded epidemiological data: {epi_data.shape}")
    print(f"  Columns: {epi_data.columns.tolist()}")
    print(f"  Date range: {epi_data['date'].min()} to {epi_data['date'].max()}")
except Exception as e:
    print(f"✗ Failed to load epi data: {e}")
    sys.exit(1)

# ============================================================================
# STEP 6: Verify everything works
# ============================================================================
print("\n" + "=" * 70)
print("✅ ALL CHECKS PASSED - READY TO TRAIN")
print("=" * 70)
print(f"""
✓ Django configured correctly
✓ Models accessible
✓ ML libraries ready
✓ Data files found: {epi_data.shape[0]} records

Next steps:
1. Run full training: python /app/train_models.py
2. Or via Jupyter: http://localhost:8888 → 02_model_training.ipynb → Run All

""")
