# 📓 Health DRC - Jupyter Notebooks Pipeline

## Overview

All Python scripts have been converted to **interactive Jupyter Notebooks** for better control, monitoring, and reproducibility.

## Available Notebooks

### 1. `01_data_preprocessing.ipynb` 🔄
**Purpose:** Prepare raw epidemiological and climate data for model training

**Features:**
- Load CSV files and database data
- Handle missing values (forward fill, linear interpolation)
- Normalize features using RobustScaler
- Generate data quality reports
- Save preprocessed data for training

**Expected Duration:** 5-15 minutes
**Output:** 
- `data/processed/epi_data.csv`
- `data/processed/climate_data.csv`
- `data/processed/scalers.pkl`

---

### 2. `02_model_training.ipynb` 🧠
**Purpose:** Train multi-tier deep learning models

**Features:**
- Create LSTM sequences from time series
- Automatic tier selection based on data availability:
  - **TIER 1:** ConvLSTM2D (≥10 months) — Spatio-temporal
  - **TIER 2:** CNN-LSTM (≥5 months) — Local patterns
  - **TIER 3:** Bi-LSTM (≥3 months) — Temporal sequences
  - **TIER 4:** Persistence fallback
- Train separate models for each disease
- Early stopping & learning rate reduction
- Detailed performance metrics (MAE, RMSE, R²)
- Save models and metadata

**Target Diseases:**
- Paludisme (Malaria)
- Choléra (Cholera)
- VIH/SIDA (HIV/AIDS)
- Tuberculose (Tuberculosis)

**Expected Duration:** 30-60 minutes
**Output:**
- `models/model_*.keras` (trained models)
- `models/scaler_*.pkl` (normalization scalers)
- `models/training_metadata.json` (performance metrics)

---

### 3. `03_generate_predictions.ipynb` 🔮
**Purpose:** Generate monthly forecasts for all health zones

**Features:**
- Load trained models and scalers
- Process latest epidemiological data
- Generate predictions for next month
- Store results in database
- Ranking of highest-risk health zones
- Performance summary by disease

**Expected Duration:** 10-20 minutes
**Output:**
- Predictions stored in Django ORM (`Prediction` model)
- Dashboard automatically updates

---

## How to Use

### Option 1: JupyterLab Interface (Recommended)
```bash
# Access JupyterLab
http://localhost:8888

# Navigate to /app/notebooks/
# Open and run notebooks interactively
```

### Option 2: Dashboard Control Panel
```bash
# Access notebook control center
http://localhost:8000/analysis/notebooks/

# Click "Execute Notebook" buttons
# Monitor execution in real-time
```

### Option 3: Command Line (Docker)
```bash
# Execute individual notebook
docker-compose exec jupyter jupyter nbconvert \
  --to notebook --execute \
  --output=/tmp/output.ipynb \
  /app/notebooks/01_data_preprocessing.ipynb

# View results
docker-compose exec app python manage.py shell
>>> from analytics.models import Prediction
>>> Prediction.objects.count()
```

---

## Typical Workflow

### Full Pipeline Execution
1. **Data Preprocessing** (5-15 min)
   ```
   01_data_preprocessing.ipynb
   ```
   - Load raw data
   - Clean & normalize
   - Generate processed CSVs

2. **Model Training** (30-60 min)
   ```
   02_model_training.ipynb
   ```
   - Build deep learning models
   - Train on pre-processed data
   - Save trained models

3. **Generate Predictions** (10-20 min)
   ```
   03_generate_predictions.ipynb
   ```
   - Load trained models
   - Create forecasts
   - Update dashboard data

**Total Time:** ~1-2 hours for full pipeline

---

## Docker Integration

### JupyterLab Service

The `docker-compose.yml` includes a `jupyter` service:

```yaml
jupyter:
  build: .
  command: jupyter lab --ip=0.0.0.0 --port=8888
  volumes:
    - ./notebooks:/app/notebooks
  ports:
    - "8888:8888"
```

**Access:** http://localhost:8888

### Automatic Data Volume Sharing

- All notebooks have access to:
  - `/app/data/` (raw data)
  - `/app/models/` (trained models)
  - `/app/notebooks/` (other notebooks)
  - Django ORM database connection

---

## API Endpoints for Dashboard

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/analysis/api/notebooks/` | POST | List available notebooks |
| `/analysis/api/notebooks/execute/` | POST | Execute a notebook |
| `/analysis/api/notebooks/status/<id>/` | GET | Check execution status |

### Example: Execute Notebook from Python
```python
import requests

response = requests.post(
    'http://localhost:8000/analysis/api/notebooks/execute/',
    json={'notebook': '01_data_preprocessing'},
    headers={'X-CSRFToken': csrf_token}
)

if response.json()['success']:
    task_id = response.json()['task_id']
    # Check status
    status = requests.get(f'/analysis/api/notebooks/status/{task_id}/')
    print(status.json()['status'])  # 'running', 'completed', or 'failed'
```

---

## Tips & Tricks

### ✅ What's New
- Notebooks are **interactive** — examine data at each step
- **No Python scripts** — everything is in notebooks
- **Integrated with Django** — direct database access
- **Progress tracking** — monitor execution in dashboard
- **Error recovery** — see detailed logs and fix issues

### ⚠️ Common Issues

**Issue:** `Module not found` in notebook
```python
# Solution: Run this cell first
import sys
sys.path.append('/app')
```

**Issue:** Database connection refused
```python
# Solution: Ensure Django is setup
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()
```

**Issue:** "Insufficient data" for tier selection
- Run more months of data ingestion first
- Or use `--force-tier=3` to override

### 📊 Viewing Results

Once predictions are generated:
```bash
# Django shell
python manage.py shell

from analytics.models import Prediction
from datetime import datetime

# Get predictions for specific date
today = datetime.now().date()
preds = Prediction.objects.filter(prediction_date__month=today.month)
preds.count()  # See how many
```

---

## Performance Monitoring

Monitor notebook execution via the dashboard:
- **✅ Completed** — Notebook finished successfully
- **⏳ Running** — Notebook currently executing
- **❌ Failed** — Check logs for errors

Click on task entries to see:
- Execution time
- Error messages
- Output/results

---

## Troubleshooting

### Notebooks won't start
```bash
# Check JupyterLab logs
docker-compose logs jupyter

# Restart service
docker-compose restart jupyter
```

### Models training too slowly
- Check GPU availability
- Reduce `WINDOW` size in notebook
- Decrease `epochs` parameter

### Out of memory errors
```python
# In notebook — manually free memory
import gc
gc.collect()  # Between training runs
```

---

## API Reference

See inline notebook documentation for parameter details:
- Each cell has markdown descriptions
- Code is heavily commented
- Example outputs shown after each section

---

## Location: `/app/notebooks/`

All notebooks are stored in the `notebooks` directory and synchronized with Docker volumes.

**Happy Forecasting! 🎯**
