import glob
import json
import os
from datetime import datetime
import csv

import numpy as np
try:
    from scripts.geo_reference import PROVINCES, KNOWN_LOCATIONS
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from scripts.geo_reference import PROVINCES, KNOWN_LOCATIONS

OUTPUT_DIR = "frontend/api"
DISEASES = ["Paludisme", "Choléra", "Rougeole", "Mpox", "Ebola"]
MONTHS = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun", "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]

def load_model_metrics():
    models = []
    for model_type in ["lstm", "cnn", "convlstm"]:
        metrics_path = f"reports/{model_type}_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                data = json.load(f)
                models.append({
                    "name": model_type.upper(),
                    "type": model_type,
                    "rmse": data.get("rmse", "—"),
                    "mae": data.get("mae", "—"),
                    "auc": data.get("auc", "—"),
                    "f1": data.get("f1_score", "—"),
                    "status": "trained",
                })
        else:
            models.append({"name": model_type.upper(), "type": model_type, "status": "pending"})
    return models

def generate_predictions():
    """Returns real predictions only, 0 if data missing."""
    predictions = {}
    for prov in PROVINCES:
        predictions[prov.lower()] = {}
        for disease in DISEASES:
            # Try to load real predictions from analysis
            pred_file = f"data/processed/{prov.lower()}_{disease.lower()}_predictions.json"
            if os.path.exists(pred_file):
                with open(pred_file, "r") as f:
                    data = json.load(f)
                    predictions[prov.lower()][disease] = data.get("predicted_cases", 0)
            else:
                predictions[prov.lower()][disease] = 0
    return predictions

def generate_trends():
    """Load real trends from processed data."""
    # Placeholder structure for real data
    datasets = {}
    for disease in DISEASES:
        path = f"data/processed/trends_{disease.lower()}.json"
        if os.path.exists(path):
            with open(path, "r") as f:
                datasets[disease] = json.load(f)
        else:
            datasets[disease] = {"data": [0]*12, "color": "#666666"}
    return {"months": MONTHS, "datasets": datasets}

def generate_alerts():
    """Load real alerts with their GPS coordinates."""
    alerts = []
    scraped_files = sorted(glob.glob("data/scraped/*.csv"), reverse=True)
    for filepath in scraped_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    alerts.append({
                        "text": row.get("text", row.get("title", "")),
                        "platform": row.get("platform", "unknown"),
                        "city": row.get("location_name", row.get("city", "")),
                        "date": row.get("date", "")[:10],
                        "severity": _classify_severity(row.get("text", "")),
                        "lat": float(row.get("lat", 0)),
                        "lon": float(row.get("lon", 0)),
                        "province": row.get("province", "")
                    })
        except Exception: continue
    return alerts[:100] # Limit for UI performance

def _classify_severity(text):
    text_lower = text.lower()
    if any(k in text_lower for k in ["épidémie", "ebola", "mort", "décès"]): return "high"
    if any(k in text_lower for k in ["alerte", "augmentation", "suspect"]): return "medium"
    return "low"

def generate_stats(predictions):
    stats = []
    for disease in DISEASES:
        total = sum(p.get(disease, 0) for p in predictions.values())
        stats.append({"name": disease, "cases": total, "trend": 0})
    return stats

def generate_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    preds = generate_predictions()
    _write_json("predictions.json", preds)
    _write_json("stats.json", generate_stats(preds))
    _write_json("trends.json", generate_trends())
    _write_json("alerts.json", generate_alerts())
    _write_json("models.json", load_model_metrics())
    _write_json("cities.json", KNOWN_LOCATIONS)
    _write_json("provinces.json", PROVINCES)
    print("\n✅ Dashboard API updated with REAL data covering all RDC.")

def _write_json(filename, data):
    with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    generate_all()
