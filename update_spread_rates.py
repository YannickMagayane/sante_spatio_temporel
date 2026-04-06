#!/usr/bin/env python
"""
📈 HealthDRC - Mise à jour des taux de propagation
===================================================

Calcule les taux de propagation (croissance) pour chaque prédiction
et met à jour les facteurs d'influence dans la base.

Taux de propagation = ((prédiction - dernier_historique) / dernier_historique) * 100
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()

from django.db import transaction
from analytics.models import Prediction, ClinicalData

# Config
PROPAGATION_WINDOW = 3  # mois pour calculer tendance
ALERT_THRESHOLD_HIGH = 50  # +50% = alerte forte
ALERT_THRESHOLD_MEDIUM = 20  # +20% = alerte modérée

def calculate_propagation_rate(zone_id, disease, pred_date):
    """Calcule taux de propagation pour une prédiction donnée"""

    # Get prediction
    pred = Prediction.objects.get(
        health_zone_id=zone_id,
        disease_name=disease,
        prediction_date=pred_date
    )

    # Get historical data (last 3 months before prediction month)
    hist_end = pred_date - relativedelta(months=1)
    hist_start = hist_end - relativedelta(months=PROPAGATION_WINDOW)

    hist_data = ClinicalData.objects.filter(
        health_zone_id=zone_id,
        disease_name=disease,
        month__gte=hist_start,
        month__lte=hist_end
    ).order_by('month')

    if hist_data.count() < 2:
        propagation_rate = 0.0
    else:
        # Calculate average of historical window
        hist_cases = [h.cases_count for h in hist_data]
        avg_historical = np.mean(hist_cases)

        # Compare to prediction
        if avg_historical > 0:
            propagation_rate = ((pred.predicted_cases - avg_historical) / avg_historical) * 100
        else:
            propagation_rate = 100.0 if pred.predicted_cases > 0 else 0.0

    # Determine alert level
    if propagation_rate >= ALERT_THRESHOLD_HIGH:
        alert_level = "ROUGE"
        attack_angle = "ALERTE ROUGE"
    elif propagation_rate >= ALERT_THRESHOLD_MEDIUM:
        alert_level = "ORANGE"
        attack_angle = "VIGILANCE"
    else:
        alert_level = "STABLE"
        attack_angle = "STABLE"

    # Update influence_factors
    factors = pred.influence_factors or {}
    factors.update({
        'weekly_spread_rate': round(propagation_rate / 4, 1),  # Approx weekly
        'propagation_rate': round(propagation_rate, 1),
        'alert_level': alert_level,
        'attack_angle': attack_angle,
        'last_calculated': datetime.now().isoformat()
    })

    # Also calculate trend acceleration (second derivative)
    if hist_data.count() >= 3:
        last_3 = [h.cases_count for h in hist_data[-3:]]
        trend = np.polyfit(range(3), last_3, 1)[0]  # Linear trend
        acceleration = np.polyfit(range(3), last_3, 2)[0] if len(last_3) == 3 else 0
        factors['trend_direction'] = 'increasing' if trend > 0 else 'decreasing'
        factors['acceleration'] = round(float(acceleration), 2)

    # Save
    pred.influence_factors = factors
    pred.save()

    return propagation_rate, alert_level

def main():
    print("="*70)
    print("📈 MISE À JOUR TAUX DE PROPAGATION")
    print("="*70)

    # Get all recent predictions (last 3 months)
    today = datetime.now().date()
    three_months_ago = today - relativedelta(months=3)

    predictions = Prediction.objects.filter(
        prediction_date__gte=three_months_ago
    )

    print(f"  • Prédictions à traiter: {predictions.count()}")

    updated = 0
    errors = 0

    for pred in predictions.iterator():
        try:
            rate, alert = calculate_propagation_rate(
                pred.health_zone_id,
                pred.disease_name,
                pred.prediction_date
            )
            print(f"  ✓ {pred.health_zone.name}/{pred.disease_name}: {rate:.1f}% ({alert})")
            updated += 1
        except Exception as e:
            print(f"  ✗ {pred.health_zone.name}/{pred.disease_name}: {e}")
            errors += 1

    print(f"\n✅ {updated} prédictions mises à jour")
    if errors > 0:
        print(f"⚠️  {errors} erreurs")

    print("="*70)

if __name__ == '__main__':
    main()
