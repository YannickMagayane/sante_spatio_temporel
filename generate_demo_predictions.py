#!/usr/bin/env python3
"""
Génère prédictions de démonstration nationales basées sur les modèles de persistance.
"""

import os
import sys
import json
import pickle
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()

from analytics.models import Prediction, HealthZone

# Créer une zone "National" si elle n'existe pas
national_zone, created = HealthZone.objects.get_or_create(
    code='NAT',
    defaults={'name': 'RDC National', 'territory': None}
)

# Charger métadonnées
with open(BASE_DIR / 'models' / 'dashboard_training_data.json') as f:
    metadata = json.load(f)

FORECAST_MONTHS = 3
models_dir = BASE_DIR / 'models'

print("="*70)
print("🔮 GÉNÉRATION PRÉDICTIONS DÉMONSTRATION (Niveau National)")
print("="*70)

new_count = 0

for disease, perf in metadata['model_performance'].items():
    # Vérifier si modèle de persistance existe
    safe_name = disease.replace(' ', '_').replace('/', '_')
    model_path = models_dir / f'model_{safe_name}_National.pkl'

    if not model_path.exists():
        print(f"⚠️  Modèle non trouvé pour {disease}")
        continue

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    last_value = model_data['last_value']
    base_prediction = last_value

    print(f"\n🦠 {disease}")
    print(f"  Dernière valeur: {last_value:,.0f}")

    # Générer prédictions pour les 3 prochains mois
    last_date = datetime.now().replace(day=1)

    for i in range(1, FORECAST_MONTHS+1):
        pred_date = last_date + relativedelta(months=i)
        # Pour persistance, légère augmentation
        prediction = base_prediction * (1 + 0.01 * i)
        propagation = round((prediction - last_value) / last_value * 100, 1) if last_value > 0 else 0.0

        # Sauvegarder
        pred, created = Prediction.objects.update_or_create(
            health_zone=national_zone,
            disease_name=disease,
            prediction_date=pred_date,
            defaults={
                'predicted_cases': float(prediction),
                'propagation_rate': propagation,
                'influence_factors': {
                    'reliability': int(perf['r2'] * 100 + 50),
                    'model_used': 'Persistence',
                    'tier': perf['tier'],
                    'r2': round(perf['r2'], 4),
                    'last_updated': datetime.now().isoformat()
                }
            }
        )
        if created:
            new_count += 1
            print(f"  ✅ {pred_date.strftime('%Y-%m')}: {prediction:,.0f} cas (prop: {propagation}%)")

print(f"\n✅ Total nouvelles prédictions: {new_count}")
print(f"   Dashboard: http://localhost:8001/")
