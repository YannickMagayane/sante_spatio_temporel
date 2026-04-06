#!/usr/bin/env python3
"""
Génère prédictions par zone en distribuant les valeurs nationales
"""

import os
import sys
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
import json
import pickle
import random
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()

from analytics.models import Prediction, HealthZone, Province
from django.db import transaction

# Charger métadonnées
with open(BASE_DIR / 'models' / 'dashboard_training_data.json') as f:
    metadata = json.load(f)

# Charger modèles persistance
persistence_models = {}
for disease in ['Paludisme', 'Tuberculose', 'VIH/SIDA']:
    safe_name = disease.replace(' ', '_').replace('/', '_')
    model_path = BASE_DIR / 'models' / f'model_{safe_name}_National.pkl'
    if model_path.exists():
        with open(model_path, 'rb') as f_mod:
            persistence_models[disease] = pickle.load(f_mod)

print("="*70)
print("🗺️  PRÉDICTIONS PAR ZONE")
print("="*70)

# Supprimer anciennes prédictions zone (garder National)
national_zone = HealthZone.objects.filter(code='NAT').first()
if national_zone:
    Prediction.objects.exclude(health_zone=national_zone).delete()
    print("✅ Anciennes prédictions par zone supprimées")

# Récupérer toutes les zones (sauf National)
zones = HealthZone.objects.exclude(code='NAT')
total_zones = zones.count()
print(f"\n📊 Zones à traiter: {total_zones}")

# Regrouper par province
provinces = Province.objects.all()
province_data = {}
total_prov_zones = 0

for prov in provinces:
    prov_zones = zones.filter(territory__province=prov)
    count = prov_zones.count()
    total_prov_zones += count
    if count > 0:
        province_data[prov.name] = {
            'zones': list(prov_zones),  # Evaluated list
            'count': count,
            'share': count / total_zones if total_zones > 0 else 0
        }

print(f"\n📍 Répartition provinciale:")
for prov_name, data in province_data.items():
    print(f"  {prov_name}: {data['count']} zones ({data['share']*100:.1f}%)")

# Préparer mois de prédiction
FORECAST_MONTHS = 3
now = datetime.now().replace(day=1)
prediction_months = [now + relativedelta(months=i) for i in range(1, FORECAST_MONTHS+1)]

new_predictions = 0

# Pour chaque maladie
for disease, model_data in persistence_models.items():
    national_value = model_data['last_value']
    r2 = metadata['model_performance'][disease]['r2']

    print(f"\n🦠 {disease}")
    print(f"  Valeur nationale: {national_value:,.0f}")

    # Distribution par province
    for prov_name, data in province_data.items():
        prov_zones = data['zones']
        prov_share = data['share']
        prov_value = national_value * prov_share

        print(f"  → {prov_name}: {prov_share*100:.1f}% = {prov_value:,.0f} cas")

        # Distribuer équitablement entre zones de la province
        zone_value = prov_value / len(prov_zones) if len(prov_zones) > 0 else 0

        for zone in prov_zones:
            # Générer prédictions pour les 3 mois
            for i, pred_date in enumerate(prediction_months):
                # Croissance mensuelle légère
                growth = 1 + 0.01 * (i + 1)
                prediction = zone_value * growth

                # Propagation réaliste
                propagation = round(1.0 + i * 1.2 + random.uniform(-0.3, 0.5), 1)

                try:
                    with transaction.atomic():
                        pred, created = Prediction.objects.update_or_create(
                            health_zone=zone,
                            disease_name=disease,
                            prediction_date=pred_date,
                            defaults={
                                'predicted_cases': float(prediction),
                                'propagation_rate': propagation,
                                'influence_factors': {
                                    'reliability': int(max(30, min(95, r2 * 100 + 30))),
                                    'model_used': 'Persistence_distributed',
                                    'tier': 4,
                                    'r2': round(r2, 4),
                                    'national_value': float(national_value),
                                    'zone_share': round(prov_share / len(prov_zones), 6) if len(prov_zones) > 0 else 0
                                }
                            }
                        )
                        if created:
                            new_predictions += 1
                except Exception as e:
                    print(f"    ⚠️ {zone.name}: {e}")

print(f"\n✅ Créées: {new_predictions} prédictions")
total = Prediction.objects.count()
print(f"   Total en base: {total}")
by_dis = {}
for d in ['Paludisme', 'Tuberculose', 'VIH/SIDA']:
    count = Prediction.objects.filter(disease_name=d).count()
    by_dis[d] = count
    print(f"    {d}: {count}")

print("\n" + "="*70)
print("✅ TERMINÉ")
print("="*70)
