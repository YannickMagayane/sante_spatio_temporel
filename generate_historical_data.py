#!/usr/bin/env python3
"""
Génère des données historiques simulées (ClinicalData)
à partir des prédictions existantes pour每 maladie et zone.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import random
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()

from analytics.models import ClinicalData, Prediction, HealthZone
from django.db import transaction

print("="*70)
print("📈 GÉNÉRATION DONNÉES HISTORIQUES (ClinicalData)")
print("="*70)

# Supprimer anciennes données historiques (garder les prédictions)
ClinicalData.objects.all().delete()
print("✅ Anciennes données historiques supprimées")

# Récupérer toutes les zones avec des prédictions
zones_with_preds = HealthZone.objects.filter(prediction__isnull=False).distinct()
print(f"\n📍 Zones avec prédictions: {zones_with_preds.count()}")

# Période historique : 24 mois avant aujourd'hui
end_date = datetime.now().replace(day=1)
start_date = end_date - timedelta(days=24*30)

total_created = 0

for zone in zones_with_preds:
    # Récupérer les prédictions futures pour cette zone (pour chaque maladie)
    preds = Prediction.objects.filter(health_zone=zone).order_by('disease_name', 'prediction_date')

    # Regrouper par maladie
    preds_by_disease = {}
    for p in preds:
        if p.disease_name not in preds_by_disease:
            preds_by_disease[p.disease_name] = []
        preds_by_disease[p.disease_name].append(p)

    for disease, disease_preds in preds_by_disease.items():
        # Prendre la première prédiction comme valeur de référence
        ref_value = disease_preds[0].predicted_cases if disease_preds else 100

        # Générer des données historiques avec tendance (growth/decline)
        current_date = start_date
        month_count = 0
        while current_date < end_date:
            # Simuler une tendance (légère croissance ou décroissance aléatoire)
            trend_factor = 1 + random.uniform(-0.02, 0.02)  # ±2% par mois
            base_historical = ref_value * (trend_factor ** month_count)

            # Ajouter du bruit aléatoire (±15%)
            noise = random.uniform(0.85, 1.15)
            cases = int(base_historical * noise)

            # Décès : ~2-5% des cas
            deaths = int(cases * random.uniform(0.02, 0.05))

            # Créer l'enregistrement
            ClinicalData.objects.create(
                health_zone=zone,
                disease_name=disease,
                cases_count=cases,
                deaths_count=deaths,
                month=current_date.replace(day=1)
            )
            total_created += 1

            month_count += 1
            # Mois suivant
            current_date += timedelta(days=32)
            current_date = current_date.replace(day=1)

    if zone.pk % 50 == 0:
        print(f"  Traité: {zone.name[:30]}")

print(f"\n✅ Données historiques créées: {total_created} enregistrements")
print(f"   Zones couvertes: {zones_with_preds.count()}")
print(f"   Maladies: Paludisme, Tuberculose, VIH/SIDA")

# Vérification
by_disease = {}
for disease in ['Paludisme', 'Tuberculose', 'VIH/SIDA']:
    count = ClinicalData.objects.filter(disease_name=disease).count()
    by_disease[disease] = count
    print(f"    {disease}: {count} enregistrements")

print("\n" + "="*70)
print("✅ TERMINÉ")
print("="*70)
