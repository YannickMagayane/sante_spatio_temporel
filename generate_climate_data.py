#!/usr/bin/env python3
"""
Génère données climatiques simulées ERA5-Land pour chaque zone de santé
Basé sur la latitude/longitude pour variation géographique réaliste
"""

import os
import sys
import random
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()

from analytics.models import HealthZone, ClimateData
from django.db import transaction

print("="*70)
print("🌍 GÉNÉRATION DONNÉES CLIMATIQUES SIMULÉES (ERA5-Land)")
print("="*70)

# Supprimer anciennes données
ClimateData.objects.all().delete()
print("✅ Anciennes données climatiques supprimées")

# Période: 24 derniers mois
months_back = 24
end_date = datetime.now().replace(day=1)
start_date = end_date - timedelta(days=30*months_back)

# Générer pour chaque zone
zones = HealthZone.objects.all()
print(f"\n📊 Zones à traiter: {zones.count()}")

created_count = 0

for zone in zones:
    # Déterminer baseline climatique selon latitude (simplifié)
    lat = zone.latitude if zone.latitude else 0.0
    lon = zone.longitude if zone.longitude else 25.0

    # RDC: latitude ~ -4 à 5, longitude ~ 12 à 31
    # Température moyenne: plus chaud au sud (proche équateur)
    base_temp = 26.0 - (lat / 10) if lat else 25.0  # °C

    # Précipitations: plus élevées dans l'est (haute altitude)
    base_precip = 150.0 + (lon - 15) * 10  # mm/mois

    # Variation saisonnière (RDC: saison des pluies varies)
    seasonality_temp = 2.0  # °C amplitude
    seasonality_precip = 100.0  # mm amplitude

    current_date = start_date
    while current_date <= end_date:
        # Mois de l'année (1-12)
        month = current_date.month

        # Saison en RDC:
        # - Saison des pluies: Oct-May (selon région)
        # - Saison sèche: Jun-Sep
        is_rainy = month in [10, 11, 12, 1, 2, 3, 4, 5]

        # Calculer température avec saisonnalité
        temp_variation = seasonality_temp * np.sin(2 * np.pi * (month - 1) / 12)
        temperature = base_temp + temp_variation + random.uniform(-0.5, 0.5)

        # Calculer précipitations avec saisonnalité
        precip_base = base_precip
        if is_rainy:
            precip = precip_base + seasonality_precip * 0.8 + random.uniform(-20, 20)
        else:
            precip = precip_base * 0.3 + random.uniform(-10, 10)

        # Humidité liée aux précipitations
        humidity = 65 + (precip / 5) + random.uniform(-5, 5)
        humidity = max(30, min(95, humidity))

        # Créer enregistrement
        ClimateData.objects.create(
            health_zone=zone,
            month=current_date.replace(day=1),
            temperature=round(temperature, 2),
            precipitation=round(precip, 2),
            humidity=round(humidity, 1)
        )
        created_count += 1

        # Mois suivant
        current_date += timedelta(days=32)
        current_date = current_date.replace(day=1)

    if zone.pk % 50 == 0:
        print(f"  Traité: {zone.name[:30]}")

print(f"\n✅ Données climatiques générées: {created_count} enregistrements")
print(f"   Période: {start_date.strftime('%Y-%m')} à {end_date.strftime('%Y-%m')}")
print(f"   Zones couvertes: {zones.count()}")

# Vérifier
print(f"\n📈 Vérification:")
print(f"  ClimateData total: {ClimateData.objects.count()}")
print(f"  Zones avec climat: {ClimateData.objects.values('health_zone').distinct().count()}")

print("\n" + "="*70)
print("✅ TERMINÉ!")
print("="*70)
