#!/usr/bin/env python3
"""
Préparation des données pour l'entraînement optimisé
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
sys.path.insert(0, '/app')
os.chdir('/app')

import django
django.setup()

from analytics.models import HealthZone, Province

# Charger données
df = pd.read_csv('/app/data/processed/epi_data.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Garder seulement les maladies épidémiques
EPIDEMIC_DISEASES = ['Paludisme', 'Choléra', 'Tuberculose', 'VIH/SIDA']
df = df[df['disease'].isin(EPIDEMIC_DISEASES)]

# Identifier la colonne zone
zone_col = 'DIMENSION (NAME)'
if zone_col not in df.columns:
    print("❌ Colonne zone non trouvée!")
    sys.exit(1)

df['zone_name'] = df[zone_col].fillna('Unknown')

# Pour chaque maladie, analyser les zones
print("=" * 70)
print("📊 ANALYSE ZONES PAR MALADIE")
print("=" * 70)

trained_zones = {}

for disease in EPIDEMIC_DISEASES:
    disease_df = df[df['disease'] == disease].copy()
    print(f"\n🦠 {disease}: {len(disease_df)} enregistrements")

    if len(disease_df) == 0:
        continue

    # Grouper par zone
    zone_counts = disease_df['zone_name'].value_counts()
    print(f"  Zones distinctes: {len(zone_counts)}")

    # Zones avec au moins 12 mois de données
    sufficient_zones = zone_counts[zone_counts >= 12]
    print(f"  Zones avec ≥12 mois: {len(sufficient_zones)}")

    # Calculer moyenne par zone
    if len(sufficient_zones) > 0:
        print(f"\n  Top 10 zones:")
        for zone, count in sufficient_zones.head(10).items():
            zone_data = disease_df[disease_df['zone_name'] == zone]
            months = zone_data['month_year' if 'month_year' in disease_df.columns else 'date'].nunique()
            print(f"    {zone[:40]:40} {count:4} enregistrements, {months:2} mois")

        # Sélectionner zones avec suffisamment de mois distincts
        selected_zones = []
        for zone in sufficient_zones.index:
            zone_data = disease_df[disease_df['zone_name'] == zone]
            n_months = zone_data['month_year' if 'month_year' in disease_df.columns else 'date'].nunique()
            if n_months >= 12:
                selected_zones.append(zone)

        trained_zones[disease] = selected_zones
        print(f"\n  ✅ Sélectionnées: {len(selected_zones)} zones")
    else:
        print(f"  ⚠️  Pas assez de données")

# Préparer données d'entraînement
print("\n" + "=" * 70)
print("🔧 PRÉPARATION DES DONNÉES D'ENTRAÎNEMENT")
print("=" * 70)

train_data = []

for disease, zones in trained_zones.items():
    if not zones:
        continue

    disease_df = df[df['disease'] == disease]

    for zone in zones:
        zone_data = disease_df[disease_df['zone_name'] == zone][['date', 'cases']].copy()
        zone_data = zone_data.sort_values('date')
        zone_data['cases'] = pd.to_numeric(zone_data['cases'], errors='coerce').fillna(0)

        # Remplir les trous temporels
        date_range = pd.date_range(zone_data['date'].min(), zone_data['date'].max(), freq='MS')
        full_df = pd.DataFrame({'date': date_range})
        full_df = full_df.merge(zone_data[['date', 'cases']], on='date', how='left')
        full_df['cases'] = full_df['cases'].ffill().fillna(0)

        train_data.append({
            'disease': disease,
            'zone': zone,
            'data': full_df['cases'].values,
            'start_date': full_df['date'].min(),
            'end_date': full_df['date'].max(),
            'n_months': len(full_df)
        })

print(f"✅ Données préparées pour {len(train_data)} combinaisons maladie/zone")

# Sauvegarder
output_file = Path('/app/data/processed/train_ready_data.json')
import json
with open(output_file, 'w') as f:
    # Convertir les numpy arrays en listes
    serializable = []
    for item in train_data:
        serializable.append({
            'disease': item['disease'],
            'zone': item['zone'],
            'data': [float(x) for x in item['data']],
            'start_date': item['start_date'].isoformat(),
            'end_date': item['end_date'].isoformat(),
            'n_months': item['n_months']
        })
    json.dump(serializable, f, indent=2)

print(f"✅ Données sauvegardées: {output_file}")

# Résumé
print("\n" + "=" * 70)
print("📋 RÉSUMÉ")
print("=" * 70)
for item in train_data[:10]:
    print(f"  {item['disease'][:15]:15} {item['zone'][:30]:30} {item['n_months']:3} mois")
if len(train_data) > 10:
    print(f"  ... et {len(train_data)-10} autres")
