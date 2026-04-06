#!/usr/bin/env python3
"""
Diagnostic complet des données d'entraînement
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
sys.path.insert(0, '/app')
os.chdir('/app')

import django
django.setup()

from analytics.models import HealthZone, ClinicalData, Province

print("=" * 70)
print("📊 DIAGNOSTIC COMPLET - DONNÉES D'ENTRAÎNEMENT")
print("=" * 70)

# 1. STATISTIQUES GLOBALES
print("\n" + "=" * 70)
print("📈 1. STATISTIQUES GLOBALES")
print("=" * 70)

# Compter les données cliniques
total_clinical = ClinicalData.objects.count()
print(f"  Total enregistrements cliniques: {total_clinical:,}")

# Compter les zones
total_zones = HealthZone.objects.count()
print(f"  Total zones de santé: {total_zones:,}")

# Compter les provinces
total_provinces = Province.objects.count()
print(f"  Total provinces: {total_provinces:,}")

# 2. DONNÉES PAR MALADIE
print("\n" + "=" * 70)
print("🦠 2. DONNÉES PAR MALADIE")
print("=" * 70)

from django.db.models import Count, Avg, Min, Max, StdDev

diseases_stats = ClinicalData.objects.values('disease_name').annotate(
    count=Count('id'),
    avg_cases=Avg('cases_count'),
    min_cases=Min('cases_count'),
    max_cases=Max('cases_count'),
    std_cases=StdDev('cases_count')
).order_by('-count')

for stat in diseases_stats:
    disease = stat['disease_name']
    print(f"\n  🦠 {disease}")
    print(f"     Enregistrements: {stat['count']:,}")
    print(f"     Cas moyen: {stat['avg_cases']:.2f}")
    print(f"     Cas min/max: {stat['min_cases']:,.0f} / {stat['max_cases']:,.0f}")
    if stat['std_cases']:
        print(f"     Écart-type: {stat['std_cases']:.2f}")

    # Nombre de zones uniques
    zones_count = ClinicalData.objects.filter(disease_name=disease).values('health_zone').distinct().count()
    print(f"     Zones distinctes: {zones_count}")

# 3. ANALYSE TEMPORELLE
print("\n" + "=" * 70)
print("📅 3. ANALYSE TEMPORELLE")
print("=" * 70)

from django.db.models.functions import TruncMonth, TruncYear

for disease_stat in diseases_stats:
    disease = disease_stat['disease_name']
    print(f"\n  🦠 {disease}")

    # Première et dernière date
    date_range = ClinicalData.objects.filter(disease_name=disease).aggregate(
        min_date=Min('date'),
        max_date=Max('date')
    )
    if date_range['min_date'] and date_range['max_date']:
        print(f"     Période: {date_range['min_date'].date()} à {date_range['max_date'].date()}")

        # Nombre de mois uniques
        monthly_data = ClinicalData.objects.filter(
            disease_name=disease
        ).annotate(
            month=TruncMonth('date')
        ).values('month').distinct().count()
        print(f"     Mois distincts: {monthly_data}")

# 4. QUALITÉ DES DONNÉES
print("\n" + "=" * 70)
print("🔍 4. QUALITÉ DES DONNÉES")
print("=" * 70)

# Vérifier les valeurs manquantes
for disease_stat in diseases_stats:
    disease = disease_stat['disease_name']
    total = disease_stat['count']
    null_cases = ClinicalData.objects.filter(disease_name=disease, cases_count__isnull=True).count()
    zero_cases = ClinicalData.objects.filter(disease_name=disease, cases_count=0).count()

    print(f"\n  🦠 {disease}")
    print(f"     Valeurs nulles: {null_cases} ({null_cases/total*100:.1f}%)")
    print(f"     Valeurs zéro: {zero_cases} ({zero_cases/total*100:.1f}%)")

# 5. DONNÉES CLIMATIQUES
print("\n" + "=" * 70)
print("🌡️ 5. DONNÉES CLIMATIQUES")
print("=" * 70)

# Si fichier processed existe
climate_path = Path('/app/data/processed/climate_data.csv')
if climate_path.exists():
    climate_df = pd.read_csv(climate_path)
    print(f"  Fichier climat: {len(climate_df)} enregistrements")
    print(f"  Colonnes: {', '.join(climate_df.columns)}")
    if 'province' in climate_df.columns:
        print(f"  Provinces: {climate_df['province'].nunique()}")
else:
    print("  ❌ Fichier climate_data.csv non trouvé")

# 6. DONNÉES PRÉPROCESSÉES
print("\n" + "=" * 70)
print("📁 6. FICHIERS PRÉPROCESSÉS")
print("=" * 70)

epi_processed = Path('/app/data/processed/epi_data.csv')
if epi_processed.exists():
    df = pd.read_csv(epi_processed)
    print(f"  Épidémiologie: {len(df):,} enregistrements")
    if 'disease' in df.columns:
        print(f"  Maladies: {', '.join(df['disease'].unique())}")
else:
    print("  ❌ Fichier epi_data.csv non trouvé")

# 7. MODÈLES EXISTANTS
print("\n" + "=" * 70)
print("🤖 7. MODÈLES EXISTANTS")
print("=" * 70)

models_dir = Path('/app/models')
if models_dir.exists():
    keras_models = list(models_dir.glob('*.keras'))
    print(f"  Modèles .keras: {len(keras_models)}")

    # Grouper par maladie
    diseases_from_models = {}
    for model in keras_models:
        name = model.stem
        parts = name.split('_')
        if len(parts) >= 2:
            disease = parts[1] if parts[0] == 'model' else parts[0]
            diseases_from_models[disease] = diseases_from_models.get(disease, 0) + 1

    for disease, count in sorted(diseases_from_models.items()):
        print(f"     {disease}: {count} modèle(s)")
else:
    print("  ❌ Répertoire models non trouvé")

# 8. RECOMMANDATIONS
print("\n" + "=" * 70)
print("💡 8. RECOMMANDATIONS POUR L'ENTRAÎNEMENT")
print("=" * 70)

print("""
  ANALYSE:
  ---------
  1. Vérifiez que chaque maladie a suffisamment d'enregistrements (>20)
  2. Les données doivent être réparties sur plusieurs mois
  3. Si R² < 0, c'est que les données sont trop peu variables ou trop bruitées

  ACTIONS POSSIBLES:
  ------------------
  a) Augmenter la fenêtre temporelle (WINDOW plus grand)
  b) Filtrer les zones avec suffisamment de données
  c) Utiliser un modèle plus simple (Tier 4: persistence)
  d) Augmenter/remplir les données manquantes
  e) Ajouter des données climatiques (actuellement très limitées)
""")

print("\n" + "=" * 70)
print("✅ DIAGNOSTIC TERMINÉ")
print("=" * 70)
