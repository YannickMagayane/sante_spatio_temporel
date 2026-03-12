#!/usr/bin/env python3
"""Diagnostic complet des données cliniques avant entraînement"""

import os
import sys
import django

# Django setup
app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, app_dir)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from analytics.models import ClinicalData, HealthZone, Territory
from django.db.models import Count, Sum

print("\n" + "="*80)
print("🔍 DIAGNOSTIC DONNÉES CLINIQUES")
print("="*80 + "\n")

# 1. Couverture globale par maladie
print("📊 COUVERTURE GLOBALE PAR MALADIE")
print("-" * 80)
for disease in ['Paludisme', 'Choléra', 'Rougeole', 'Tuberculose', 'VIH/SIDA']:
    count = ClinicalData.objects.filter(disease_name=disease).count()
    total_cases = ClinicalData.objects.filter(disease_name=disease).aggregate(Sum('cases_count'))['cases_count__sum'] or 0
    zones = ClinicalData.objects.filter(disease_name=disease).values('health_zone').distinct().count()
    months = ClinicalData.objects.filter(disease_name=disease).values('month').distinct().count()
    print(f"  {disease:15} → {count:8,} records | {total_cases:15,} cas | {zones:3} zones | {months:2} mois")

# 2. Couverture par province et maladie
print("\n\n📍 ZONES DE SANTÉ MANQUANTES PAR PROVINCE")
print("-" * 80)

provinces = Territory.objects.filter(province__isnull=False).values('name').distinct().order_by('name')

for prov in provinces:
    prov_name = prov['name']
    total_zones = HealthZone.objects.filter(territory__name=prov_name).count()
    
    disease_coverage = {}
    for disease in ['Paludisme', 'Choléra', 'Rougeole', 'Tuberculose', 'VIH/SIDA']:
        zones_with_data = ClinicalData.objects.filter(
            disease_name=disease,
            health_zone__territory__name=prov_name
        ).values('health_zone').distinct().count()
        disease_coverage[disease] = zones_with_data
    
    # Zones manquantes
    min_coverage = min(disease_coverage.values()) if disease_coverage else 0
    missing = total_zones - min_coverage if min_coverage > 0 else total_zones
    
    if missing > 0:
        print(f"  {prov_name:20} | Total zones: {total_zones:3} | Min couverture: {min_coverage:3} | MANQUANTES: {missing:3}")

# 3. Distribution par province (top 5)
print("\n\n📊 DISTRIBUTION DATA PAR PROVINCE - TOP 5 (Choléra)")
print("-" * 80)
from django.db.models import Sum as DjangoSum

provinces_dist = ClinicalData.objects.filter(
    disease_name='Choléra'
).values('health_zone__territory__name').annotate(
    total=Count('id'),
    cases=DjangoSum('cases_count')
).order_by('-cases')[:5]

for dist in provinces_dist:
    pname = dist['health_zone__territory__name']
    total_rec = dist['total']
    total_cases = dist['cases'] or 0
    print(f"  {pname:20} | {total_rec:5,} records | {total_cases:10,} cas")

# 4. Detect problèmes d'entraînement
print("\n\n⚠️  PROBLÈMES DIAGNOSTIQUÉS")
print("-" * 80)

# Vérifier variance et couverture pour VIH/SIDA
vih_data = ClinicalData.objects.filter(disease_name='VIH/SIDA').values_list('cases_count', flat=True)
vih_list = list(vih_data)

if vih_list:
    mean_vih = sum(vih_list) / len(vih_list)
    min_vih = min(vih_list)
    max_vih = max(vih_list)
    
    print(f"  ❌ VIH/SIDA: µ={mean_vih:.2f}, min={min_vih:.0f}, max={max_vih:.0f}")
    print(f"     → Ratio max/min = {max_vih/min_vih if min_vih > 0 else 'INF':.0f}x")
    if max_vih / (min_vih if min_vih > 0 else 1) > 100:
        print(f"     → PROBLÈME: Variance énorme → CNN-LSTM va DIVERGER")

# Vérifier données par province
print("\n" + "="*80)
print("✅ FIN DIAGNOSTIC")
print("="*80 + "\n")
