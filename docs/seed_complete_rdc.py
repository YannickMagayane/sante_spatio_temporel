#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script pour générer 180 mois de données (2010-2025) pour TOUTES les maladies
dans TOUTES les provinces RDC pour pouvoir utiliser LSTM/ConvLSTM/CNN
"""

import os
import sys
import django
from datetime import datetime, timedelta
import random
from dateutil.relativedelta import relativedelta

sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from analytics.models import HealthZone, ClinicalData, Province
from django.db.models import Count

# Liste des 19 maladies
DISEASES = [
    'Choléra', 'Mpox', 'Tuberculose', 'VIH/SIDA', 'Rougeole',
    'Malnutrition', 'Onchocercose', 'Anémie', 'Paludisme',
    'Trypanosomiase', 'Coqueluche', 'Rubéole', 'Tétanos', 'COVID-19',
    'Ebola', 'Malnutrition Aiguë', 'Ulcère de Buruli', 'Hépatite', 'Fièvre Jaune'
]

SEASONALITY = {
    'Cholera': [10, 11, 12, 1],
    'Malaria': [11, 12, 1, 2, 3],
    'Rougeole': [2, 3, 4],
    'Ebola': [],
    'COVID-19': [1, 2],
    'Malnutrition': [6, 7, 8, 9],
    'Malnutrition Aiguë': [6, 7, 8, 9],
}

def get_seasonality_factor(disease, month):
    for key in SEASONALITY:
        if key.lower() in disease.lower():
            if month in SEASONALITY[key]:
                return random.uniform(1.5, 3.0)
    return random.uniform(0.3, 1.0)

def generate_cases(disease, zone, month):
    base_cases = {
        'Choléra': 50, 'Mpox': 10, 'Tuberculose': 150, 'VIH/SIDA': 200, 'Rougeole': 100,
        'Malnutrition': 80, 'Onchocercose': 5000, 'Anémie': 30, 'Paludisme': 300,
        'Trypanosomiase': 5, 'Coqueluche': 8, 'Rubéole': 5, 'Tétanos': 2, 'COVID-19': 20,
        'Ebola': 2, 'Malnutrition Aiguë': 50, 'Ulcère de Buruli': 1, 'Hépatite': 100,
        'Fièvre Jaune': 1,
    }
    
    base = base_cases.get(disease, 50)
    seasonal = get_seasonality_factor(disease, month.month)
    noise = random.uniform(0.8, 1.2)
    year_factor = 1.0 + (month.year - 2010) * 0.02
    result = int(base * seasonal * noise * year_factor)
    return max(0, result)

def main():
    print("\n" + "="*80)
    print("🌍 GÉNÉRATION DE 180 MOIS DE DONNÉES COMPLÈTES POUR LA RDC")
    print("="*80)
    
    zones = HealthZone.objects.all()
    total_zones = zones.count()
    print(f"\n📍 Zones trouvées: {total_zones}")
    
    if total_zones == 0:
        print("❌ ERREUR: Aucune zone de santé trouvée!")
        return
    
    provinces = Province.objects.all()
    print(f"🏛️  Provinces: {provinces.count()}")
    
    start_date = datetime(2010, 1, 1)
    dates = []
    current = start_date
    while current <= datetime(2025, 12, 1):
        dates.append(current.replace(day=1))
        current += relativedelta(months=1)
    
    print(f"📅 Mois à générer: {len(dates)} (2010-01 à 2025-12)")
    print(f"💊 Maladies: {len(DISEASES)}")
    
    print("\n🗑️  Suppression des anciennes données...")
    old_count = ClinicalData.objects.count()
    ClinicalData.objects.all().delete()
    print(f"   ✅ {old_count:,} enregistrements supprimés")
    
    print("\n📊 Génération des nouvelles données...")
    
    batch_size = 5000
    all_data = []
    counter = 0
    
    for zone in zones:
        for month in dates:
            for disease in DISEASES:
                cases = generate_cases(disease, zone, month)
                all_data.append(ClinicalData(
                    health_zone=zone,
                    disease_name=disease,
                    month=month,
                    cases_count=cases
                ))
                counter += 1
                
                if len(all_data) >= batch_size:
                    ClinicalData.objects.bulk_create(all_data)
                    print(f"   ✅ {counter:,} enregistrements créés...")
                    all_data = []
    
    if all_data:
        ClinicalData.objects.bulk_create(all_data)
    
    print("\n" + "="*80)
    print("✨ GÉNÉRATION COMPLÈTE")
    print("="*80)
    
    total_records = ClinicalData.objects.count()
    expected_records = total_zones * len(dates) * len(DISEASES)
    
    print(f"\n📊 Statistiques finales:")
    print(f"   • Total zones: {total_zones}")
    print(f"   • Total mois: {len(dates)}")
    print(f"   • Total maladies: {len(DISEASES)}")
    print(f"   • Enregistrements créés: {total_records:,}")
    print(f"   • Enregistrements attendus: {expected_records:,}")
    
    diseases_db = ClinicalData.objects.values('disease_name').annotate(
        count=Count('id')
    ).order_by('-count')[:5]
    print(f"\n📋 Top 5 maladies:")
    for d in diseases_db:
        print(f"   • {d['disease_name']:20} : {d['count']:6,}")
    
    dates_in_db = ClinicalData.objects.values_list('month', flat=True).distinct().order_by('month')
    print(f"\n📅 Couverture temporelle:")
    print(f"   • De {dates_in_db.first()} à {dates_in_db.last()}")
    print(f"   • Total mois: {dates_in_db.count()}")
    
    print("\n✅ SUCCÈS! Les données sont prêtes pour LSTM/ConvLSTM/CNN\n")

if __name__ == '__main__':
    main()
