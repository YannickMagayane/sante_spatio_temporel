#!/usr/bin/env python3
"""
Script pour diagnostiquer et corriger les données climatiques aberrantes
 dans la base de données HealthDRC.
"""

import os
import sys
import django

# Configurer Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
sys.path.append('/home/yannick-magayane/Bureau/projets/data_analyse/health')

try:
    django.setup()
except Exception as e:
    print(f"Erreur configuration Django: {e}")
    sys.exit(1)

from analytics.models import ClimateData, HealthZone, Province
from django.db.models import Avg, Min, Max
import numpy as np

def diagnose_climate_data():
    """Diagnostic des données climatiques"""
    print("=" * 80)
    print("DIAGNOSTIC DES DONNÉES CLIMATIQUES - HealthDRC")
    print("=" * 80)

    # 1. Compter les enregistrements
    total_climate = ClimateData.objects.count()
    print(f"\n📊 Total enregistrements climatiques: {total_climate:,}")

    if total_climate == 0:
        print("❌ AUCUNE DONNÉE CLIMATIQUE DISPONIBLE!")
        print("   → Exécutez les scripts d'ingestion climatique.")
        return

    # 2. Plage de dates
    date_range = ClimateData.objects.aggregate(
        min_date=Min('month'),
        max_date=Max('month')
    )
    print(f"\n📅 Période couverte: {date_range['min_date']} à {date_range['max_date']}")

    # 3. Zones de santé avec données climatiques
    zones_with_climate = HealthZone.objects.filter(climatedata__isnull=False).distinct().count()
    total_zones = HealthZone.objects.count()
    print(f"\n📍 Zones avec climat: {zones_with_climate:,} / {total_zones:,}")

    # 4. Statistiques sur les temperatures (en Kelvin)
    temp_stats = ClimateData.objects.aggregate(
        min_temp=Min('temperature'),
        max_temp=Max('temperature'),
        avg_temp=Avg('temperature')
    )
    print(f"\n🌡️ TEMPÉRATURES (Kelvin):")
    print(f"   Min: {temp_stats['min_temp']:.2f} K")
    print(f"   Max: {temp_stats['max_temp']:.2f} K")
    print(f"   Moy: {temp_stats['avg_temp']:.2f} K")

    # Convertir en Celsius pour référence
    if temp_stats['avg_temp']:
        avg_c = temp_stats['avg_temp'] - 273.15
        min_c = temp_stats['min_temp'] - 273.15
        max_c = temp_stats['max_temp'] - 273.15
        print(f"\n   Équivalent Celsius:")
        print(f"   Min: {min_c:.1f}°C")
        print(f"   Max: {max_c:.1f}°C")
        print(f"   Moy: {avg_c:.1f}°C")

    # 5. Détecter les valeurs aberrantes (en dessous de 200K = -73°C, improbable)
    aberrant_low = ClimateData.objects.filter(temperature__lt=200).count()
    aberrant_high = ClimateData.objects.filter(temperature__gt=320).count()  # > 47°C

    if aberrant_low > 0:
        print(f"\n⚠️  VALEURS TEMPÉRATURE TROP BASSES (<200K): {aberrant_low:,} enregistrements")
        samples = ClimateData.objects.filter(temperature__lt=200).values(
            'health_zone__name', 'month', 'temperature'
        )[:5]
        print("   Exemples:")
        for s in samples:
            print(f"   - Zone: {s['health_zone__name']}, Date: {s['month']}, T(K): {s['temperature']}")

    if aberrant_high > 0:
        print(f"\n⚠️  VALEURS TEMPÉRATURE TROP HAUTES (>320K): {aberrant_high:,} enregistrements")

    # 6. Précipitations
    precip_stats = ClimateData.objects.aggregate(
        min_precip=Min('precipitation'),
        max_precip=Max('precipitation'),
        avg_precip=Avg('precipitation')
    )
    print(f"\n💧 PRÉCIPITATIONS (m):")
    print(f"   Min: {precip_stats['min_precip']:.6f}")
    print(f"   Max: {precip_stats['max_precip']:.6f}")
    print(f"   Moy: {precip_stats['avg_precip']:.6f}")

    # 7. Humidité
    humidity_stats = ClimateData.objects.aggregate(
        min_hum=Min('humidity'),
        max_hum=Max('humidity'),
        avg_hum=Avg('humidity')
    )
    print(f"\n💨 HUMIDITÉ (%):")
    print(f"   Min: {humidity_stats['min_hum']:.1f}")
    print(f"   Max: {humidity_stats['max_hum']:.1f}")
    print(f"   Moy: {humidity_stats['avg_hum']:.1f}")

    # 8. Provinces couvertes
    provinces = Province.objects.filter(
        territories__health_zones__climatedata__isnull=False
    ).distinct().count()
    total_provinces = Province.objects.count()
    print(f"\n🗺️  Provinces avec données: {provinces}/{total_provinces}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC TERMINÉ")
    print("=" * 80)

if __name__ == '__main__':
    diagnose_climate_data()
