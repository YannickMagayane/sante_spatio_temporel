#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script interactif pour explorer les données ClinicalData par Maladie, Province, Année
"""

import os
import sys
import django
from datetime import datetime

sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from analytics.models import ClinicalData, Province
from django.db.models import Count, Sum, Min, Max
import pandas as pd

def print_header(text):
    print(f"\n{'='*70}")
    print(f"🔍 {text}")
    print(f"{'='*70}\n")

def show_total_stats():
    """Statistiques totales"""
    print_header("STATISTIQUES GLOBALES")
    
    total_records = ClinicalData.objects.count()
    total_cases = ClinicalData.objects.aggregate(Sum('cases_count'))['cases_count__sum'] or 0
    unique_diseases = ClinicalData.objects.values('disease_name').distinct().count()
    unique_provinces = ClinicalData.objects.values('health_zone__territory__province').distinct().count()
    unique_zones = ClinicalData.objects.values('health_zone').distinct().count()
    
    date_range = ClinicalData.objects.aggregate(
        min_date=Min('month'),
        max_date=Max('month')
    )
    
    print(f"📊 Total records: {total_records:,}")
    print(f"📊 Total cases: {total_cases:,}")
    print(f"📊 Unique diseases: {unique_diseases}")
    print(f"📊 Unique provinces: {unique_provinces}")
    print(f"📊 Unique zones: {unique_zones}")
    print(f"📊 Date range: {date_range['min_date']} → {date_range['max_date']}")

def show_by_disease():
    """Détail par maladie"""
    print_header("DONNÉES PAR MALADIE")
    
    diseases = ClinicalData.objects.values('disease_name').annotate(
        count=Count('id'),
        total_cases=Sum('cases_count')
    ).order_by('-count')
    
    print(f"{'Maladie':<30} {'Records':>12} {'Total Cases':>15}")
    print("-" * 70)
    
    for disease in diseases:
        print(f"{disease['disease_name']:<30} {disease['count']:>12,} {disease['total_cases']:>15,}")

def show_by_province():
    """Détail par province"""
    print_header("DONNÉES PAR PROVINCE")
    
    provinces = ClinicalData.objects.values(
        'health_zone__territory__province__name'
    ).annotate(
        count=Count('id'),
        total_cases=Sum('cases_count'),
        zones_count=Count('health_zone', distinct=True)
    ).order_by('-count')
    
    print(f"{'Province':<30} {'Records':>12} {'Total Cases':>15} {'Zones':>8}")
    print("-" * 70)
    
    for prov in provinces:
        name = prov['health_zone__territory__province__name'] or 'UNKNOWN'
        print(f"{name:<30} {prov['count']:>12,} {prov['total_cases']:>15,} {prov['zones_count']:>8}")

def show_by_year():
    """Détail par année"""
    print_header("DONNÉES PAR ANNÉE")
    
    # Extraire l'année du champ month
    years_data = ClinicalData.objects.extra(
        select={'year': "EXTRACT(YEAR FROM month)"}
    ).values('year').annotate(
        count=Count('id'),
        total_cases=Sum('cases_count')
    ).order_by('year')
    
    print(f"{'Année':<15} {'Records':>12} {'Total Cases':>15}")
    print("-" * 70)
    
    for year_data in years_data:
        if year_data['year']:
            print(f"{int(year_data['year']):<15} {year_data['count']:>12,} {year_data['total_cases']:>15,}")

def show_by_disease_province():
    """Détail par maladie & province"""
    print_header("DONNÉES PAR MALADIE × PROVINCE")
    
    disease_prov = ClinicalData.objects.values(
        'disease_name',
        'health_zone__territory__province__name'
    ).annotate(
        count=Count('id'),
        total_cases=Sum('cases_count')
    ).order_by('disease_name', '-total_cases')
    
    current_disease = None
    for dp in disease_prov:
        if dp['disease_name'] != current_disease:
            print(f"\n🦠 {dp['disease_name']}:")
            current_disease = dp['disease_name']
        
        prov_name = dp['health_zone__territory__province__name'] or 'UNKNOWN'
        print(f"   {prov_name:<45} {dp['count']:>8,} records | {dp['total_cases']:>10,} cases")

def show_by_disease_year():
    """Détail par maladie & année"""
    print_header("DONNÉES PAR MALADIE × ANNÉE")
    
    disease_year = ClinicalData.objects.extra(
        select={'year': "EXTRACT(YEAR FROM month)"}
    ).values('disease_name', 'year').annotate(
        count=Count('id'),
        total_cases=Sum('cases_count')
    ).order_by('disease_name', 'year')
    
    current_disease = None
    for dy in disease_year:
        if dy['disease_name'] != current_disease:
            print(f"\n🦠 {dy['disease_name']}:")
            current_disease = dy['disease_name']
        
        if dy['year']:
            print(f"   {int(dy['year']):<15} {dy['count']:>8,} records | {dy['total_cases']:>10,} cases")

def show_by_province_year():
    """Détail par province & année"""
    print_header("DONNÉES PAR PROVINCE × ANNÉE")
    
    prov_year = ClinicalData.objects.extra(
        select={'year': "EXTRACT(YEAR FROM month)"}
    ).values(
        'health_zone__territory__province__name',
        'year'
    ).annotate(
        count=Count('id'),
        total_cases=Sum('cases_count')
    ).order_by('health_zone__territory__province__name', 'year')
    
    current_province = None
    for py in prov_year:
        prov_name = py['health_zone__territory__province__name'] or 'UNKNOWN'
        if prov_name != current_province:
            print(f"\n🗺️  {prov_name}:")
            current_province = prov_name
        
        if py['year']:
            print(f"   {int(py['year']):<15} {py['count']:>8,} records | {py['total_cases']:>10,} cases")

def interactive_menu():
    """Menu interactif"""
    while True:
        print("\n" + "="*70)
        print("📊 EXPLORATEUR DE BASE DE DONNÉES — MENU INTERACTIF")
        print("="*70)
        print("\n1. Statistiques globales")
        print("2. Données par MALADIE")
        print("3. Données par PROVINCE")
        print("4. Données par ANNÉE")
        print("5. Données par MALADIE × PROVINCE")
        print("6. Données par MALADIE × ANNÉE")
        print("7. Données par PROVINCE × ANNÉE")
        print("8. Tous les rapports (long!)")
        print("9. Quitter")
        
        choice = input("\n🔹 Choisir une option (1-9): ").strip()
        
        if choice == '1':
            show_total_stats()
        elif choice == '2':
            show_by_disease()
        elif choice == '3':
            show_by_province()
        elif choice == '4':
            show_by_year()
        elif choice == '5':
            show_by_disease_province()
        elif choice == '6':
            show_by_disease_year()
        elif choice == '7':
            show_by_province_year()
        elif choice == '8':
            show_total_stats()
            show_by_disease()
            show_by_province()
            show_by_year()
            show_by_disease_province()
            show_by_disease_year()
            show_by_province_year()
        elif choice == '9':
            print("\n👋 Au revoir!")
            break
        else:
            print("❌ Option invalide. Réessayez.")

if __name__ == '__main__':
    try:
        interactive_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Arrêt...")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
