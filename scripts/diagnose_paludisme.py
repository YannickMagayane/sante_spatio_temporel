#!/usr/bin/env python
"""
Script diagnostic: Comparer Paludisme vs Choléra
"""
import os
import sys
import django
import pandas as pd
import numpy as np
from datetime import datetime

# Setup Django
sys.path.insert(0, '/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from analytics.models import ClinicalData
from django.db.models import Count, Min, Max, Avg, Q, Sum

print("\n╔═══════════════════════════════════════════════════════════════╗")
print("║ 🔍 DIAGNOSTIC: Paludisme vs Choléra")
print("╚═══════════════════════════════════════════════════════════════╝\n")

# 1. Comparaison globale
print("📊 COMPARAISON GLOBALE")
print("="*65)

for disease in ['Paludisme', 'Choléra', 'Tuberculose']:
    qs = ClinicalData.objects.filter(disease_name=disease)
    
    if qs.exists():
        agg = qs.aggregate(
            count=Count('id'),
            min_cases=Min('cases_count'),
            max_cases=Max('cases_count'),
            avg=Avg('cases_count'),
            min_date=Min('month'),
            max_date=Max('month'),
            zones=Count('health_zone', distinct=True)
        )
        
        print(f"\n{disease}:")
        print(f"  Records: {agg['count']:,}")
        print(f"  Zones: {agg['zones']}")
        print(f"  Min/Max cases: {agg['min_cases']}/{agg['max_cases']}")
        print(f"  Avg cases: {agg['avg']:.1f}")
        print(f"  Period: {agg['min_date']} → {agg['max_date']}")
        
        # Check for anomalies
        zeros = qs.filter(cases_count=0).count()
        nulls = qs.filter(cases_count__isnull=True).count()
        neg = qs.filter(cases_count__lt=0).count()
        
        print(f"  Zeros: {zeros} ({100*zeros/agg['count']:.1f}%)")
        if nulls > 0: print(f"  ❌ Nulls: {nulls}")
        if neg > 0: print(f"  ❌ Negatives: {neg}")

# 2. Distribution par province pour Paludisme
print(f"\n\n📍 PALUDISME PAR PROVINCE (Top 5)")
print("="*65)

paludisme_prov = ClinicalData.objects.filter(
    disease_name='Paludisme'
).values('health_zone__territory__province__name').annotate(
    total_cases=Sum('cases_count'),
    count=Count('id'),
    months=Count('month', distinct=True)
).order_by('-total_cases')[:5]

for p in paludisme_prov:
    print(f"  {p['health_zone__territory__province__name']}: {p['total_cases']:,} cases | {p['count']} records | {p['months']} months")

print("\n✅ Diagnosis complete")
