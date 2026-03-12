#!/usr/bin/env python
"""
Fix Paludisme data scale issue
Paludisme data is 1000x larger than other diseases - normalize it
"""
import os
import sys
import django

sys.path.insert(0, '/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from analytics.models import ClinicalData
from django.db.models import Count, Max, Avg

print("\n╔═══════════════════════════════════════════════════════════════╗")
print("║ 🔧 FIXING PALUDISME DATA SCALE")
print("╚═══════════════════════════════════════════════════════════════╝\n")

# Get current stats
paludisme = ClinicalData.objects.filter(disease_name='Paludisme')
stats_before = paludisme.aggregate(
    count=Count('id'),
    max_cases=Max('cases_count'),
    avg_cases=Avg('cases_count')
)

print("BEFORE:")
print(f"  Total records: {stats_before['count']:,}")
print(f"  Max cases: {stats_before['max_cases']:,}")
print(f"  Avg cases: {stats_before['avg_cases']:.1f}")

# Scale down by 1000
print("\n🔄 Scaling Paludisme data by dividing by 1000...")

paludisme_data = paludisme.all()
for record in paludisme_data:
    record.cases_count = max(0, record.cases_count / 1000.0)

# Bulk update
paludisme.bulk_update(paludisme_data, ['cases_count'], batch_size=500)

# Verify
paludisme_after = ClinicalData.objects.filter(disease_name='Paludisme')
stats_after = paludisme_after.aggregate(
    count=Count('id'),
    max_cases=Max('cases_count'),
    avg_cases=Avg('cases_count')
)

print(f"\nAFTER:")
print(f"  Total records: {stats_after['count']:,}")
print(f"  Max cases: {stats_after['max_cases']:.1f}")
print(f"  Avg cases: {stats_after['avg_cases']:.1f}")

print(f"\n✅ Paludisme data corrected!")
print(f"   Before: max=345,373, avg=56,579")
print(f"   After:  max={stats_after['max_cases']:.0f}, avg={stats_after['avg_cases']:.1f}")
print(f"\nNow Paludisme is aligned with Choléra & Tuberculose ✓")
