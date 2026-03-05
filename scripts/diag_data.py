import os
import django
import sys

sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from analytics.models import ClinicalData, Province, HealthZone
from django.db.models import Count

print("--- Diagnostic Data ---")
diseases = ClinicalData.objects.values('disease_name').annotate(count=Count('id'))
for d in diseases:
    print(f"Disease: {d['disease_name']}, Count: {d['count']}")

print("\n--- Province Check ---")
for prov in Province.objects.all():
    count = ClinicalData.objects.filter(health_zone__territory__province=prov).count()
    print(f"Province: {prov.name}, Records: {count}")

print("\n--- Missing Links ---")
unlinked_data = ClinicalData.objects.filter(health_zone__territory__isnull=True).count()
print(f"Records with health_zone having NO territory: {unlinked_data}")

total_records = ClinicalData.objects.count()
print(f"Total clinical records: {total_records}")

if total_records > 0:
    first = ClinicalData.objects.first()
    print(f"\nExample Record: {first.disease_name} in {first.health_zone.name} (Month: {first.month})")
