from analytics.models import HealthZone, ClinicalData
import random
from datetime import date
zones = HealthZone.objects.filter(province__icontains='Katanga')
data = []
for y in [2022, 2023]:
    for m in range(1, 13):
        for z in zones:
            data.append(ClinicalData(
                health_zone=z, 
                disease_name='Paludisme', 
                month=date(y, m, 1), 
                cases_count=random.randint(200, 800)
            ))
ClinicalData.objects.bulk_create(data)
print(f'Inserted {len(data)} climat-ready rows for Katanga')
