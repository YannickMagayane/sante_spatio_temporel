import os
import django
import random
import logging
from datetime import date

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from analytics.models import HealthZone, ClinicalData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def distribute():
    ClinicalData.objects.all().delete()
    logger.info("🗑️ Deleted old clinical data.")

    # Base National Stats (Simplified)
    DISEASES = {
        'Paludisme': 15000000,
        'Choléra': 50000,
        'Rougeole': 300000,
        'Tuberculose': 320000,
        'Ebola': 100,
        'Fièvre Jaune': 50,
        'COVID-19': 2000,
        'Mpox': 40000
    }

    # All Health Zones (Level 3)
    zones = HealthZone.objects.filter(geometry__contains={'level': '3'})
    count = len(zones)
    logger.info(f"📍 Distributing to {count} Health Zones...")

    for disease, national_total in DISEASES.items():
        base_per_zone = national_total / (count if count > 0 else 1)
        
        clinical_objs = []
        for zone in zones:
            for year in range(2020, 2025):
                # Variabilité locale (0.2x to 5x) for realism
                variance = random.uniform(0.1, 5.0)
                # Kivu provinces usually have higher weights for Cholera/Mpox
                if zone.province in ['Nord-Kivu', 'Sud-Kivu', 'Ituri'] and disease in ['Mpox', 'Ebola', 'Choléra']:
                    variance *= 3.0
                
                cases = int(base_per_zone * variance)
                if cases > 0:
                    clinical_objs.append(ClinicalData(
                        health_zone=zone,
                        disease_name=disease,
                        month=date(year, 1, 1),
                        cases_count=cases,
                        deaths_count=int(cases * 0.02)
                    ))
        
        ClinicalData.objects.bulk_create(clinical_objs)
        logger.info(f"✅ Created data for {disease}")

    logger.info(f"🏁 Distribution finished. Total: {ClinicalData.objects.count()}")

if __name__ == "__main__":
    distribute()
