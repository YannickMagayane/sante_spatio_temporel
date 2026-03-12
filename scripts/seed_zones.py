"""
Seed script to populate HealthZone model from HDX GeoJSON data.
"""

import os
import sys
import django
import json
import logging

# Add the app directory to path so Django can find core module
app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from analytics.models import HealthZone, Territory, Province

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_health_zones(geojson_path="data/geospatial/zones_sante/cod_ab_health_zones.geojson"):
    if not os.path.exists(geojson_path):
        logger.error(f"GeoJSON not found at {geojson_path}. Run auto_sync.py first.")
        return

    logger.info(f"Seeding Health Zones from {geojson_path}...")
    with open(geojson_path, 'r') as f:
        data = json.load(f)

    count = 0
    for feature in data.get('features', []):
        props = feature.get('properties', {})
        geom = feature.get('geometry')
        
        # Pcode acts as a unique identifier
        pcode = props.get('adm2_pcode')  # or adm3_pcode for health zones
        if not pcode: continue

        # Get or create the Territory first
        adm2_pcode = props.get('adm2_pcode')
        adm1_name = props.get('adm1_fr', 'Unknown')
        adm2_name = props.get('adm2_fr', 'Unknown')
        
        # Ensure Province exists
        province, _ = Province.objects.get_or_create(
            code=props.get('adm1_pcode', f'prov_{adm1_name}'),
            defaults={'name': adm1_name}
        )
        
        # Ensure Territory exists
        territory, _ = Territory.objects.get_or_create(
            code=adm2_pcode,
            defaults={
                'name': adm2_name,
                'province': province,
                'latitude': props.get('latitude'),
                'longitude': props.get('longitude'),
                'population': props.get('population', 0)
            }
        )

        lat = props.get('latitude')
        lon = props.get('longitude')

        HealthZone.objects.update_or_create(
            code=pcode,
            defaults={
                'name': props.get('adm3_fr', props.get('name', 'Unknown')),
                'territory': territory,
                'latitude': lat,
                'longitude': lon,
                'geometry': geom,
                'population': props.get('population', 0)
            }
        )
        count += 1

    logger.info(f"✅ Successfully seeded {count} health zones.")

if __name__ == "__main__":
    seed_health_zones()
