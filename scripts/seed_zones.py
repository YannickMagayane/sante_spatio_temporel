"""
Seed script to populate HealthZone model from HDX GeoJSON data.
"""

import os
import django
import json
import logging
from django.core.management import execute_from_command_line

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from analytics.models import HealthZone

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
        pcode = props.get('adm2_pcode')
        if not pcode: continue

        # Calculate representative point (centroid-like-ish)
        # Using the middle of coordinates if centroid is too complex to compute without deps
        # But we have GeoPandas in the requirements, so we could use it if pip finishes.
        # For seeding, we can store the raw geometry and compute lat/lon later or via properties.
        
        # Some HDX files have 'latitude' and 'longitude' in properties
        lat = props.get('latitude')
        lon = props.get('longitude')

        HealthZone.objects.update_or_create(
            code=pcode,
            defaults={
                'name': props.get('adm2_fr', 'Unknown'),
                'province': props.get('adm1_fr', 'Unknown'),
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
