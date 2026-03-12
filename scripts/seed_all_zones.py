import os
import sys
import django
import geopandas as gpd
import logging
import random

# Add the app directory to path so Django can find core module
app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from analytics.models import HealthZone, Territory, Province

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Liste des villes majeures pour la différenciation (simplifiée)
VILLES_MAJEURES = [
    "Kinshasa", "Lubumbashi", "Goma", "Bukavu", "Kisangani", 
    "Kananga", "Mbuji-Mayi", "Matadi", "Bandundu", "Mbandaka",
    "Beni", "Butembo", "Zongo", "Boma", "Likasi", "Kolwezi"
]

def seed(extracted_dir="data/extracted_hdx"):
    # Clear all data
    HealthZone.objects.all().delete()
    Territory.objects.all().delete()
    Province.objects.all().delete()
    logger.info("🗑️ Base de données vidée (Provinces, Territoires, Zones de Santé).")

    # 1. ADM1 - 26 PROVINCES
    adm1_path = os.path.join(extracted_dir, "cod_admin1.geojson")
    if os.path.exists(adm1_path):
        gdf1 = gpd.read_file(adm1_path)
        logger.info(f"📍 Injection de {len(gdf1)} Provinces (ADM1)...")
        for _, row in gdf1.iterrows():
            Province.objects.update_or_create(
                code=row['adm1_pcode'],
                defaults={
                    "name": row['adm1_name'],
                }
            )

    # 2. ADM2 - TERRITOIRES & VILLES
    adm2_path = os.path.join(extracted_dir, "cod_admin2.geojson")
    if os.path.exists(adm2_path):
        gdf2 = gpd.read_file(adm2_path)
        logger.info(f"📍 Injection de {len(gdf2)} Territoires/Villes (ADM2)...")
        count_t = 0
        count_v = 0
        for _, row in gdf2.iterrows():
            province = Province.objects.get(code=row['adm1_pcode'])
            name = row['adm2_name']
            is_ville = any(v in name for v in VILLES_MAJEURES)
            
            Territory.objects.update_or_create(
                code=row['adm2_pcode'],
                defaults={
                    "name": name,
                    "province": province,
                    "latitude": float(row['center_lat']) if row['center_lat'] else None,
                    "longitude": float(row['center_lon']) if row['center_lon'] else None,
                    "geometry": {"level": "2", "type": "Ville" if is_ville else "Territoire"},
                    "population": int(row.get('population', 0)) if row.get('population') else 0
                }
            )
            if is_ville: count_v += 1
            else: count_t += 1
        logger.info(f"✅ ADM2: {count_t} Territoires et {count_v} Villes créés.")

    # 3. ADM3 - 519 ZONES DE SANTÉ (Noyau de l'IA)
    adm3_path = os.path.join(extracted_dir, "cod_admin3.geojson")
    if os.path.exists(adm3_path):
        gdf3 = gpd.read_file(adm3_path)
        logger.info(f"📍 Injection de {len(gdf3)} Zones de Santé (ADM3)...")
        for _, row in gdf3.iterrows():
            # Get or create territory - handle case where it might not exist
            territory, _ = Territory.objects.get_or_create(
                code=row['adm2_pcode'],
                defaults={
                    'name': row['adm2_name'],
                    'province': Province.objects.filter(code=row['adm1_pcode']).first() or Province.objects.first(),
                    'latitude': float(row['center_lat']) if row['center_lat'] else None,
                    'longitude': float(row['center_lon']) if row['center_lon'] else None,
                }
            )
            
            # Métadonnées démographiques pour le dashboard
            demographics = {
                "level": "3",
                "type": "Zone de Santé",
                "age_under_15": random.randint(45, 52), # Courbe démographique RDC
                "rural_percentage": random.randint(60, 98) if territory.province.name != 'Kinshasa' else random.randint(2, 15)
            }
            HealthZone.objects.update_or_create(
                code=row['adm3_pcode'],
                defaults={
                    "name": row['adm3_name'],
                    "territory": territory,
                    "latitude": float(row['center_lat']) if row['center_lat'] else None,
                    "longitude": float(row['center_lon']) if row['center_lon'] else None,
                    "geometry": demographics,
                    "population": int(row.get('population', 0)) if row.get('population') else 0
                }
            )

    total = HealthZone.objects.count()
    logger.info(f"🏁 Amorçage terminé. Total entités : {total}")

if __name__ == "__main__":
    seed()
