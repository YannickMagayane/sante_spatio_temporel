import os
import django
import geopandas as gpd
import logging
import random

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from analytics.models import HealthZone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Liste des villes majeures pour la différenciation (simplifiée)
VILLES_MAJEURES = [
    "Kinshasa", "Lubumbashi", "Goma", "Bukavu", "Kisangani", 
    "Kananga", "Mbuji-Mayi", "Matadi", "Bandundu", "Mbandaka",
    "Beni", "Butembo", "Zongo", "Boma", "Likasi", "Kolwezi"
]

def seed(extracted_dir="data/extracted_hdx"):
    HealthZone.objects.all().delete()
    logger.info("🗑️ Base de données vidée (Zones de Santé).")

    # 1. ADM1 - 26 PROVINCES
    adm1_path = os.path.join(extracted_dir, "cod_admin1.geojson")
    if os.path.exists(adm1_path):
        gdf1 = gpd.read_file(adm1_path)
        logger.info(f"📍 Injection de {len(gdf1)} Provinces (ADM1)...")
        for _, row in gdf1.iterrows():
            HealthZone.objects.update_or_create(
                code=row['adm1_pcode'],
                defaults={
                    "name": row['adm1_name'],
                    "province": row['adm1_name'],
                    "latitude": float(row['center_lat']) if row['center_lat'] else None,
                    "longitude": float(row['center_lon']) if row['center_lon'] else None,
                    "geometry": {"level": "1", "type": "Province"}
                }
            )

    # 2. ADM2 - 149 TERRITOIRES & VILLES
    adm2_path = os.path.join(extracted_dir, "cod_admin2.geojson")
    if os.path.exists(adm2_path):
        gdf2 = gpd.read_file(adm2_path)
        # On limite aux 149 entités demandées ou on traite toutes en distinguant
        logger.info(f"📍 Analyse de {len(gdf2)} Entités ADM2...")
        count_t = 0
        count_v = 0
        for _, row in gdf2.iterrows():
            name = row['adm2_name']
            is_ville = any(v in name for v in VILLES_MAJEURES)
            type_label = "Ville" if is_ville else "Territoire"
            
            HealthZone.objects.update_or_create(
                code=row['adm2_pcode'],
                defaults={
                    "name": name,
                    "province": row['adm1_name'],
                    "latitude": float(row['center_lat']) if row['center_lat'] else None,
                    "longitude": float(row['center_lon']) if row['center_lon'] else None,
                    "geometry": {"level": "2", "type": type_label, "parent": row['adm1_pcode']}
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
            # Métadonnées démographiques pour le dashboard
            demographics = {
                "level": "3",
                "type": "Zone de Santé",
                "parent": row['adm2_pcode'],
                "age_under_15": random.randint(45, 52), # Courbe démographique RDC
                "rural_percentage": random.randint(60, 98) if row['adm1_name'] != 'Kinshasa' else random.randint(2, 15)
            }
            HealthZone.objects.update_or_create(
                code=row['adm3_pcode'],
                defaults={
                    "name": row['adm3_name'],
                    "province": row['adm1_name'],
                    "latitude": float(row['center_lat']) if row['center_lat'] else None,
                    "longitude": float(row['center_lon']) if row['center_lon'] else None,
                    "geometry": demographics,
                    "population": random.randint(80000, 250000)
                }
            )

    total = HealthZone.objects.count()
    logger.info(f"🏁 Amorçage terminé. Total entités : {total}")

if __name__ == "__main__":
    seed()
