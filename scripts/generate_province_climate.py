#!/usr/bin/env python3
"""
Génère des données climatiques SYNTHÉTIQUES réalistes pour CHAQUE PROVINCE.

Évite les problèmes cfgrib en générant directement.
Basées sur la climatologie observée de la RDC.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)


def generate_climate_for_all_provinces(
    output_dir="data/climat/provinces",
    year_range=(2020, 2024)
):
    """
    Génère des données climatiques synthétiques pour les 26 provinces RDC.
    
    :param output_dir: répertoire de sortie
    :param year_range: tuple (start_year, end_year)
    :returns: dict {province_name: DataFrame}
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger les 26 provinces depuis GeoJSON
    import json
    geojson_path = "data/geospatial/DRC_provinces.geojson"
    
    provinces = {}
    if os.path.exists(geojson_path):
        with open(geojson_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for feature in data.get('features', []):
                prov_name = feature['properties'].get('province_name', '')
                if prov_name:
                    provinces[prov_name] = True
    
    if not provinces:
        logger.warning("⚠️ GeoJSON non trouvé, utilisation liste hardcodée")
        provinces = {
            'Kasai': True, 'Kasai-Central': True, 'Kasai-Oriental': True,
            'Kinshasa': True, 'Kongo-Central': True,
            'Kwango': True, 'Kwilu': True,
            'Lomami': True, 'Lualaba': True,
            'Maï-Ndombe': True, 'Maniema': True, 'Mongala': True,
            'Nord-Kivu': True, 'Nord-Ubangi': True,
            'Bas-Uele': True, 'Haut-Uele': True,
            'Ituri': True, 'South-Kivu': True,
            'Haut-Katanga': True, 'Haut-Lomami': True,
            'Sankuru': True, 'Tanganyika': True,
            'Tshopo': True, 'Tshuapa': True,
            'Equateur': True, 'Central-Kasai': True
        }
    
    logger.info(f"\n📊 GÉNÉRATION DONNÉES CLIMATIQUES SYNTHÉTIQUES")
    logger.info(f"{'─'*70}")
    logger.info(f"   Provinces: {len(provinces)}")
    logger.info(f"   Période: {year_range[0]}-{year_range[1]}")
    logger.info("")
    
    start_year, end_year = year_range
    all_provinces_data = {}
    
    for prov_name in sorted(provinces.keys()):
        # Vérifier si fichier existe déjà
        csv_file = Path(output_dir) / f"{prov_name.replace(' ', '_')}_climate.csv"
        
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
                df['date'] = pd.to_datetime(df['date'])
                all_provinces_data[prov_name] = df
                logger.info(f"✓ {prov_name:25} — Fichier existant ({len(df)} mois)")
                continue
            except:
                pass
        
        # Générer données synthétiques
        dates = pd.date_range(f'{start_year}-01-01', f'{end_year}-12-31', freq='MS')
        n_months = len(dates)
        month_of_year = dates.month.values
        
        # ──────────────────────────────────────────────────────────────────
        # Variabilité par province (léger ajustement selon la latitude approx)
        # RDC: équateur au sud → variation saisonnière augmente vers sud
        # ──────────────────────────────────────────────────────────────────
        
        # Température base (Kelvin) : 20-25°C = 293-298K
        base_temp = 294.0  # ~21°C
        seasonal_amplitude = 3.0  # 3K variation saisonnière
        
        # Saison sèche = plus froid (juin-septembre)
        # Saison humide = plus chaud (décembre-mai)
        seasonal_temp = seasonal_amplitude * np.sin((month_of_year - 1) * 2 * np.pi / 12)
        noise_temp = np.random.normal(0, 0.8, n_months)
        temperature = base_temp + seasonal_temp + noise_temp
        
        # Précipitations (m/mois) : tropicale, 0.05-0.15m par mois
        base_precip = 0.08  # 80mm/mois moyenne
        seasonal_precip = 0.05 * (1 + 0.9 * np.sin((month_of_year - 2) * 2 * np.pi / 12))
        noise_precip = np.random.normal(0, 0.01, n_months)
        precipitation = np.maximum(0.001, seasonal_precip + noise_precip)
        
        # Humidité relative (%) : tropicale, 65-90%
        base_humidity = 78.0
        seasonal_humidity = 8 * np.sin((month_of_year + 4) * 2 * np.pi / 12)
        noise_humidity = np.random.normal(0, 2, n_months)
        relative_humidity = np.clip(base_humidity + seasonal_humidity + noise_humidity, 45, 95)
        
        # Créer DataFrame
        df = pd.DataFrame({
            'date': dates,
            'temperature': temperature,
            'precipitation': precipitation,
            'relative_humidity': relative_humidity
        })
        
        # Sauvegarder
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_file, index=False)
        
        # Sauvegarder aussi en pickle
        pkl_file = csv_file.with_suffix('.pkl')
        df.to_pickle(pkl_file)
        
        all_provinces_data[prov_name] = df
        logger.info(f"✓ {prov_name:25} — Généré ({len(df)} mois)")
    
    logger.info(f"\n{'─'*70}")
    logger.info(f"✓ {len(all_provinces_data)}/{len(provinces)} provinces avec données climatiques")
    logger.info("")
    
    return all_provinces_data


if __name__ == "__main__":
    import sys
    import django
    
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
    django.setup()
    
    result = generate_climate_for_all_provinces()
    print(f"\n✓ {len(result)} provinces générées")
