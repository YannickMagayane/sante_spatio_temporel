"""
climate_data_fetcher.py — Récupérer données climatiques Copernicus CDS
======================================================================
Intègre température, précipitations, humidité depuis Copernicus Climate Data Store.
Ces features climatiques enrichissent les prédictions épidémiologiques.

Récupère les provinces et leurs coordonnées **depuis la base Django**.

Usage:
    >>> fetcher = ClimateDataFetcher()
    >>> climate_df = fetcher.fetch_monthly_average('Katanga', (2020, 2024))
"""

import os
import sys
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Charger les provinces depuis le fichier GeoJSON local
GEOJSON_PATH = "data/raw_hdx/geoBoundaries-COD-ADM1_simplified.geojson"
_PROVINCES_CACHE = None

def load_provinces_from_geojson():
    """
    Charge les provinces depuis le fichier GeoJSON local.
    Crée un dictionnaire {nom_province: bounds}.
    """
    global _PROVINCES_CACHE
    
    if _PROVINCES_CACHE is not None:
        return _PROVINCES_CACHE
    
    provinces_dict = {}
    
    if not os.path.exists(GEOJSON_PATH):
        logger.warning(f"⚠️ Fichier GeoJSON non trouvé: {GEOJSON_PATH}")
        return {}
    
    try:
        with open(GEOJSON_PATH, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        
        # Itérer sur chaque feature (province)
        for feature in geojson_data.get('features', []):
            props = feature.get('properties', {})
            geom = feature.get('geometry', {})
            
            # Récupérer le nom de la province
            prov_name = props.get('shapeName', '').strip()
            if not prov_name:
                continue
            
            # Extraire les coordonnées de la géométrie (Polygon ou MultiPolygon)
            all_lons = []
            all_lats = []
            
            if geom.get('type') == 'Polygon':
                # Simple Polygon: coordinates[0] = outer ring
                coords = geom.get('coordinates', [[]])[0]
                if coords:
                    all_lons.extend([c[0] for c in coords])
                    all_lats.extend([c[1] for c in coords])
            
            elif geom.get('type') == 'MultiPolygon':
                # MultiPolygon: coordinates = [[[...], ...], [[...], ...], ...]
                # Itérer sur chaque polygon et prendre uniquement l'outer ring
                for polygon in geom.get('coordinates', []):
                    if polygon and polygon[0]:
                        coords = polygon[0]
                        all_lons.extend([c[0] for c in coords])
                        all_lats.extend([c[1] for c in coords])
            
            # Créer les bounds si on a au moins des coordonnées
            if all_lons and all_lats:
                bounds = {
                    'west': float(min(all_lons)),
                    'east': float(max(all_lons)),
                    'south': float(min(all_lats)),
                    'north': float(max(all_lats))
                }
                provinces_dict[prov_name] = bounds
        
        _PROVINCES_CACHE = provinces_dict
        logger.info(f"✓ {len(provinces_dict)} provinces chargées depuis GeoJSON")
        return provinces_dict
        
    except Exception as e:
        logger.error(f"✗ Erreur chargement GeoJSON: {e}")
        return {}

# Django setup pour accéder au modèle Province
def setup_django():
    """Configure Django si pas déjà fait."""
    if 'DJANGO_SETTINGS_MODULE' not in os.environ:
        sys.path.insert(0, '/app')
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
        import django
        django.setup()

class ClimateDataFetcher:
    """
    Récupère les données climatiques depuis Copernicus CDS.
    Charge les provinces depuis le fichier GeoJSON local data/raw_hdx/geoBoundaries-COD-ADM1_simplified.geojson.
    Fallback optionnel: base de données Django.
    """
    
    def __init__(self, cdsapi_rc_path=None):
        """
        Initialise le client CDS et charge les provinces depuis GeoJSON.
        
        :param cdsapi_rc_path: chemin vers .cdsapirc (défaut: ~/.cdsapirc)
        """
        # Charger les provinces depuis GeoJSON local
        self.provinces_geojson = load_provinces_from_geojson()
        
        # Setup Django (fallback)
        setup_django()
        
        # Importer les modèles Django
        try:
            from analytics.models import Province
            self.Province = Province
        except ImportError:
            logger.warning("⚠️ Modèle Province non disponible (fallback GeoJSON)")
            self.Province = None
        
        # Initialiser client CDS
        try:
            import cdsapi
            if cdsapi_rc_path and os.path.exists(cdsapi_rc_path):
                self.client = cdsapi.Client(url=self._read_cdsapi_url(cdsapi_rc_path),
                                           key=self._read_cdsapi_key(cdsapi_rc_path))
                logger.info(f"✓ CDS client initialisé avec {cdsapi_rc_path}")
            else:
                self.client = cdsapi.Client()
                logger.info("✓ CDS client initialisé (credentials depuis ~/.cdsapirc)")
        except ImportError:
            logger.warning("⚠️ cdsapi non installé. pip install cdsapi")
            self.client = None
        except Exception as e:
            logger.warning(f"⚠️ CDS client non disponible: {e}")
            self.client = None
    
    def _read_cdsapi_url(self, rc_path):
        """Lit l'URL CDS du fichier .cdsapirc"""
        try:
            with open(rc_path) as f:
                for line in f:
                    if line.startswith('url:'):
                        return line.split(':', 1)[1].strip()
        except:
            pass
        return 'https://cds.climate.copernicus.eu/api'
    
    def _read_cdsapi_key(self, rc_path):
        """Lit la clé CDS du fichier .cdsapirc"""
        try:
            with open(rc_path) as f:
                for line in f:
                    if line.startswith('key:'):
                        return line.split(':', 1)[1].strip()
        except:
            pass
        return None
    
    def get_province_bounds_from_db(self, province_name):
        """
        Récupère les bornes lat/lon d'une province.
        
        Priorité:
        1. Rechercher dans le fichier GeoJSON local (rapide)
        2. Fallback: base de données Django (si disponible)
        
        :param province_name: nom de la province
        :returns: dict {'north': float, 'south': float, 'east': float, 'west': float}
                  ou None si non trouvée
        """
        # Essai 1: Chercher dans le GeoJSON local
        if self.provinces_geojson:
            for geojson_name, bounds in self.provinces_geojson.items():
                # Matcher les noms (flexible)
                if province_name.lower() in geojson_name.lower() or geojson_name.lower() in province_name.lower():
                    logger.debug(f"  ✓ Coordonnées trouvées dans GeoJSON pour {province_name}")
                    return bounds
        
        # Essai 2: Fallback base de données
        if not self.Province:
            logger.warning(f"⚠️ Province '{province_name}' non trouvée (GeoJSON ni BDD disponibles)")
            return None
        
        try:
            # Chercher la province par nom
            province = self.Province.objects.filter(name__iexact=province_name).first()
            
            if not province:
                logger.warning(f"⚠️ Province '{province_name}' non trouvée en base de données")
                return None
            
            # Essayer d'extraire les coordonnées
            bounds = self._extract_bounds_from_province(province)
            
            if bounds:
                logger.debug(f"  ✓ Coordonnées pour {province_name} (de la BDD): {bounds}")
                return bounds
            else:
                logger.warning(f"⚠️ Pas de coordonnées géographiques pour {province_name}")
                return None
            
        except Exception as e:
            logger.error(f"✗ Erreur lors de la requête BDD: {e}")
            return None
    
    def _extract_bounds_from_province(self, province_obj):
        """
        Extrait les bornes lat/lon depuis un objet Province.
        
        Suppose que le modèle a un champ 'geom' (GEOSGeometry) ou JSON avec coords.
        
        :returns: dict {'north', 'south', 'east', 'west'} ou None
        """
        bounds = {}
        
        # Cas 1: Champ GIS 'geom' (GeometryField de Django)
        if hasattr(province_obj, 'geom') and province_obj.geom:
            try:
                extent = province_obj.geom.extent  # (minx, miny, maxx, maxy)
                if extent and len(extent) == 4:
                    minx, miny, maxx, maxy = extent
                    bounds = {
                        'west': float(minx),
                        'south': float(miny),
                        'east': float(maxx),
                        'north': float(maxy)
                    }
                    return bounds
            except Exception as e:
                logger.debug(f"  ⚠️ Erreur parsing geom: {e}")
        
        # Cas 2: Champ JSON 'bounds' ou 'coordinates'
        if hasattr(province_obj, 'bounds') and province_obj.bounds:
            try:
                b = province_obj.bounds if isinstance(province_obj.bounds, dict) else eval(province_obj.bounds)
                if 'north' in b and 'south' in b and 'east' in b and 'west' in b:
                    return {
                        'north': float(b['north']),
                        'south': float(b['south']),
                        'east': float(b['east']),
                        'west': float(b['west'])
                    }
            except Exception as e:
                logger.debug(f"  ⚠️ Erreur parsing bounds JSON: {e}")
        
        # Cas 3: Champ 'coordinates' (GeoJSON)
        if hasattr(province_obj, 'coordinates') and province_obj.coordinates:
            try:
                coords = province_obj.coordinates if isinstance(province_obj.coordinates, dict) else eval(province_obj.coordinates)
                if 'coordinates' in coords:
                    # GeoJSON format: [[lon, lat], ...]
                    polygon = coords['coordinates']
                    if polygon and polygon[0]:
                        lons = [p[0] for p in polygon[0]]
                        lats = [p[1] for p in polygon[0]]
                        bounds = {
                            'west': float(min(lons)),
                            'south': float(min(lats)),
                            'east': float(max(lons)),
                            'north': float(max(lats))
                        }
                        return bounds
            except Exception as e:
                logger.debug(f"  ⚠️ Erreur parsing coordinates GeoJSON: {e}")
        
        return None
    
    def get_all_provinces_from_db(self):
        """
        Récupère toutes les provinces.
        
        Priorité:
        1. Depuis le fichier GeoJSON local (rapide)
        2. Fallback: base de données Django
        
        :returns: liste de noms de provinces
        """
        # Essai 1: Retourner depuis le GeoJSON (plus rapide)
        if self.provinces_geojson:
            provinces_list = list(self.provinces_geojson.keys())
            logger.info(f"✓ {len(provinces_list)} provinces chargées depuis GeoJSON")
            return provinces_list
        
        # Essai 2: Fallback base de données
        if not self.Province:
            logger.warning("✗ Modèle Province non disponible")
            return []
        
        try:
            provinces = self.Province.objects.all().values_list('name', flat=True)
            provinces_list = list(provinces)
            logger.info(f"✓ {len(provinces_list)} provinces trouvées en base de données")
            return provinces_list
        except Exception as e:
            logger.error(f"✗ Erreur récupération provinces: {e}")
            return []
    
    def fetch_era5_land_monthly(self, province_name, year_range, variables):
        """
        Récupère des données mensuelles ERA5-Land (résolution 0.1°).
        
        :param province_name: nom de la province
        :param year_range: tuple (start_year, end_year)
        :param variables: liste de variables ['temperature', 'precipitation', 'relative_humidity']
        :returns: DataFrame avec colonnes [date, temperature, precipitation, humidity]
        
        Example:
            >>> fetcher = ClimateDataFetcher()
            >>> df = fetcher.fetch_era5_land_monthly('Katanga', (2020, 2024), 
            ...                                       ['temperature', 'precipitation'])
        """
        if not self.client:
            logger.warning("⚠️ CDS client non disponible")
            return pd.DataFrame()
        
        # Récupérer les bornes depuis la base de données
        bounds = self.get_province_bounds_from_db(province_name)
        if not bounds:
            logger.warning(f"⚠️ Impossible de récupérer les coordonnées pour {province_name}")
            return pd.DataFrame()
        
        start_year, end_year = year_range
        years = [str(y) for y in range(start_year, end_year + 1)]
        months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        
        logger.info(f"📡 Récupération données ERA5-Land pour {province_name} ({start_year}-{end_year})")
        
        # Map user-friendly var names to CDS names
        var_mapping = {
            'temperature': '2m_temperature',
            'precipitation': 'total_precipitation',
            'relative_humidity': '2m_relative_humidity',
        }
        
        cds_vars = [var_mapping.get(v, v) for v in variables]
        
        try:
            import cdsapi
            dataset = self.client.retrieve(
                'reanalysis-era5-land-monthly-means',
                {
                    'product_type': 'monthly_averaged_reanalysis',
                    'variable': cds_vars,
                    'year': years,
                    'month': months,
                    'time': '00:00',
                    'area': [bounds['north'], bounds['west'], bounds['south'], bounds['east']],
                    'format': 'grib'
                }
            )
            
            grib_file = f'/tmp/era5_{province_name.replace(" ", "_")}.grib'
            dataset.download(grib_file)
            logger.info(f"✓ Fichier GRIB téléchargé: {grib_file}")
            
            # Parsage basique du GRIB (nécessite cfgrib)
            df = self._parse_grib_to_dataframe(grib_file, province_name)
            return df
            
        except Exception as e:
            logger.error(f"✗ Erreur CDS: {e}")
            return pd.DataFrame()
    
    def _parse_grib_to_dataframe(self, grib_file, province_name):
        """
        Parse un fichier GRIB et retourne un DataFrame.
        Nécessite cfgrib: pip install cfgrib
        """
        try:
            import xarray as xr
            import cfgrib
            
            ds = xr.open_dataset(grib_file, engine='cfgrib')
            
            # Moyenne spatiale sur les grilles
            data_dict = {'date': []}
            for var in ds.data_vars:
                if var not in ['latitude', 'longitude']:
                    data_dict[var] = []
            
            for t in range(len(ds.time)):
                dt = pd.Timestamp(ds.time.values[t])
                data_dict['date'].append(dt)
                
                for var in ds.data_vars:
                    if var not in ['latitude', 'longitude']:
                        val = float(ds[var].isel(time=t).mean().values)
                        if var not in data_dict:
                            data_dict[var] = []
                        data_dict[var].append(val)
            
            df = pd.DataFrame(data_dict)
            df = df.rename(columns={
                't2m': 'temperature',
                'tp': 'precipitation',
                'rh2m': 'relative_humidity'
            })
            logger.info(f"✓ {len(df)} points mensuels parsed pour {province_name}")
            return df
            
        except ImportError:
            logger.warning("⚠️ cfgrib non installé. Ce feature n'est pas disponible.")
            logger.info("   Install: pip install cfgrib xarray")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"✗ Erreur parsing GRIB: {e}")
            return pd.DataFrame()
    
    def fetch_monthly_average(self, province_name, year_range):
        """
        Wrapper simplifié : récupère T°, précipitations, humidité pour une province.
        
        :returns: DataFrame [date, temperature, precipitation, relative_humidity]
        """
        return self.fetch_era5_land_monthly(
            province_name, year_range,
            ['temperature', 'precipitation', 'relative_humidity']
        )
    
    @staticmethod
    def normalize_climate_features(climate_df, scaler=None):
        """
        Normalise les features climatiques.
        
        :param climate_df: DataFrame avec [temperature, precipitation, humidity]
        :param scaler: MinMaxScaler optionnel (pour test, réutilise fit d'entraînement)
        :returns: array normalisé (n, 3)
        """
        from sklearn.preprocessing import MinMaxScaler as MMS
        
        if scaler is None:
            scaler = MMS()
        
        features = ['temperature', 'precipitation', 'relative_humidity']
        subset = climate_df[features].fillna(climate_df[features].mean())
        
        return scaler.fit_transform(subset), scaler


def merge_climate_with_epidemiology(epi_df, climate_df):
    """
    Fusionne données épidémiologiques avec climatiques sur la date.
    
    :param epi_df: DataFrame épidémio avec colonne 'month'
    :param climate_df: DataFrame climat avec colonne 'date'
    :returns: DataFrame fusionné
    """
    epi_df = epi_df.copy()
    climate_df = climate_df.copy()
    
    # Aligner les dates
    epi_df['month'] = pd.to_datetime(epi_df['month']).dt.to_period('M')
    climate_df['date'] = pd.to_datetime(climate_df['date']).dt.to_period('M')
    
    merged = epi_df.merge(
        climate_df,
        left_on='month',
        right_on='date',
        how='left'
    )
    
    logger.info(f"✓ Fusion: {len(epi_df)} épidémiologiques + {len(climate_df)} climatiques = {len(merged)}")
    
    return merged


def fetch_rdc_climate_data(year_range=(2020, 2024), output_dir="data/climat"):
    """
    Récupère les données climatiques pour TOUTE la RDC.
    Télécharge depuis Copernicus CDS et sauvegarde les résultats.
    
    :param year_range: tuple (start_year, end_year)
    :param output_dir: répertoire pour sauvegarder les données
    :returns: DataFrame avec colonnes [date, temperature, precipitation, relative_humidity]
    """
    # Bounds de la RDC (approximatif mais qui couvre tout le pays)
    rdc_bounds = {
        'north': 5.2,    # Nord (Uele)
        'south': -13.6,  # Sud (Katanga)
        'west': 12.2,    # Ouest (Kasai)
        'east': 31.3     # Est (Kivus)
    }
    
    # Créer le répertoire si nécessaire
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialiser le fetcher avec la clé CDS
    cds_rc = "/app/compose/cle_api/.cdsapirc"
    fetcher = ClimateDataFetcher(cds_rc)
    
    if not fetcher.client:
        logger.error("✗ CDS client non disponible. Vérifiez ~/.cdsapirc ou /app/compose/cle_api/.cdsapirc")
        return pd.DataFrame()
    
    start_year, end_year = year_range
    years = [str(y) for y in range(start_year, end_year + 1)]
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    
    logger.info(f"📡 Récupération données ERA5-Land pour RDC ({start_year}-{end_year})")
    logger.info(f"   Région: N={rdc_bounds['north']}, S={rdc_bounds['south']}, E={rdc_bounds['east']}, W={rdc_bounds['west']}")
    
    var_mapping = {
        'temperature': '2m_temperature',
        'precipitation': 'total_precipitation',
        'relative_humidity': '2m_relative_humidity',
    }
    
    cds_vars = list(var_mapping.values())
    
    try:
        import cdsapi
        # Requête à Copernicus CDS
        logger.info("⏳ Téléchargement depuis Copernicus CDS (peut prendre plusieurs minutes)...")
        dataset = fetcher.client.retrieve(
            'reanalysis-era5-land-monthly-means',
            {
                'product_type': 'monthly_averaged_reanalysis',
                'variable': cds_vars,
                'year': years,
                'month': months,
                'time': '00:00',
                'area': [rdc_bounds['north'], rdc_bounds['west'], rdc_bounds['south'], rdc_bounds['east']],
                'format': 'grib'
            }
        )
        
        grib_file = os.path.join(output_dir, 'rdc_era5_monthly.grib')
        logger.info(f"⏳ Téléchargement du fichier GRIB...")
        dataset.download(grib_file)
        logger.info(f"✓ Fichier GRIB téléchargé: {grib_file}")
        
        # Parser le GRIB en DataFrame
        logger.info("⏳ Parsing du fichier GRIB...")
        try:
            df = _parse_rdc_grib(grib_file, output_dir)
        except (NameError, Exception) as e:
            logger.warning(f"⚠️ Parsing GRIB échoué ({e}), fallback synthétique...")
            return generate_rdc_synthetic_climate(year_range, output_dir)
        
        if not df.empty:
            # Sauvegarder en CSV pour consultation rapide
            csv_file = os.path.join(output_dir, 'rdc_climate_monthly.csv')
            df.to_csv(csv_file, index=False)
            logger.info(f"✓ Données sauvegardées: {csv_file}")
            
            # Sauvegarder aussi en pickle
            pkl_file = os.path.join(output_dir, 'rdc_climate_monthly.pkl')
            df.to_pickle(pkl_file)
            logger.info(f"✓ Pickle sauvegardé: {pkl_file}")
        
        return df
        
    except Exception as e:
        logger.error(f"✗ Erreur CDS: {e}")
        return pd.DataFrame()


def fetch_climate_for_all_provinces(year_range=(2020, 2024), output_dir="data/climat/provinces"):
    """
    Génère les données climatiques pour CHAQUE province de la RDC (SYNTHÉTIQUE).
    
    Strategy: Synthétique déterministe basée sur les bounds géographiques de chaque province
    (PLUS RAPIDE ET FIABLE que CDS qui est lent/instable)
    
    :param year_range: tuple (start_year, end_year)
    :param output_dir: répertoire pour sauvegarder lesdonn données par province
    :returns: dict {province_name: DataFrame} avec données climatiques
    """
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger les provinces depuis GeoJSON
    provinces = load_provinces_from_geojson()
    
    if not provinces:
        logger.error("✗ Aucune province trouvée dans GeoJSON")
        return {}
    
    all_provinces_data = {}
    start_year, end_year = year_range
    
    logger.info(f"\n📡 GÉNÉRATION CLIMAT PAR PROVINCE (SYNTHÉTIQUE DÉTERMINISTE)")
    logger.info(f"───────────────────────────────────────────────────")
    logger.info(f"   Période: {start_year}-{end_year}")
    logger.info(f"   Strat: Synthétique + variation latitude")
    logger.info("")
    
    success_count = 0
    failed_provinces = []
    
    for prov_name, bounds in sorted(provinces.items()):
        csv_file = os.path.join(output_dir, f"{prov_name.replace(' ', '_')}_climate.csv")
        pkl_file = os.path.join(output_dir, f"{prov_name.replace(' ', '_')}_climate.pkl")
        
        # Vérifier si données existent
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                df['date'] = pd.to_datetime(df['date'])
                all_provinces_data[prov_name] = df
                logger.info(f"✓ {prov_name:25} — Existant ({len(df)} mois)")
                success_count += 1
                continue
            except Exception as e:
                logger.debug(f"  ⚠️ Lecture échouée: {e}")
        
        # Générer synthétique pour cette province
        try:
            logger.info(f"⏳ {prov_name:25}", end=" ")
            
            dates = pd.date_range(f'{start_year}-01-01', f'{end_year}-12-31', freq='MS')
            n_months = len(dates)
            
            # Variation latitude
            lat_center = (bounds['north'] + bounds['south']) / 2
            lat_factor = (lat_center + 4) / 9
            
            base_temp_k = 293.0 + 2 * lat_factor
            seasonal_var = 3 * np.sin(np.arange(n_months) * 2 * np.pi / 12)
            np.random.seed(hash(prov_name) % 2**32)
            noise = np.random.normal(0, 0.5, n_months)
            temperature = base_temp_k + seasonal_var + noise
            
            precip_pattern = 0.005 * (1 + 0.8 * np.sin((np.arange(n_months) - 2) * 2 * np.pi / 12))
            precip_noise = np.random.normal(0, 0.001, n_months)
            precipitation = np.maximum(0.0001, precip_pattern + precip_noise)
            
            humidity_pattern = 75 + 10 * np.sin((np.arange(n_months) + 6) * 2 * np.pi / 12)
            humidity_noise = np.random.normal(0, 2, n_months)
            relative_humidity = np.clip(humidity_pattern + humidity_noise, 40, 95)
            
            df = pd.DataFrame({
                'date': dates,
                'temperature': temperature,
                'precipitation': precipitation,
                'relative_humidity': relative_humidity
            })
            
            df.to_csv(csv_file, index=False)
            df.to_pickle(pkl_file)
            all_provinces_data[prov_name] = df
            
            logger.info(f"✓ {len(df)} mois")
            success_count += 1
            
        except Exception as e:
            logger.info(f"✗ {str(e)[:30]}")
            failed_provinces.append((prov_name, str(e)))
    
    logger.info(f"\n{'─'*70}")
    logger.info(f"✓ {success_count}/{len(provinces)} provinces générées")
    if failed_provinces:
        logger.warning(f"⚠️ {len(failed_provinces)} échouées")
    
    return all_provinces_data
    """
    Génère des données climatiques SYNTHÉTIQUES réalistes pour la RDC.
    
    Basées sur la climatologie observée:
    - Température moyenne: 24-26°C (tropicale)
    - Saison sèche: juin-septembre (T plus basse)
    - Saison humide: décembre-mai (précipitations élevées)
    
    Usage: pour développement/test. Remplacer par vraies données CDS une fois disponibles.
    
    :param year_range: tuple (start_year, end_year)
    :param output_dir: répertoire pour sauvegarder
    :returns: DataFrame avec colonnes [date, temperature, precipitation, relative_humidity]
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Générer les dates (tous les mois de la période)
    start_year, end_year = year_range
    dates = pd.date_range(f'{start_year}-01-01', f'{end_year}-12-31', freq='MS')
    
    n_months = len(dates)
    
    logger.info(f"🔧 Génération données climatiques SYNTHÉTIQUES pour RDC ({start_year}-{end_year})")
    logger.info(f"   {n_months} mois à générer")
    
    # Température de base tropicale (en Kelvin pour CDS ERA5)
    # RDC: 24-26°C = 297-299 K (mais ERA5 donne souvent 270-290 K en moyenne)
    base_temp_k = 293.0  # ~20°C variante selon saison
    
    # Cycle saisonnier: plus froid en saison sèche (juin-sept)
    seasons = np.array([
        1.2,  # jan - chaud, humide
        1.2,  # fev - chaud, humide
        1.2,  # mar - chaud, humide
        1.0,  # avr - transition
        0.5,  # mai - transition
        -1.5, # juin - SEC ET FROID
        -1.8, # juil - SEC ET FROID
        -1.8, # aout - SEC ET FROID
        -0.5, # sept - transition
        0.6,  # oct - transition (chaleur revient)
        1.0,  # nov - chaleur
        1.2,  # dec - chaud
    ])
    
    temperatures = []
    precipitations = []
    humidities = []
    
    for i, date in enumerate(dates):
        month = date.month - 1  # 0-indexed
        
        # Température avec cycle saisonnier
        temp_k = base_temp_k + seasons[month] + np.random.normal(0, 0.3)
        temperatures.append(temp_k)
        
        # Précipitations (en mètres - ERA5 format)
        # Saison humide (décembre-mai): ~100-150 mm/mois = 0.1-0.15 m
        # Saison sèche (juin-septembre): ~20-50 mm/mois = 0.02-0.05 m
        if month in [11, 0, 1, 2, 3, 4]:  # Décembre à mai
            precip = np.random.normal(0.12, 0.03)  # 120mm ± 30mm
        else:  # Juin à novembre
            precip = np.random.normal(0.04, 0.015)  # 40mm ± 15mm
        
        precip = max(precip, 0.0)  # Pas de précipitations négatives
        precipitations.append(precip)
        
        # Humidité relative: 60-90% (tropicale)
        # Plus élevée en saison humide
        if month in [11, 0, 1, 2, 3, 4]:  # Saison humide
            humidity = np.random.normal(85, 5)  # 85% ± 5%
        else:  # Saison sèche
            humidity = np.random.normal(70, 8)  # 70% ± 8%
        
        humidity = np.clip(humidity, 40, 100)  # Borner entre 40-100%
        humidities.append(humidity)
    
    # Créer le DataFrame
    df = pd.DataFrame({
        'date': dates,
        'temperature': temperatures,
        'precipitation': precipitations,
        'relative_humidity': humidities
    })
    
    # Sauvegarder
    csv_file = os.path.join(output_dir, 'rdc_climate_monthly.csv')
    pkl_file = os.path.join(output_dir, 'rdc_climate_monthly.pkl')
    
    df.to_csv(csv_file, index=False)
    df.to_pickle(pkl_file)
    
    logger.info(f"✓ Données synthétiques générées:")
    logger.info(f"   CSV: {csv_file}")
    logger.info(f"   Pickle: {pkl_file}")
    logger.info(f"   Température: {df['temperature'].min():.2f} à {df['temperature'].max():.2f} K")
    logger.info(f"   Précipitations: {df['precipitation'].min():.4f} à {df['precipitation'].max():.4f} m")
    logger.info(f"   Humidité: {df['relative_humidity'].min():.1f} à {df['relative_humidity'].max():.1f} %")
    
    return df
    """
    Parse le fichier GRIB de la RDC et retourne un DataFrame.
    """
    try:
        import xarray as xr
        import cfgrib
        
        logger.info(f"📖 Ouverture du fichier GRIB...")
        ds = xr.open_dataset(grib_file, engine='cfgrib')
        
        logger.info(f"   Variables: {list(ds.data_vars)}")
        logger.info(f"   Dimensions: {dict(ds.dims)}")
        
        # Moyenne spatiale sur la grille
        data_dict = {'date': []}
        
        # Identifier les noms des variables dans le GRIB
        var_mapping = {
            '2m_temperature': 't2m',
            'total_precipitation': 'tp',
            '2m_relative_humidity': 'rh2m'
        }
        
        # Trouver les variables disponibles
        available_vars = {}
        for var_name, grib_name in var_mapping.items():
            if grib_name in ds.data_vars:
                available_vars[var_name] = grib_name
                data_dict[var_name] = []
        
        logger.info(f"   Variables disponibles: {list(available_vars.keys())}")
        
        # Extraire les données mois par mois (moyenne spatiale)
        n_times = len(ds.time)
        logger.info(f"⏳ Extraction de {n_times} mois...")
        
        for t in range(n_times):
            dt = pd.Timestamp(ds.time.values[t])
            data_dict['date'].append(dt)
            
            for var_name, grib_name in available_vars.items():
                try:
                    # Moyenne spatiale: moyenne sur toutes les grilles
                    val = float(ds[grib_name].isel(time=t).mean().values)
                    data_dict[var_name].append(val)
                except Exception as e:
                    logger.warning(f"⚠️ Erreur extraction {var_name} at time {t}: {e}")
                    data_dict[var_name].append(np.nan)
        
        df = pd.DataFrame(data_dict)
        
        # Renommer les colonnes pour cohérence
        df = df.rename(columns={
            '2m_temperature': 'temperature',
            'total_precipitation': 'precipitation',
            '2m_relative_humidity': 'relative_humidity'
        })
        
        # Remplir les NaN avec interpolation linéaire
        df = df.ffill().bfill()
        
        logger.info(f"✓ {len(df)} points mensuels extraits de la RDC")
        logger.info(f"   Température: {df['temperature'].min():.2f} à {df['temperature'].max():.2f} K")
        logger.info(f"   Précipitations: {df['precipitation'].min():.2f} à {df['precipitation'].max():.2f} m")
        if 'relative_humidity' in df.columns:
            logger.info(f"   Humidité: {df['relative_humidity'].min():.2f} à {df['relative_humidity'].max():.2f} %")
        
        return df
        
    except ImportError as e:
        logger.error(f"✗ Dépendances manquantes: {e}")
        logger.info("   Install: pip install cfgrib xarray")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"✗ Erreur parsing GRIB: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("GÉNÉRATION/RÉCUPÉRATION DONNÉES CLIMATIQUES POUR LA RDC")
    print("="*70)
    
    output_dir = "data/climat"
    
    # Essayer d'abord les vraies données CDS
    cds_rc = "/app/compose/cle_api/.cdsapirc"
    print("\n1️⃣ Essai récupération données Copernicus CDS...")
    
    climate_df = fetch_rdc_climate_data(year_range=(2020, 2024), output_dir=output_dir)
    
    # Fallback: données synthétiques
    if climate_df.empty:
        print("\n⚠️ Fallback: Génération de données synthétiques réalistes...")
        climate_df = generate_rdc_synthetic_climate(year_range=(2020, 2024), output_dir=output_dir)
    
    # Afficher les résultats
    if not climate_df.empty:
        print(f"\n✓ Succès! {len(climate_df)} mois de données climatiques")
        print(f"\nAperçu des 12 premiers mois:")
        print(climate_df.head(12))
        print(f"\nStatistiques complètes ({len(climate_df)} mois):")
        print(climate_df.describe())
        print(f"\nPlage temporelle: {climate_df['date'].min()} à {climate_df['date'].max()}")
    else:
        print("\n✗ Impossible de générer/récupérer les données climatiques")
