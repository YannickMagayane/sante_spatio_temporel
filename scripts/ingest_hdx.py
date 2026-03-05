"""
Smart HDX Data Ingestion Pipeline for HealthDRC.

Ingests real HDX data into properly structured administrative hierarchy:
    Province (26) → Territory (145) → HealthZone (519)
Then injects clinical, demographic, and nutritional data linked to the correct level.

Data Sources:
    - GeoJSON (cod_admin1/2/3.geojson) for administrative boundaries
    - CSV (cod_admpop_adm1/2_2020.csv) for real population demographics
    - CSV (rdc-ipc-amn-july2024-june2025.csv) for nutrition/malnutrition data per ZS
    - CSV (WHO indicators: malaria, TB, HIV...) for clinical data
    - XLSX (rdc_mouvement_de_population_*.xlsx) for IDP displacement data

Usage:
    docker exec -it <app_container> python scripts/ingest_hdx.py

:author: HealthDRC Team
:version: 2.0
"""

import os
import sys
import zipfile
import csv
import json
import logging
import re
import unicodedata
from datetime import date
from pathlib import Path

# Ensure the project root is in sys.path (needed inside Docker)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from django.db import transaction

from analytics.models import (
    ClinicalData,
    HealthZone,
    Province,
    Territory,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_HDX_DIR = "data/raw_hdx"
EXTRACTED_DIR = "data/extracted_hdx"

# ---------------------------------------------------------------------------
# Extraction Utility
# ---------------------------------------------------------------------------

def extract_hdx_archives():
    """Extract all ZIP files found in data/raw_hdx."""
    logger.info("📦 Extraction des archives HDX...")
    for path in Path(RAW_HDX_DIR).rglob("*.zip"):
        extract_to = Path(EXTRACTED_DIR)
        os.makedirs(extract_to, exist_ok=True)
        try:
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            logger.info(f"  ✅ Extrait : {path.name}")
        except Exception as e:
            logger.error(f"  ❌ Erreur ZIP {path.name}: {e}")

# Disease mapping: keyword → normalized disease name
DISEASES_MAP = {
    'malaria': 'Paludisme',
    'paludisme': 'Paludisme',
    'cholera': 'Choléra',
    'choléra': 'Choléra',
    'ebola': 'Ebola',
    'rougeole': 'Rougeole',
    'measles': 'Rougeole',
    'covid': 'COVID-19',
    'corona': 'COVID-19',
    'mpox': 'Mpox',
    'monkeypox': 'Mpox',
    'yellow fever': 'Fièvre Jaune',
    'fièvre jaune': 'Fièvre Jaune',
    'tuberculosis': 'Tuberculose',
    'tuberculose': 'Tuberculose',
    'hiv': 'VIH/SIDA',
    'vih': 'VIH/SIDA',
    'hepatitis': 'Hépatite',
    'hépatite': 'Hépatite',
}

# Major cities in DRC (used to differentiate Ville from 145 official Territoires)
VILLES_MAJEURES = [
    "Kinshasa", "Lubumbashi", "Goma", "Bukavu", "Kisangani",
    "Kananga", "Mbuji-Mayi", "Matadi", "Bandundu", "Mbandaka",
    "Beni", "Butembo", "Zongo", "Boma", "Likasi", "Kolwezi",
    "Tshikapa", "Uvira", "Kikwit", "Mwene-Ditu", "Gbadolite", "Isiro",
    "Baraka", "Kamituga", "Mwene Ditu", "Mbuji-Mayi",
]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def normalize_name(name):
    """
    Normalize a place name for fuzzy matching:
    strips accents, lowercases, removes hyphens and extra spaces.

    :param name: Raw place name string.
    :returns: Normalized lowercase string.

    Example:
        >>> normalize_name("Bas-Uélé")
        'bas uele'
    """
    if not name:
        return ""
    name = str(name).strip()
    # Remove accents
    nfkd = unicodedata.normalize('NFKD', name)
    ascii_str = nfkd.encode('ASCII', 'ignore').decode('ASCII')
    return ascii_str.lower().replace('-', ' ').replace('_', ' ').strip()


def parse_numeric(value):
    """
    Parse a numeric value from various formats.

    Handles formats like:
        '1 234', '1,234', '1234.5', '6690.8 [5273.6-7980.2]'

    :param value: Raw string value.
    :returns: Parsed float or None.

    Example:
        >>> parse_numeric("1 234")
        1234.0
        >>> parse_numeric("6690.8 [5273.6-7980.2]")
        6690.8
    """
    if not value:
        return None
    s = str(value).strip()
    if not s or s.lower() == 'nan':
        return None
    # Take part before '[' (confidence intervals)
    s = s.split('[')[0].strip()
    # Remove spaces and non-numeric chars except . and -
    s = s.replace(' ', '').replace(',', '.')
    # Handle European comma decimals like "5,8" -> already replaced
    try:
        return float(s)
    except ValueError:
        return None


def parse_comma_decimal(value):
    """
    Parse a European-style decimal (comma as separator).

    :param value: String with possible comma decimal.
    :returns: Parsed float or None.

    Example:
        >>> parse_comma_decimal("5,8")
        5.8
    """
    if not value:
        return None
    s = str(value).strip().replace('"', '').replace(',', '.')
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# STEP 1: Seed Provinces (ADM1 = 26 provinces)
# ---------------------------------------------------------------------------


def seed_provinces(extracted_dir=EXTRACTED_DIR):
    """
    Populate Province model from GeoJSON ADM1 boundaries.

    Creates 26 provinces of the DRC from the official HDX administrative
    boundaries GeoJSON file.

    :param extracted_dir: Directory containing extracted GeoJSON files.
    :returns: Dict mapping admin1_pcode → Province instance.

    Example:
        >>> provinces = seed_provinces()
        >>> len(provinces)
        26
    """
    import geopandas as gpd

    adm1_path = os.path.join(extracted_dir, "cod_admin1.geojson")
    if not os.path.exists(adm1_path):
        logger.error(f"❌ ADM1 GeoJSON not found: {adm1_path}")
        return {}

    gdf = gpd.read_file(adm1_path)
    logger.info(f"📍 Injection de {len(gdf)} Provinces (ADM1)...")

    provinces_map = {}
    for _, row in gdf.iterrows():
        pcode = row.get('adm1_pcode') or row.get('admin1Pcode')
        name = row.get('adm1_name') or row.get('admin1Name_fr', 'Unknown')
        if not pcode:
            continue

        province, created = Province.objects.update_or_create(
            code=pcode,
            defaults={'name': name},
        )
        provinces_map[pcode] = province
        action = "créée" if created else "mise à jour"
        logger.debug(f"  Province {name} ({pcode}) {action}")

    logger.info(f"✅ {len(provinces_map)} Provinces injectées.")
    return provinces_map


# ---------------------------------------------------------------------------
# STEP 2: Seed Territories (ADM2 = 145 territoires + villes)
# ---------------------------------------------------------------------------


def seed_territories(provinces_map, extracted_dir=EXTRACTED_DIR):
    """
    Populate Territory model from GeoJSON ADM2 boundaries and population CSV.

    Creates ~145 territories/cities, linked to the correct province.
    Population data is enriched from the ``cod_admpop_adm2_2020.csv`` file.

    :param provinces_map: Dict mapping admin1_pcode → Province instance.
    :param extracted_dir: Directory containing extracted GeoJSON files.
    :returns: Dict mapping admin2_pcode → Territory instance.

    Example:
        >>> territories = seed_territories(provinces_map)
        >>> len(territories)
        145
    """
    import geopandas as gpd

    adm2_path = os.path.join(extracted_dir, "cod_admin2.geojson")
    if not os.path.exists(adm2_path):
        logger.error(f"❌ ADM2 GeoJSON not found: {adm2_path}")
        return {}

    # Load real population data per territory
    pop_data = _load_adm2_population()

    gdf = gpd.read_file(adm2_path)
    logger.info(f"📍 Injection de {len(gdf)} Territoires/Villes (ADM2)...")

    territories_map = {}
    total_found = len(gdf)
    count_villes = 0
    count_territoires = 0

    for _, row in gdf.iterrows():
        pcode = row.get('adm2_pcode') or row.get('admin2Pcode')
        name = row.get('adm2_name') or row.get('admin2Name_fr', 'Unknown')
        parent_pcode = row.get('adm1_pcode') or row.get('admin1Pcode')
        if not pcode:
            continue

        province = provinces_map.get(parent_pcode)
        if not province:
            logger.warning(f"⚠️ Province {parent_pcode} introuvable pour {name}")
            continue

        # If it's a city or a known city, we label it.
        # Otherwise it's likely one of the 145 territories.
        is_ville = any(v.lower() == name.lower() or f"ville de {v.lower()}" in name.lower() for v in VILLES_MAJEURES)
        
        # Heuristic: Kinshasa is a city-province, but also an ADM2 unit.
        if name.lower() == 'kinshasa':
            is_ville = True
            
        type_label = "Ville" if is_ville else "Territoire"

        lat = _safe_float(row.get('center_lat'))
        lon = _safe_float(row.get('center_lon'))
        population = pop_data.get(pcode, 0)

        geometry_meta = {
            "level": "2",
            "type": type_label,
            "parent_pcode": parent_pcode,
        }

        territory, created = Territory.objects.update_or_create(
            code=pcode,
            defaults={
                'name': name,
                'province': province,
                'latitude': lat,
                'longitude': lon,
                'population': population,
                'geometry': geometry_meta,
            },
        )
        territories_map[pcode] = territory
        if is_ville:
            count_villes += 1
        else:
            count_territoires += 1

    logger.info(
        f"✅ ADM2: {count_territoires} Territoires (proche de l'objectif 145) + {count_villes} Villes "
        f"= {len(territories_map)} total."
    )
    return territories_map


def _load_adm2_population():
    """
    Load real population data from HDX population CSV (ADM2 level, 2020).

    :returns: Dict mapping admin2Pcode → total population (int).

    Example:
        >>> pop = _load_adm2_population()
        >>> pop.get('CD1000', 0)  # Kinshasa
        9924784
    """
    pop_file = os.path.join(RAW_HDX_DIR, "cod_admpop_adm2_2020.csv")
    pop_data = {}

    if not os.path.exists(pop_file):
        logger.warning(f"⚠️ Population CSV not found: {pop_file}")
        return pop_data

    with open(pop_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pcode = row.get('admin2Pcode', '').strip()
            total = parse_numeric(row.get('T_TL', '0'))
            if pcode and total:
                pop_data[pcode] = int(total)

    logger.info(f"📊 Population ADM2: {len(pop_data)} entrées chargées.")
    return pop_data


def _safe_float(val):
    """
    Safely convert a value to float.

    :param val: Value to convert.
    :returns: Float value or None.

    Example:
        >>> _safe_float('3.14')
        3.14
        >>> _safe_float(None) is None
        True
    """
    try:
        return float(val) if val is not None else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# STEP 3: Seed Health Zones (ADM3 = 519 zones de santé)
# ---------------------------------------------------------------------------


def seed_health_zones(territories_map, extracted_dir=EXTRACTED_DIR):
    """
    Populate HealthZone model from GeoJSON ADM3 boundaries.

    Creates 519 health zones, linked to the correct territory.
    Uses fuzzy matching on territory name if pcode matching fails.

    :param territories_map: Dict mapping admin2_pcode → Territory instance.
    :param extracted_dir: Directory with extracted GeoJSON files.
    :returns: Dict mapping admin3_pcode → HealthZone instance.

    Example:
        >>> zones = seed_health_zones(territories_map)
        >>> len(zones)
        519
    """
    import geopandas as gpd

    adm3_path = os.path.join(extracted_dir, "cod_admin3.geojson")
    if not os.path.exists(adm3_path):
        logger.error(f"❌ ADM3 GeoJSON not found: {adm3_path}")
        return {}

    gdf = gpd.read_file(adm3_path)
    logger.info(f"📍 Injection de {len(gdf)} Zones de Santé (ADM3)...")

    # Build reverse index for fuzzy territory matching
    territory_name_index = {}
    for pcode, territory in territories_map.items():
        key = normalize_name(territory.name)
        territory_name_index[key] = territory

    zones_map = {}
    unlinked = 0

    for _, row in gdf.iterrows():
        pcode = row.get('adm3_pcode') or row.get('admin3Pcode')
        name = row.get('adm3_name') or row.get('admin3Name_fr', 'Unknown')
        parent_pcode = row.get('adm2_pcode') or row.get('admin2Pcode')
        if not pcode:
            continue

        # Find parent territory (exact pcode or fuzzy name)
        territory = territories_map.get(parent_pcode)
        if not territory:
            parent_name = row.get('adm2_name') or row.get('admin2Name_fr', '')
            territory = territory_name_index.get(normalize_name(parent_name))
            if not territory:
                unlinked += 1

        lat = _safe_float(row.get('center_lat'))
        lon = _safe_float(row.get('center_lon'))

        zone, _ = HealthZone.objects.update_or_create(
            code=pcode,
            defaults={
                'name': name,
                'territory': territory,
                'latitude': lat,
                'longitude': lon,
                'geometry': {
                    "level": "3",
                    "type": "Zone de Santé",
                    "parent_pcode": parent_pcode,
                },
            },
        )
        zones_map[pcode] = zone

    if unlinked:
        logger.warning(f"⚠️ {unlinked} zones sans territoire parent.")
    logger.info(f"✅ {len(zones_map)} Zones de Santé injectées.")
    return zones_map


# ---------------------------------------------------------------------------
# STEP 4: Inject demographic data (population by ADM1 + ADM2)
# ---------------------------------------------------------------------------


def inject_demographics():
    """
    Inject demographic data into Province and Territory models.

    Reads ``cod_admpop_adm1_2020.csv`` population data (by age/sex) and
    stores enriched JSON metadata in the Territory geometry field.

    Example:
        >>> inject_demographics()
    """
    # ADM1 province-level demographics
    adm1_file = os.path.join(RAW_HDX_DIR, "cod_admpop_adm1_2020.csv")
    if os.path.exists(adm1_file):
        with open(adm1_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                pcode = row.get('admin1Pcode', '').strip()
                if not pcode:
                    continue
                try:
                    province = Province.objects.get(code=pcode)
                except Province.DoesNotExist:
                    continue

                # Update territory population rollup
                total = parse_numeric(row.get('T_TL', '0'))
                if total:
                    # Store demographics in province territories via bulk
                    count += 1

            logger.info(f"📊 Démographies ADM1: {count} provinces traitées.")

    # ADM2 territory-level demographics with age breakdown
    adm2_file = os.path.join(RAW_HDX_DIR, "cod_admpop_adm2_2020.csv")
    if os.path.exists(adm2_file):
        with open(adm2_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                pcode = row.get('admin2Pcode', '').strip()
                if not pcode:
                    continue
                try:
                    territory = Territory.objects.get(code=pcode)
                except Territory.DoesNotExist:
                    continue

                total = parse_numeric(row.get('T_TL', '0'))
                t_under5 = parse_numeric(row.get('T_00_04', '0'))
                t_5_19 = parse_numeric(row.get('T_05_19', '0'))
                t_adult = parse_numeric(row.get('T_adulte', '0'))
                t_elderly = parse_numeric(row.get('T_agee', '0'))
                f_total = parse_numeric(row.get('F_TL', '0'))
                m_total = parse_numeric(row.get('M_TL', '0'))

                demographics = {
                    "total": int(total) if total else 0,
                    "under_5": int(t_under5) if t_under5 else 0,
                    "age_5_19": int(t_5_19) if t_5_19 else 0,
                    "adult": int(t_adult) if t_adult else 0,
                    "elderly": int(t_elderly) if t_elderly else 0,
                    "female": int(f_total) if f_total else 0,
                    "male": int(m_total) if m_total else 0,
                    "pct_under_15": round(
                        ((t_under5 or 0) + (t_5_19 or 0) * 0.67)
                        / max(total or 1, 1) * 100,
                        1,
                    ),
                }

                # Merge demographics into existing geometry metadata
                geo = territory.geometry or {}
                geo['demographics'] = demographics
                territory.geometry = geo
                territory.population = int(total) if total else territory.population
                territory.save()
                count += 1

            logger.info(f"📊 Démographies ADM2: {count} territoires enrichis.")


# ---------------------------------------------------------------------------
# STEP 5: Inject IPC Nutrition data (Province → Territoire → Zone de Santé)
# ---------------------------------------------------------------------------


def inject_nutrition_data():
    """
    Inject IPC/AMN nutrition data per HealthZone.

    Reads the IPC malnutrition CSV which contains data at the Zone de Santé
    level. Creates ClinicalData entries for malnutrition indicators.

    Example:
        >>> inject_nutrition_data()
    """
    ipc_file = os.path.join(RAW_HDX_DIR, "rdc-ipc-amn-july2024-june2025.csv")
    if not os.path.exists(ipc_file):
        logger.warning(f"⚠️ IPC file not found: {ipc_file}")
        return

    logger.info("🍎 Injection des données nutritionnelles IPC...")

    # Build zone name index for matching
    zone_index = _build_zone_name_index()

    count = 0
    with open(ipc_file, 'r', encoding='utf-8-sig') as f:
        # Skip the 2 header lines (multi-row header)
        next(f)
        next(f)
        reader = csv.reader(f)
        header = next(reader)  # Province, Territoire, Zone de Santé (ZS),...

        for row in reader:
            if len(row) < 18:
                continue

            province_name = row[0].strip() if row[0] else ''
            territoire_name = row[1].strip() if row[1] else ''
            zone_name = row[2].strip() if row[2] else ''

            if not zone_name:
                continue

            # Find zone by name (fuzzy)
            zone = _find_zone(zone_index, zone_name, province_name)
            if not zone:
                logger.debug(f"  ⚠️ Zone non trouvée: {zone_name} ({province_name})")
                continue

            # Extract nutrition values
            # Handle different formats for 2025 specifically
            try:
                pop_girls_u5 = parse_comma_decimal(row[7]) if len(row) > 7 else 0
                pop_boys_u5 = parse_comma_decimal(row[8]) if len(row) > 8 else 0
                mag_pct = parse_comma_decimal(row[11]) if len(row) > 11 else 0
                mas_pct = parse_comma_decimal(row[14]) if len(row) > 14 else 0
                pop_fefa = parse_comma_decimal(row[17]) if len(row) > 17 else 0
                mag_fefa_pct = parse_comma_decimal(row[18]) if len(row) > 18 else 0
            except:
                continue

            # Update zone population from under-5 data (estimate total)
            if pop_girls_u5 and pop_boys_u5:
                pop_u5 = int(pop_girls_u5 + pop_boys_u5)
                # Under-5 is ~17% of population in DRC
                estimated_pop = int(pop_u5 / 0.17)
                if estimated_pop > zone.population:
                    zone.population = estimated_pop

            # Predict or include 2025 Specifically
            for target_year in [2024, 2025]:
                # Store nutrition metadata in zone geometry
                geo = zone.geometry or {}
                geo[f'nutrition_{target_year}'] = {
                    "population_under_5_girls": int(pop_girls_u5) if pop_girls_u5 else 0,
                    "population_under_5_boys": int(pop_boys_u5) if pop_boys_u5 else 0,
                    "mas_pct": mas_pct,
                    "mag_pct": mag_pct,
                    "population_fefa": int(pop_fefa) if pop_fefa else 0,
                    "mag_fefa_pct": mag_fefa_pct,
                    "priority": row[3].strip() if len(row) > 3 else None,
                }
                zone.geometry = geo
                zone.save()

                # Create clinical data for malnutrition for current and next year if projections available
                if mag_pct and pop_girls_u5 and pop_boys_u5:
                    total_u5 = int(pop_girls_u5 + pop_boys_u5)
                    cases = int(total_u5 * mag_pct / 100)
                    if cases > 0:
                        ClinicalData.objects.update_or_create(
                            health_zone=zone,
                            disease_name='Malnutrition Aiguë',
                            month=date(target_year, 1, 1),
                            defaults={
                                'cases_count': cases,
                                'deaths_count': int(cases * 0.05), # Estimated deaths from lack of data
                            },
                        )
                        count += 1

    logger.info(f"✅ Nutrition IPC (incl. 2025): {count} entrées cliniques créées.")


def _build_zone_name_index():
    """
    Build a fast lookup index for HealthZone by normalized name.

    :returns: Dict mapping normalized_name → list of HealthZone instances.

    Example:
        >>> index = _build_zone_name_index()
        >>> index['kinshasa']
        [<HealthZone: Kinshasa>]
    """
    index = {}
    for zone in HealthZone.objects.select_related('territory__province').all():
        key = normalize_name(zone.name)
        if key not in index:
            index[key] = []
        index[key].append(zone)
    return index


def _find_zone(index, zone_name, province_name=None):
    """
    Find a HealthZone by name with optional province disambiguation.

    :param index: Zone name index from ``_build_zone_name_index``.
    :param zone_name: Raw zone name to search.
    :param province_name: Province name for disambiguation.
    :returns: HealthZone instance or None.

    Example:
        >>> zone = _find_zone(index, "Aketi", "Bas-Uele")
    """
    key = normalize_name(zone_name)
    candidates = index.get(key, [])

    if not candidates:
        # Try partial matching
        for k, zones in index.items():
            if key in k or k in key:
                candidates = zones
                break

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Disambiguate by province
    if province_name:
        norm_prov = normalize_name(province_name)
        for zone in candidates:
            if zone.territory and zone.territory.province:
                if normalize_name(zone.territory.province.name) == norm_prov:
                    return zone

    return candidates[0]


# ---------------------------------------------------------------------------
# STEP 6: Inject WHO clinical indicators (national → distributed per ZS)
# ---------------------------------------------------------------------------


# Map CSV filenames to disease categories
WHO_DISEASE_FILES = {
    'malaria_indicators_cod.csv': 'Paludisme',
    'tuberculosis_indicators_cod.csv': 'Tuberculose',
    'hiv_indicators_cod.csv': 'VIH/SIDA',
    'hepatitis_indicators_cod.csv': 'Hépatite',
    'child_mortality_indicators_cod.csv': None,  # detected from indicator
    'immunization_coverage_and_vaccine_preventable_diseases_indicators_cod.csv': None,
    'neglected_tropical_diseases_indicators_cod.csv': None,
    'nutrition_indicators_cod.csv': None,
    'maternal_and_reproductive_health_indicators_cod.csv': None,
}

# Indicators that contain actual case counts (not rates/percentages)
CASE_COUNT_INDICATORS = {
    'MALARIA_EST_CASES', 'MALARIA_MICR_POS', 'MALARIA_2000_EST_CASES',
    'TB_c_newinc', 'TB_e_inc_num',
    'HIV_0000000026', 'HIV_0000000001',
    'WHS3_62',  # Measles reported cases
    'MORT_100',  # Child deaths by cause
    'CHOLERA_0000000001',
}


def inject_who_clinical_data():
    """
    Inject WHO clinical indicator data from HDX CSV files.

    National-level data is distributed intelligently across health zones
    based on population weight. Only indicators with actual case counts
    (not rates or percentages) are ingested.

    Example:
        >>> inject_who_clinical_data()
    """
    logger.info("🧪 Injection des indicateurs cliniques OMS...")

    # Get all zones with population for weighted distribution
    zones = list(
        HealthZone.objects.filter(
            code__startswith='CD',
            geometry__has_key='level',
        )
        .exclude(population=0)
        .select_related('territory__province')
    )

    # Fallback: just get all zones with population > 0
    if not zones:
        zones = list(
            HealthZone.objects.filter(population__gt=0)
            .select_related('territory__province')
        )

    if not zones:
        logger.warning("⚠️ Aucune zone avec population trouvée.")
        return

    total_pop = sum(z.population for z in zones)
    logger.info(f"📊 Distribution vers {len(zones)} zones (pop totale: {total_pop:,})")

    total_entries = 0

    for filename, forced_disease in WHO_DISEASE_FILES.items():
        filepath = os.path.join(RAW_HDX_DIR, filename)
        if not os.path.exists(filepath):
            logger.debug(f"  Fichier non trouvé: {filename}")
            continue

        logger.info(f"  📄 Traitement: {filename}")
        file_entries = 0

        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)

            for row in reader:
                gho_code = row.get('GHO (CODE)', '')
                gho_display = row.get('GHO (DISPLAY)', '')
                year_str = row.get('STARTYEAR', '') or row.get('YEAR (DISPLAY)', '')
                value_str = row.get('Numeric', '')

                # Parse year
                try:
                    year = int(str(year_str)[:4])
                except (ValueError, TypeError):
                    continue

                if year < 2010 or year > 2026:
                    continue

                # Parse value
                value = parse_numeric(value_str)
                if not value or value <= 0:
                    continue

                # Determine disease
                disease = forced_disease
                if not disease:
                    disease = _detect_disease(gho_code, gho_display)
                if not disease:
                    continue

                # Check if this is a case count indicator worth distributing
                is_case_count = gho_code in CASE_COUNT_INDICATORS
                display_lower = gho_display.lower()
                if not is_case_count:
                    # Heuristic: check if display mentions "number" and not "rate"
                    if 'number' in display_lower and 'rate' not in display_lower:
                        is_case_count = True
                    elif 'cases' in display_lower:
                        is_case_count = True

                if not is_case_count:
                    continue  # Skip rates/percentages

                # National value: distribute among zones proportionally
                d_date = date(year, 1, 1)
                national_cases = int(value)

                # Distribute to top-N zones to avoid creating 519 entries for each
                # For yearly national data, keep aggregated at province level
                file_entries += _distribute_cases_to_zones(
                    zones, total_pop, disease, d_date, national_cases,
                )

        logger.info(f"    → {file_entries} entrées cliniques.")
        total_entries += file_entries

    logger.info(f"✅ Indicateurs OMS: {total_entries} entrées cliniques totales.")


def _detect_disease(gho_code, gho_display):
    """
    Detect disease name from WHO GHO code and display text.

    :param gho_code: GHO indicator code.
    :param gho_display: GHO indicator display text.
    :returns: Normalized disease name or None.

    Example:
        >>> _detect_disease('MALARIA_EST_CASES', 'Estimated malaria cases')
        'Paludisme'
    """
    combined = f"{gho_code} {gho_display}".lower()
    for keyword, disease in DISEASES_MAP.items():
        if keyword in combined:
            return disease
    return None


def _distribute_cases_to_zones(zones, total_pop, disease, d_date, national_cases):
    """
    Distribute national case count across health zones by population weight.

    Uses a proportional distribution: each zone gets a share proportional
    to its population. Only zones receiving >= 1 case are stored.

    :param zones: List of HealthZone instances with population.
    :param total_pop: Total population across all zones.
    :param disease: Normalized disease name.
    :param d_date: Date for the clinical data entry.
    :param national_cases: Total national cases to distribute.
    :returns: Number of entries created.

    Example:
        >>> count = _distribute_cases_to_zones(zones, 90000000, 'Paludisme',
        ...                                    date(2023, 1, 1), 30000000)
    """
    if total_pop <= 0:
        return 0

    # Check if data already exists for this disease/date to avoid duplicates
    existing = set(
        ClinicalData.objects.filter(
            disease_name=disease,
            month=d_date,
        ).values_list('health_zone_id', flat=True)
    )

    to_create = []
    to_update_ids = []

    for zone in zones:
        weight = zone.population / total_pop
        zone_cases = int(national_cases * weight)
        if zone_cases < 1:
            continue

        if zone.pk in existing:
            # Update existing entry
            ClinicalData.objects.filter(
                health_zone=zone,
                disease_name=disease,
                month=d_date,
            ).update(cases_count=zone_cases)
            to_update_ids.append(zone.pk)
        else:
            to_create.append(
                ClinicalData(
                    health_zone=zone,
                    disease_name=disease,
                    month=d_date,
                    cases_count=zone_cases,
                    deaths_count=0,
                )
            )

    # Bulk create in batches of 500
    if to_create:
        ClinicalData.objects.bulk_create(to_create, batch_size=500, ignore_conflicts=True)

    return len(to_create) + len(to_update_ids)


# ---------------------------------------------------------------------------
# STEP 7: Inject IDP (Internally Displaced Persons) data from XLSX
# ---------------------------------------------------------------------------


def inject_idp_data():
    """
    Inject IDP (Internally Displaced Persons) data from HDX Excel files.

    Reads ``rdc_mouvement_de_population_deplace_*.xlsx`` files and extracts
    displacement counts per health zone or territory.

    Example:
        >>> inject_idp_data()
    """
    import pandas as pd

    logger.info("🏃 Injection des données de déplacements IDP...")

    idp_files = sorted(Path(RAW_HDX_DIR).glob("rdc_mouvement_de_population_deplace*.xlsx"))
    if not idp_files:
        logger.warning("⚠️ Aucun fichier IDP trouvé.")
        return

    zone_index = _build_zone_name_index()
    count = 0

    # Process all files that contain 2024 or 2025 in the name or the most recent ones
    files_to_process = [f for f in idp_files if "2024" in f.name or "2025" in f.name]
    if not files_to_process:
        files_to_process = idp_files[-5:]

    logger.info(f"  📄 Traitement de {len(files_to_process)} fichiers IDP (incluant 2025)...")

    for filepath in files_to_process:
        logger.info(f"    - Analyse IDP: {filepath.name}")
        try:
            xl = pd.ExcelFile(filepath, engine='openpyxl')
            for sheet_name in xl.sheet_names:
                try:
                    df = pd.read_excel(xl, sheet_name=sheet_name, nrows=2000)
                except Exception:
                    continue

                if df.empty:
                    continue

                # Find zone and IDP columns
                cols = list(df.columns)
                cols_lower = [str(c).lower() for c in cols]

                zs_col = None
                idp_col = None
                prov_col = None

                for i, c in enumerate(cols_lower):
                    if 'zone' in c and 'sant' in c:
                        zs_col = cols[i]
                    elif 'province' in c:
                        prov_col = cols[i]
                    elif any(k in c for k in ['idp', 'deplac', 'déplac', 'pdis', 'stock']):
                        if not idp_col:
                            idp_col = cols[i]
                    elif 'total' in c and not idp_col:
                        idp_col = cols[i]

                if not zs_col and not idp_col:
                    continue

                for _, row in df.iterrows():
                    zone_name = str(row.get(zs_col, '')) if zs_col else ''
                    prov_name = str(row.get(prov_col, '')) if prov_col else ''

                    if not zone_name or zone_name.lower() == 'nan':
                        continue

                    zone = _find_zone(zone_index, zone_name, prov_name)
                    if not zone:
                        continue

                    val = None
                    if idp_col:
                        try:
                            val = int(pd.to_numeric(row[idp_col], errors='coerce'))
                        except (ValueError, TypeError):
                            continue

                    if val and val > 0:
                        # Store IDP data in zone metadata
                        geo = zone.geometry or {}
                        geo['idp'] = {
                            "displaced_count": val,
                            "source_file": filepath.name,
                        }
                        zone.geometry = geo
                        zone.save()
                        count += 1

        except Exception as e:
            logger.error(f"❌ Erreur IDP {filepath.name}: {e}")

    logger.info(f"✅ IDP: {count} zones mises à jour.")


# ---------------------------------------------------------------------------
# STEP 8: Inject additional CSV clinical data (all WHO datasets)
# ---------------------------------------------------------------------------


def inject_extra_csv_data():
    """
    Inject extra clinical data from all remaining WHO CSV files in raw_hdx.

    Scans CSV files not already handled by ``inject_who_clinical_data``
    and extracts identifiable disease case counts.

    Example:
        >>> inject_extra_csv_data()
    """
    logger.info("📊 Scan des CSV supplémentaires (données OMS)...")

    already_handled = set(WHO_DISEASE_FILES.keys())
    skip_prefixes = ('cod_adm', 'democratic-republic', 'infrastructure', 'urban', 'rdc-ipc')
    
    # Large files: we include them but only for case-related indicators
    include_files = {
        'health_indicators_cod.csv',
        'world_health_statistics_indicators_cod.csv',
        'noncommunicable_diseases_indicators_cod.csv',
    }
    
    # Environmental/Policy files we still skip to save time
    skip_files = {
        'air_pollution_indicators_cod.csv',
        'tobacco_control_indicators_cod.csv',
        'global_information_system_on_alcohol_and_health_indicators_cod.csv',
    }

    zones = list(
        HealthZone.objects.filter(population__gt=0)
        .select_related('territory__province')
    )
    if not zones:
        logger.warning("⚠️ Aucune zone avec population.")
        return

    total_pop = sum(z.population for z in zones)
    total_entries = 0

    # First pass: collect all (disease, year) -> national_cases to avoid
    # distributing the same disease/year multiple times
    aggregated = {}  # (disease, year) -> max national_cases

    csv_files = sorted(Path(RAW_HDX_DIR).glob("*.csv"))
    for filepath in csv_files:
        if filepath.name in already_handled:
            continue
        if filepath.name in skip_files:
            continue
        if any(filepath.name.startswith(p) for p in skip_prefixes):
            continue

        logger.info(f"  📄 Scan: {filepath.name}")
        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue

                has_gho = 'GHO (CODE)' in reader.fieldnames
                if not has_gho:
                    continue

                for row in reader:
                    gho_code = row.get('GHO (CODE)', '')
                    gho_display = row.get('GHO (DISPLAY)', '')
                    year_str = row.get('STARTYEAR', '')
                    value_str = row.get('Numeric', '')

                    disease = _detect_disease(gho_code, gho_display)
                    if not disease:
                        continue

                    try:
                        year = int(str(year_str)[:4])
                    except (ValueError, TypeError):
                        continue

                    if year < 2010 or year > 2026:
                        continue

                    value = parse_numeric(value_str)
                    if not value or value <= 0:
                        continue

                    display_lower = gho_display.lower()
                    if 'number' in display_lower or 'cases' in display_lower:
                        key = (disease, year)
                        # Keep the maximum value for deduplication
                        aggregated[key] = max(aggregated.get(key, 0), int(value))

        except Exception as e:
            logger.debug(f"  Ignoré {filepath.name}: {e}")

    # Second pass: distribute aggregated data
    logger.info(f"  📊 Distribution de {len(aggregated)} indicateurs agrégés...")
    for (disease, year), national_cases in aggregated.items():
        d_date = date(year, 1, 1)
        total_entries += _distribute_cases_to_zones(
            zones, total_pop, disease, d_date, national_cases,
        )

    logger.info(f"✅ CSV extra: {total_entries} entrées supplémentaires.")


# ---------------------------------------------------------------------------
# STEP 9: Health facility data (OSM)
# ---------------------------------------------------------------------------


def inject_health_facilities():
    """
    Inject health facility data from OSM CSV into health zone metadata.

    Reads ``democratic-republic-of-the-congo.csv`` and counts health
    facilities per zone based on proximity.

    Example:
        >>> inject_health_facilities()
    """
    osm_file = os.path.join(RAW_HDX_DIR, "democratic-republic-of-the-congo.csv")
    if not os.path.exists(osm_file):
        logger.warning("⚠️ OSM health facilities CSV not found.")
        return

    logger.info("🏥 Comptage des structures sanitaires OSM...")

    # Count facilities per health zone name
    zone_facility_count = {}
    zone_index = _build_zone_name_index()

    with open(osm_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            hz_name = row.get('is_in_health_zone', '')
            if not hz_name or hz_name.lower() == 'nan':
                continue
            key = normalize_name(hz_name)
            zone_facility_count[key] = zone_facility_count.get(key, 0) + 1

    # Update zones
    count = 0
    for key, facility_count in zone_facility_count.items():
        candidates = zone_index.get(key, [])
        for zone in candidates:
            geo = zone.geometry or {}
            geo['health_facilities_count'] = facility_count
            zone.geometry = geo
            zone.save()
            count += 1

    logger.info(f"✅ Structures sanitaires: {count} zones enrichies.")


# ---------------------------------------------------------------------------
# Main Pipeline Orchestrator
# ---------------------------------------------------------------------------


def run_full_pipeline():
    """
    Execute the complete intelligent data ingestion pipeline.

    Steps:
        1. Seed 26 Provinces (ADM1)
        2. Seed 145 Territories/Cities (ADM2) with real population
        3. Seed 519 Health Zones (ADM3) linked to territories
        4. Inject demographic data (age, sex breakdown)
        5. Inject IPC nutrition data per health zone
        6. Inject WHO clinical indicators (distributed by population weight)
        7. Inject IDP displacement data
        8. Inject extra CSV clinical data
        9. Inject OSM health facility counts

    Example:
        >>> run_full_pipeline()
    """
    logger.info("=" * 70)
    logger.info("🚀 LANCEMENT DU PIPELINE D'INGESTION INTELLIGENT (HealthDRC v2)")
    logger.info("=" * 70)

    # STEP 0: Extraction
    extract_hdx_archives()

    # Step 1: Administrative hierarchy
    logger.info("\n" + "─" * 50)
    logger.info("ÉTAPE 1/9: Provinces (ADM1)")
    logger.info("─" * 50)
    provinces = seed_provinces()

    logger.info("\n" + "─" * 50)
    logger.info("ÉTAPE 2/9: Territoires (ADM2)")
    logger.info("─" * 50)
    territories = seed_territories(provinces)

    logger.info("\n" + "─" * 50)
    logger.info("ÉTAPE 3/9: Zones de Santé (ADM3)")
    logger.info("─" * 50)
    zones = seed_health_zones(territories)

    # Step 2: Demographics
    logger.info("\n" + "─" * 50)
    logger.info("ÉTAPE 4/9: Données démographiques")
    logger.info("─" * 50)
    inject_demographics()

    # Step 3: Clinical & Nutrition
    logger.info("\n" + "─" * 50)
    logger.info("ÉTAPE 5/9: Données nutritionnelles IPC")
    logger.info("─" * 50)
    inject_nutrition_data()

    logger.info("\n" + "─" * 50)
    logger.info("ÉTAPE 6/9: Indicateurs cliniques OMS")
    logger.info("─" * 50)
    inject_who_clinical_data()

    # Step 4: IDP
    logger.info("\n" + "─" * 50)
    logger.info("ÉTAPE 7/9: Données IDP (déplacements de population)")
    logger.info("─" * 50)
    inject_idp_data()

    # Step 5: Extra
    logger.info("\n" + "─" * 50)
    logger.info("ÉTAPE 8/9: CSV cliniques supplémentaires")
    logger.info("─" * 50)
    inject_extra_csv_data()

    logger.info("\n" + "─" * 50)
    logger.info("ÉTAPE 9/9: Structures sanitaires OSM")
    logger.info("─" * 50)
    inject_health_facilities()

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("🏁 PIPELINE TERMINÉ — RÉSUMÉ:")
    logger.info(f"   📍 Provinces:       {Province.objects.count()}")
    logger.info(f"   📍 Territoires:     {Territory.objects.count()}")
    logger.info(f"   📍 Zones de Santé:  {HealthZone.objects.count()}")
    logger.info(f"   🧪 Données Cliniques: {ClinicalData.objects.count()}")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_full_pipeline()
