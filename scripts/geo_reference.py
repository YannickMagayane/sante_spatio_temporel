"""
Refactored Geographic reference data for DRC.
Supports Zone de Santé (ZS) level geocoding using HDX data and spatial joins.
"""

import os
import sys
import json
import logging
import geopandas as gpd
from shapely.geometry import Point

logger = logging.getLogger(__name__)

# Les provinces sont maintenant chargées dynamiquement via le GeoManager et les données HDX.

class GeoManager:
    """Manages geospatial data for DRC, specifically Health Zones (ZS)."""
    
    def __init__(self, zs_path="data/geospatial/zones_sante/cod_ab_health_zones.geojson"):
        self.zs_path = zs_path
        self._zs_gdf = None
        self._zs_name_cache = {} # To speed up string lookups
        
    @property
    def zs_gdf(self):
        """Lazy load the Health Zones GeoDataFrame."""
        if self._zs_gdf is None:
            if os.path.exists(self.zs_path):
                try:
                    logger.info(f"Loading Health Zones from {self.zs_path}...")
                    self._zs_gdf = gpd.read_file(self.zs_path)
                    # Normalize names for faster search
                    self._zs_gdf['name_clean'] = self._zs_gdf['adm2_fr'].str.lower()
                    logger.info(f"Loaded {len(self._zs_gdf)} health zones.")
                except Exception as e:
                    logger.error(f"Error loading ZS data: {e}")
            else:
                logger.warning(f"ZS data not found at {self.zs_path}. Running with limited fallback.")
        return self._zs_gdf

    def geocode_zs_from_point(self, lat, lon):
        """Find which Zone de Santé (ZS) a point belongs to."""
        if self.zs_gdf is None:
            return None
            
        point = Point(lon, lat)
        # Performance note: Spatial join or within check
        match = self.zs_gdf[self.zs_gdf.geometry.contains(point)]
        
        if not match.empty:
            row = match.iloc[0]
            return {
                "zs": row.get('adm2_fr'),
                "province": row.get('adm1_fr'),
                "lat": lat,
                "lon": lon,
                "name": row.get('adm2_fr')
            }
        return None

    def find_zs_in_text(self, text):
        """Find ZS mention in text and return its context."""
        if self.zs_gdf is None:
            return None
            
        text_lower = text.lower()
        # Look for ZS names in the text
        # Simple string matching for now (could be improved with Fuzzy matching)
        for _, row in self.zs_gdf.iterrows():
            zs_name = row['name_clean']
            if zs_name in text_lower:
                return {
                    "zs": row['adm2_fr'],
                    "province": row['adm1_fr'],
                    "lat": row.geometry.centroid.y,
                    "lon": row.geometry.centroid.x,
                    "name": row['adm2_fr']
                }
        return None

# Singleton instance
geo_manager = GeoManager()

def geocode_location(text):
    """Bridge function for backward compatibility."""
    # First try ZS
    res = geo_manager.find_zs_in_text(text)
    if res:
        return res
        
    # Fallback to simple city search if needed (legacy)
    # We can keep some of the hardcoded cities for common mentions
    KNOWN_CITIES = {
        "kinshasa": {"lat": -4.3250, "lon": 15.3222, "province": "Kinshasa", "name": "Kinshasa"},
        "goma": {"lat": -1.6585, "lon": 29.2200, "province": "Nord-Kivu", "name": "Goma"},
        # ... 
    }
    for city, data in KNOWN_CITIES.items():
        if city in text.lower():
            return data
            
    return None

def geocode_or_default(text, default="Kinshasa"):
    res = geocode_location(text)
    if res:
        return res
    # Hardcoded fallback for Goma or Kinshasa
    return {"lat": -4.3250, "lon": 15.3222, "province": "Kinshasa", "name": "Kinshasa"}
