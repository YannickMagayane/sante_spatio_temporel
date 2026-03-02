"""
Download geospatial data from OpenStreetMap and HDX for target cities in DRC.

Retrieves administrative boundaries, health zones (zones de santé),
roads, buildings, and water points using OSMnx and the Overpass API.

Example:
    $ python scripts/download_geodata.py --cities kinshasa beni --output data/geospatial/
"""

import argparse
import os
import sys

import geopandas as gpd
import osmnx as ox
import requests

# Importation de la base de données géo centrale
try:
    from geo_reference import PROVINCES
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from geo_reference import PROVINCES


def download_region_boundary(region_name, output_path):
    """
    Download administrative boundary for a province or city from OpenStreetMap.
    """
    osm_query = f"{region_name}, Democratic Republic of the Congo"
    safe_name = region_name.lower().replace(" ", "_")
    filepath = os.path.join(output_path, f"{safe_name}_boundary.geojson")

    if os.path.exists(filepath):
        print(f"⏭️  {filepath} exists, skipping.")
        return filepath

    print(f"📥 Downloading boundary for {region_name}...")
    try:
        gdf = ox.geocode_to_gdf(osm_query)
        gdf.to_file(filepath, driver="GeoJSON")
        print(f"✅ Saved: {filepath}")
        return filepath
    except Exception as exc:
        print(f"❌ Error for {region_name}: {exc}", file=sys.stderr)
        return None


def download_region_features(region_name, output_path, feature_type="buildings"):
    """
    Download features (buildings or water) from OpenStreetMap for a region.
    """
    osm_query = f"{region_name}, Democratic Republic of the Congo"
    safe_name = region_name.lower().replace(" ", "_")
    filepath = os.path.join(output_path, f"{safe_name}_{feature_type}.geojson")

    if os.path.exists(filepath):
        print(f"⏭️  {filepath} exists, skipping.")
        return filepath

    print(f"🏗️  Downloading {feature_type} for {region_name}...")
    try:
        if feature_type == "buildings":
            tags = {"building": True}
        else:
            tags = {"natural": "water", "waterway": True}

        gdf = ox.features_from_place(osm_query, tags=tags)
        if not gdf.empty:
            gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon", "Point", "LineString"])]
            gdf.to_file(filepath, driver="GeoJSON")
            print(f"✅ Saved: {filepath} ({len(gdf)} features)")
            return filepath
        else:
            print(f"⚠️  No {feature_type} found for {region_name}.")
            return None
    except Exception as exc:
        print(f"❌ Error for {region_name}: {exc}", file=sys.stderr)
        return None


def download_all_geodata(regions, output_path):
    """
    Download all geospatial data for a list of regions.
    """
    os.makedirs(output_path, exist_ok=True)
    results = {}

    target_regions = []
    if "all" in [r.lower() for r in regions]:
        target_regions = list(PROVINCES.keys())
    else:
        target_regions = regions

    for region in target_regions:
        results[region] = {
            "boundary": download_region_boundary(region, output_path),
            "buildings": download_region_features(region, output_path, "buildings"),
            "water": download_region_features(region, output_path, "water"),
        }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download geospatial data for DRC regions from OSM."
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        required=True,
        help="Provinces or cities to download (use 'all' for all 26 provinces)",
    )
    parser.add_argument(
        "--output", type=str, default="data/geospatial/", help="Output directory"
    )
    args = parser.parse_args()

    results = download_all_geodata(args.regions, args.output)
    print(f"\n🏁 Done. Data downloaded for {len(results)} region(s).")
