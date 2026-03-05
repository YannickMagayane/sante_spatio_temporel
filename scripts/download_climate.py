import argparse
import os
import sys

import cdsapi

PROVINCES = {
    "Kinshasa": {"lat": -4.3250, "lon": 15.3222},
    "Nord-Kivu": {"lat": -1.6585, "lon": 29.2200},
    "Sud-Kivu": {"lat": -2.5000, "lon": 28.8667},
    "Equateur": {"lat": 0.0487, "lon": 18.2612},
    "Haut-Katanga": {"lat": -11.6609, "lon": 27.4794},
}

VARIABLES = [
    "2m_temperature",
    "total_precipitation",
    "2m_dewpoint_temperature",
]


def build_area(lat, lon, radius=0.5):
    """
    Build a bounding box around a coordinate point.
    """
    return [lat + radius, lon - radius, lat - radius, lon + radius]


def download_climate_data(start_year, end_year, regions, output_path):
    """
    Download ERA5-Land monthly data for a list of regions over a date range.
    """
    os.makedirs(output_path, exist_ok=True)

    client = cdsapi.Client()
    created_files = []

    months = [f"{m:02d}" for m in range(1, 13)]

    # Résolution des régions (toutes ou spécifiques)
    target_regions = []
    if "all" in [r.lower() for r in regions]:
        target_regions = list(PROVINCES.keys())
    else:
        target_regions = regions

    for region in target_regions:
        region_key = region # Garder la casse originale pour la recherche
        # Recherche insensible à la casse
        found_key = next((k for k in PROVINCES.keys() if k.lower() == region.lower()), None)
        
        if not found_key:
            print(f"⚠️  Région inconnue '{region}'. Ignorée.")
            continue

        coords = PROVINCES[found_key]
        area = build_area(coords["lat"], coords["lon"])
        region_filename = found_key.lower().replace(" ", "_")

        for year in range(start_year, end_year + 1):
            filename = f"{region_filename}_{year}.nc"
            filepath = os.path.join(output_path, filename)

            if os.path.exists(filepath):
                print(f"⏭️  {filepath} exists, skipping.")
                continue

            print(f"📥 Downloading climate data for {found_key} ({year})...")

            try:
                client.retrieve(
                    "reanalysis-era5-land-monthly-means",
                    {
                        "product_type": "monthly_averaged_reanalysis",
                        "variable": VARIABLES,
                        "year": str(year),
                        "month": months,
                        "time": "00:00",
                        "area": area,
                        "format": "netcdf",
                    },
                    filepath,
                )
                created_files.append(filepath)
                print(f"✅ Saved: {filepath}")
            except Exception as exc:
                print(f"❌ Error downloading {found_key} {year}: {exc}", file=sys.stderr)

    return created_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download ERA5-Land monthly climate data for DRC provinces."
    )
    parser.add_argument("--start", type=int, required=True, help="Start year")
    parser.add_argument("--end", type=int, required=True, help="End year")
    parser.add_argument(
        "--regions",
        nargs="+",
        required=True,
        help="Provinces to download (use 'all' for all 26 provinces)",
    )
    parser.add_argument(
        "--output", type=str, default="data/climat/", help="Output directory"
    )
    args = parser.parse_args()

    files = download_climate_data(args.start, args.end, args.regions, args.output)
    print(f"\n🏁 Done. {len(files)} file(s) downloaded.")
