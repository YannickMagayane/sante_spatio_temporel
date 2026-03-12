#!/usr/bin/env python
"""Script pour parser les données climatiques GRIB téléchargées."""

import logging
import os
import sys

sys.path.insert(0, '/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from scripts.climate_data_fetcher import _parse_rdc_grib

# Parser le fichier GRIB téléchargé
print("\n" + "="*70)
print("PARSING DU FICHIER GRIB RDC")
print("="*70 + "\n")

grib_file = "data/climat/rdc_era5_monthly.grib"
output_dir = "data/climat"

if os.path.exists(grib_file):
    print(f"📖 Fichier trouvé: {grib_file}")
    print(f"   Taille: {os.path.getsize(grib_file) / 1e6:.1f} MB\n")
    
    df = _parse_rdc_grib(grib_file, output_dir)
    
    if not df.empty:
        print(f"\n✓ Parsing réussi!")
        print(f"\nDonnées: {len(df)} mois ({df['date'].min()} à {df['date'].max()})\n")
        print(df.head(10))
        print("\nStatistiques globales:")
        print(df.describe())
        
        # Vérifier les fichiers sauvegardés
        print("\n" + "="*70)
        print("FICHIERS SAUVEGARDÉS:")
        csv_file = os.path.join(output_dir, 'rdc_climate_monthly.csv')
        pkl_file = os.path.join(output_dir, 'rdc_climate_monthly.pkl')
        
        if os.path.exists(csv_file):
            print(f"✓ CSV: {csv_file} ({os.path.getsize(csv_file)} octets)")
        if os.path.exists(pkl_file):
            print(f"✓ Pickle: {pkl_file} ({os.path.getsize(pkl_file)} octets)")
    else:
        print("\n✗ Parsing échoué (DataFrame vide)")
        sys.exit(1)
else:
    print(f"✗ Fichier non trouvé: {grib_file}")
    sys.exit(1)
