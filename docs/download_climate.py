#!/usr/bin/env python
"""
Script standalone pour télécharger les données climatiques par province.
Usage: python download_climate.py
"""

import os
import sys
import django

# Setup Django
sys.path.insert(0, '/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from scripts.climate_data_fetcher import fetch_climate_for_all_provinces

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌍 TÉLÉCHARGEMENT DONNÉES CLIMATIQUES PAR PROVINCE")
    print("="*70 + "\n")
    
    try:
        result = fetch_climate_for_all_provinces()
        
        print(f"\n{'─'*70}")
        print(f"✓ SUCCÈS: {len(result)} provinces chargées\n")
        
        # Afficher un résumé
        for prov_name in sorted(result.keys())[:10]:
            df = result[prov_name]
            print(f"  ✓ {prov_name:25} → {len(df):2d} mois | T: {df['temperature'].mean():.1f}K")
        
        if len(result) > 10:
            print(f"  ... et {len(result) - 10} autres provinces")
            
    except Exception as e:
        print(f"\n✗ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
