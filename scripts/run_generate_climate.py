#!/usr/bin/env python3

import os
import sys
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
sys.path.insert(0, '/app')

django.setup()

from scripts.generate_province_climate import generate_climate_for_all_provinces

result = generate_climate_for_all_provinces()
print(f"\n✓ {len(result)} provinces générées avec données climatiques")
for prov, df in list(result.items())[:3]:
    print(f"  - {prov}: {len(df)} mois | Temp: {df['temperature'].mean():.1f}K | Precip: {df['precipitation'].mean():.4f}m")

print(f"\n✅ Prêt pour l'entraînement!")
