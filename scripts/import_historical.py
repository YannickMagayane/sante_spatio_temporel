"""
Script d'importation des données historiques (2018-2025).
Charge les fichiers CSV fournis par le PNLP/INRB dans le modèle ClinicalData.
"""

import os
import django
import pandas as pd
from datetime import datetime

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from analytics.models import HealthZone, ClinicalData

def import_csv(file_path):
    if not os.path.exists(file_path):
        print(f"Fichier non trouvé : {file_path}")
        return

    df = pd.DataFrame()
    # On suppose un format : zone_code, maladie, cas, deces, date (YYYY-MM)
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Erreur de lecture : {e}")
        return

    for _, row in df.iterrows():
        try:
            zs = HealthZone.objects.get(code=row['zone_code'])
            ClinicalData.objects.update_or_create(
                health_zone=zs,
                disease_name=row['maladie'],
                month=datetime.strptime(row['date'], "%Y-%m").date(),
                defaults={
                    'cases_count': row['cas'],
                    'deaths_count': row['deces']
                }
            )
        except HealthZone.DoesNotExist:
            print(f"Zone inconnue : {row['zone_code']}")
        except Exception as e:
            print(f"Erreur sur la ligne : {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        import_csv(sys.argv[1])
    else:
        print("Usage: python scripts/import_historical.py path/to/file.csv")
