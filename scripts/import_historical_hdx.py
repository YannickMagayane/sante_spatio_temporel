import os
import csv
import django
import logging
from datetime import datetime, date

# Configuration Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from analytics.models import ClinicalData, HealthZone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def import_historical_data(raw_dir="data/raw_hdx", start_year=2017):
    """
    Importe les vraies données historiques de santé à partir des fichiers HDX (WHO, etc.)
    téléchargés dans le dossier raw_hdx. On extrait les cas depuis les années indiquées.
    """
    logger.info(f"Début de l'importation historique (depuis {start_year})...")
    
    if not os.path.exists(raw_dir):
        logger.error(f"Le dossier {raw_dir} n'existe pas.")
        return

    # Par défaut on attribue à 'Kinshasa' si on a une donnée nationale non localisée.
    default_zone = HealthZone.objects.filter(name__icontains='Kinshasa').first()
    if not default_zone:
        # Création d'une zone par défaut si aucune n'existe (anomalie normalement)
        default_zone, _ = HealthZone.objects.get_or_create(
            code="N/A",
            defaults={"name": "Donnée Nationale (RDC)", "province": "National"}
        )

    count_inserted = 0
    
    for filename in os.listdir(raw_dir):
        if not filename.endswith('.csv'):
            continue
            
        filepath = os.path.join(raw_dir, filename)
        logger.info(f"Analyse de {filename}...")
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                
                # Une détection heuristique simple pour les fichiers OMS / HDX
                year_column = None
                value_column = None
                disease_column = None
                
                for col in reader.fieldnames:
                    col_lower = col.lower()
                    if 'year' in col_lower or 'period' in col_lower or 'date' in col_lower:
                        year_column = col
                    if 'value' in col_lower or 'cases' in col_lower or 'numeric' in col_lower:
                        value_column = col
                    if 'indicator' in col_lower or 'disease' in col_lower:
                        disease_column = col
                        
                if not year_column or not value_column:
                    # Sauter les fichiers de métadonnées géographiques
                    continue
                    
                logger.info(f"Structure reconnue: Année={year_column}, Valeur={value_column}, Maladie={disease_column}")
                
                for row in reader:
                    raw_year = row.get(year_column, '')
                    
                    try:
                        # Extraire l'année
                        year = int(raw_year[:4])
                    except (ValueError, TypeError):
                        continue
                        
                    if year < start_year:
                        continue
                        
                    raw_value = row.get(value_column, '')
                    try:
                        cases = int(float(raw_value.replace(' ', '').replace(',', '')))
                    except (ValueError, TypeError):
                        continue
                        
                    # Extraire le nom de la maladie (très varié dans les fichiers OMS)
                    disease = "Général"
                    if disease_column:
                        disease_raw = str(row.get(disease_column, '')).lower()
                        if 'malaria' in disease_raw or 'paludisme' in disease_raw:
                            disease = "Malaria"
                        elif 'cholera' in disease_raw:
                            disease = "Choléra"
                        elif 'ebola' in disease_raw:
                            disease = "Ebola"
                        elif 'measles' in disease_raw or 'rougeole' in disease_raw:
                            disease = "Rougeole"
                        elif 'tuberculosis' in disease_raw or 'tb' in disease_raw.split():
                            disease = "Tuberculose"
                            
                    if disease == "Général":
                        # Exclure les données qui ne sont pas des maladies infectieuses clés
                        continue
                        
                    # Sauvegarde en BD (on crée une entrée au 1er janvier de l'année pour la donnée annuelle)
                    d_date = date(year, 1, 1)
                    
                    ClinicalData.objects.update_or_create(
                        health_zone=default_zone,
                        disease_name=disease,
                        month=d_date,
                        defaults={
                            'cases_count': cases,
                            'deaths_count': 0  # Peut être mappé plus spécifiquement si la colonne "deaths" existe
                        }
                    )
                    count_inserted += 1
                    
                    # On évite d'exploser la DB si c'est énorme (juste pour l'historique de base)
                    if count_inserted % 1000 == 0:
                        logger.info(f"{count_inserted} lignes insérées...")
                        
        except Exception as e:
            logger.error(f"Erreur avec le fichier {filename}: {str(e)}")
            
    logger.info(f"✅ Importation depuis HDX terminée ! Total insérés : {count_inserted}")

if __name__ == "__main__":
    import_historical_data()
