"""
Orchestrateur central du Pilier HealthDRC.
Coordonne le téléchargement, la synchronisation DB, et la préparation des modèles.
"""

import os
import subprocess
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_step(command, description):
    logger.info(f"▶️  {description}...")
    try:
        subprocess.run(command, check=True, shell=True)
        logger.info(f"✅ {description} terminé.")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erreur lors de : {description}")
        logger.error(e)

import time

def main():
    logger.info("📡 Démarrage du pipeline intelligent HealthDRC (Mode Continu 24h)")
    
    while True:
        logger.info(f"🔄 Début du cycle de synchronisation : {datetime.now()}")
        
        # 1. Synchronisation Géospatiale HDX
        run_step("python scripts/auto_sync.py", "Synchronisation HDX (Zones de Santé)")
        
        # 2. Peuplement de la base de données Django (ADM 1, 2, 3)
        run_step("python scripts/seed_all_zones.py", "Peuplement exhaustif des 519 Zones de Santé")
        
        # 3. Répartition des données cliniques historiques
        run_step("python scripts/distribute_clinical.py", "Répartition spatiale des données sanitaires")

        # 4. Synchronisation Climatique
        current_year = datetime.now().year
        run_step(f"python scripts/download_climate.py --start 2018 --end {current_year} --regions all", "MAJ Données Climatiques")
        
        # 5. Entraînement et Prédictions IA (Deep Learning)
        run_step("python scripts/train_models.py", "Entraînement des modèles ConvLSTM/CNN/LSTM")
        
        logger.info("🏁 Cycle terminé. Prochaine mise à jour dans 24 heures.")
        
        # Sleep for 24 hours (86400 seconds)
        time.sleep(86400)

if __name__ == "__main__":
    main()
