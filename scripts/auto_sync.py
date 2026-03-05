"""
Auto-sync script for HealthDRC.
Sychronizes geospatial, health, and climate data from HDX and Copernicus.
Checks for updates before downloading to save bandwidth and resources.
"""

import os
import time
import json
import logging
from datetime import datetime
from hdx.api.configuration import Configuration
from hdx.data.dataset import Dataset

# Configuration logic
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of critical HDX datasets for DRC
DATASETS = {
    "zones_sante": "cod-ab-health_zones",
    "boundaries": "cod-administrative-boundaries",
    "hydrography": "cod-rdc-hydrographie",
    "population": "cod-population-density-maps",
    "health_facilities": "cod-health-facilities",
}

def setup_hdx():
    """Initialize HDX configuration."""
    try:
        # Use a more descriptive user agent
        Configuration.create(hdx_site="prod", user_agent="HealthDRC-Pilier-Sante", hdx_read_only=True)
        return True
    except Exception as e:
        logger.error(f"Failed to initialize HDX: {e}")
        return False

def check_and_download_dataset(dataset_id, output_dir):
    """
    Check if an HDX dataset has been updated and download it if necessary.
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata_file = os.path.join(output_dir, "sync_metadata.json")
    
    sync_metadata = {}
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            sync_metadata = json.load(f)
            
    try:
        dataset = Dataset.read_from_hdx(dataset_id)
        if not dataset:
            logger.warning(f"Dataset {dataset_id} not found on HDX.")
            return False
            
        last_modified = dataset.get('last_modified')
        
        if dataset_id in sync_metadata and sync_metadata[dataset_id] == last_modified:
            logger.info(f"⏭️  {dataset_id} est à jour ({last_modified}).")
            return False
            
        logger.info(f"📥 Mise à jour détectée pour {dataset_id}...")
        
        resources = dataset.get_resources()
        for resource in resources:
            fmt = resource.get('format', '').lower()
            # On privilégie GeoJSON et CSV pour le Pilier
            if any(ext in fmt for ext in ['geojson', 'csv']):
                url, path = resource.download(output_dir)
                logger.info(f"✅ Téléchargé : {path}")
                
        sync_metadata[dataset_id] = last_modified
        with open(metadata_file, 'w') as f:
            json.dump(sync_metadata, f, indent=4)
            
        return True
    except Exception as e:
        logger.error(f"Erreur synchro {dataset_id}: {e}")
        return False

import subprocess
import hashlib

# Folders to monitor for manual updates (XLSX, CSV, GeoJSON)
MONITOR_DIRS = ["data/sanitaires", "data/raw_hdx", "data/geospatial"]

def get_dir_hash(directory):
    """Calcule un hash unique basé sur l'état des fichiers dans un dossier."""
    hasher = hashlib.md5()
    if not os.path.exists(directory):
        return ""
    for root, dirs, files in os.walk(directory):
        for names in sorted(files):
            filepath = os.path.join(root, names)
            hasher.update(names.encode()) # Hash filename
            try:
                # Hash mtime for speed instead of full content
                mtime = os.path.getmtime(filepath)
                hasher.update(str(mtime).encode())
            except OSError:
                pass
    return hasher.hexdigest()

def trigger_pipeline():
    """Déclenche l'ingestion et le ré-entrainement immédiat du réseau."""
    logger.info("🔥 MISE À JOUR DÉTECTÉE : Lancement de l'ingestion précise...")
    try:
        # 1. Ingestion/Preprocess
        logger.info("⚙️  Etape 1 : Ingestion des données dans la base...")
        subprocess.run(["python", "scripts/preprocess.py"], check=True)
        
        # 2. Retraining
        logger.info("🧠 Etape 2 : Ré-entrainement direct du NeuralCore...")
        subprocess.run(["python", "scripts/train_models.py"], check=True)
        
        logger.info("✅ Pipeline complété : Les prédictions sont à jour.")
    except Exception as e:
        logger.error(f"❌ Erreur lors du déclenchement du pipeline : {e}")

def run_sync():
    """Main sync loop covering 2018-2026 context."""
    logger.info("🚀 Lancement de la synchronisation automatique (Cycle 2018-2026)...")
    
    # 0. Check for local file changes since last run
    state_file = "data/sync_state.json"
    last_hashes = {}
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            last_hashes = json.load(f)
            
    current_hashes = {d: get_dir_hash(d) for d in MONITOR_DIRS}
    local_changed = any(current_hashes[d] != last_hashes.get(d) for d in MONITOR_DIRS)

    if not setup_hdx():
        logger.warning("HDX API inaccessible, focus sur le monitoring local...")
    else:
        for name, ds_id in DATASETS.items():
            if check_and_download_dataset(ds_id, f"data/geospatial/{name}"):
                local_changed = True
    
    if local_changed:
        trigger_pipeline()
        # Save new state
        with open(state_file, 'w') as f:
            json.dump(current_hashes, f)
    else:
        logger.info("💤 Aucun changement détecté. Système au repos.")
        
    logger.info("🏁 Synchronisation terminée.")

if __name__ == "__main__":
    run_sync()
