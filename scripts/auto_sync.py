#!/usr/bin/env python3
"""
Script d'ordonnancement pour le scraping et la mise à jour automatique.
Gère le premier scraping (2021-Présent) et les mises à jour quotidiennes.
"""

import subprocess
import os
import sys
from datetime import datetime

# Dossier racine du projet
PROJECT_ROOT = "/home/yannick-magayane/Bureau/projets/data_analyse/health"
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

def run_scraping(since_year=2021):
    """Lance le scraping pour les mots-clés de base sur toute la RDC."""
    keywords = ["paludisme", "choléra", "rougeole", "mpox", "ebola", "épidémie", "cas suspect"]
    
    print(f"--- Début du scraping (Historique depuis {since_year}) ---")
    
    # On itère sur les mots-clés principaux
    for kw in keywords:
        cmd = [
            "python3", os.path.join(SCRIPTS_DIR, "web_scraping.py"),
            "--platform", "all",
            "--keywords", kw,
            "--city", "RDC"
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Erreur lors du scraping de '{kw}': {e}")

def update_dashboard():
    """Régénère les données du dashboard si de nouvelles alertes sont présentes."""
    print("--- Mise à jour des données du dashboard ---")
    cmd = ["python3", os.path.join(SCRIPTS_DIR, "generate_dashboard_data.py")]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    # 1. Scraping
    run_scraping()
    
    # 2. Mise à jour Dashboard
    update_dashboard()
    
    print(f"--- Cycle terminé à {datetime.now()} ---")
