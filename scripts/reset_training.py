#!/usr/bin/env python
"""
Script de reset complet de l'entraînement.
Supprime tous les modèles entraînés, prédictions, et tâches d'entraînement.
"""
import os
import sys
import shutil
from pathlib import Path

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
sys.path.append('/app')
import django
django.setup()

from django.conf import settings
from analytics.models import Prediction, TaskRun

print("=" * 60)
print("🗑️  RESET COMPLET DE L'ENTRAÎNEMENT")
print("=" * 60)

# 1. Supprimer les fichiers du dossier models
models_dir = Path(settings.BASE_DIR) / 'models'
if models_dir.exists():
    print(f"\n📁 Suppression du contenu de: {models_dir}")
    count = 0
    for file_path in models_dir.glob('*'):
        try:
            if file_path.is_file():
                file_path.unlink()
                count += 1
            elif file_path.is_dir():
                shutil.rmtree(file_path)
                count += 1
        except Exception as e:
            print(f"⚠️  Erreur suppression {file_path.name}: {e}")
    print(f"✅ {count} fichiers/dossiers supprimés dans models/")
else:
    print(f"❌ Dossier models non trouvé: {models_dir}")

# 2. Supprimer toutes les prédictions
print("\n🗄️  Suppression des prédictions en base de données...")
deleted_predictions, details = Prediction.objects.all().delete()
print(f"✅ {deleted_predictions} prédictions supprimées")
if details:
    print(f"   Détails: {details}")

# 3. Supprimer toutes les tâches d'entraînement
print("\n📋 Suppression des tâches d'entraînement...")
deleted_tasks, details = TaskRun.objects.all().delete()
print(f"✅ {deleted_tasks} tâches supprimées")
if details:
    print(f"   Détails: {details}")

print("\n" + "=" * 60)
print("🎉 Reset terminé! L'application est prête pour un nouvel entraînement.")
print("   Lancez: python scripts/train_models.py")
print("=" * 60)
