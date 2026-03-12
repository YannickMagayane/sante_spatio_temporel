#!/usr/bin/env python3
"""Test minimal du pipeline d'entraînement avec données climatiques par province."""

import os
import sys
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
sys.path.insert(0, '/app')
django.setup()

from scripts.train_models import train_and_predict

print("\n" + "="*70)
print("🧠 TEST ENTRAÎNEMENT AVEC DONNÉES CLIMATIQUES PAR PROVINCE")
print("="*70 + "\n")

try:
    train_and_predict()
    print("\n✅ Entraînement complété avec succès!")
except Exception as e:
    print(f"\n❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
