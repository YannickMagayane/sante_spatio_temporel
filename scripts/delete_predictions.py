#!/usr/bin/env python3
import os
import sys
import django

app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, app_dir)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from analytics.models import Prediction

print("🗑️  Suppression des prédictions précédentes...")
deleted_count, stats = Prediction.objects.all().delete()
print(f"✅ Supprimé : {deleted_count} prédictions")
print(f"   Détails : {stats}")

# Vérification
remaining = Prediction.objects.count()
print(f"✏️  Prédictions restantes : {remaining}")
