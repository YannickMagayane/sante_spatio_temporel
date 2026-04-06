#!/usr/bin/env python
"""Debug script to check why kernel reliability is low"""

import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from analytics.models import Prediction
from collections import defaultdict

# Check predictions
all_preds = Prediction.objects.all()
print(f"\n📊 Total Predictions: {all_preds.count()}")

if all_preds.count() == 0:
    print("❌ NO PREDICTIONS FOUND - Run training first!")
    exit(1)

# Analyze metrics
reliabilities = []
r2_values = []
spread_rates = []
model_stats = defaultdict(list)

for p in all_preds:
    inf = p.influence_factors or {}
    rel = inf.get('reliability', 0)
    r2 = inf.get('r2', 0)
    spread = inf.get('weekly_spread_rate', 0)
    model = inf.get('model_used', 'Unknown')
    
    reliabilities.append(rel)
    r2_values.append(r2)
    spread_rates.append(spread)
    model_stats[model].append({'rel': rel, 'r2': r2})

print(f"\n🎯 Fiabilité Noyau:")
print(f"  Average: {sum(reliabilities)/len(reliabilities):.1f}%")
print(f"  Min: {min(reliabilities):.1f}% | Max: {max(reliabilities):.1f}%")

print(f"\n📈 Précision IA (R²):")
print(f"  Average: {sum(r2_values)*100/len(r2_values):.1f}%")
print(f"  Min: {min(r2_values)*100:.1f}% | Max: {max(r2_values)*100:.1f}%")

print(f"\n🦠 Spread Rates:")
print(f"  Average: {sum(spread_rates)/len(spread_rates):.1f}%")

print(f"\n🔧 By Model:")
for model, stats in sorted(model_stats.items()):
    avg_rel = sum(s['rel'] for s in stats) / len(stats)
    avg_r2 = sum(s['r2'] for s in stats) / len(stats) * 100
    print(f"  {model}: {len(stats)} preds | Rel={avg_rel:.1f}% | R²={avg_r2:.1f}%")

# Diagnosis
print(f"\n⚠️ DIAGNOSIS:")
if sum(reliabilities)/len(reliabilities) < 60:
    print("  🔴 Kernel Reliability TOO LOW (<60%)")
    print("     → Models are unreliable, using Persistence fallback")
if sum(r2_values)*100/len(r2_values) < 30:
    print("  🔴 AI Precision TOO LOW (<30%)")
    print("     → Models not learning well, check data quality")
