#!/usr/bin/env python
"""Debug script to test chart generation"""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from analytics.models import ClinicalData, Prediction, ClimateData, Province
from analytics.smart_graphs import SmartGraphGenerator, remove_empty_periods
from django.db.models import Sum, Avg
import pandas as pd

# Get Haut-Katanga Paludisme data
province = Province.objects.filter(name='Haut-Katanga').first()
print(f"Province: {province}\n")

# Get clinical data
clinical = ClinicalData.objects.filter(
    disease_name='Paludisme',
    health_zone__territory__province=province
).values('month').annotate(cases_count=Sum('cases_count')).order_by('month')

df = pd.DataFrame(list(clinical))
print(f"Original clinical data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(df.head(10))
print()

# Convert month column
df['month'] = pd.to_datetime(df['month'])
print(f"After datetime conversion:")
print(df.head(10))
print()

# Test remove_empty_periods
df_clean = remove_empty_periods(df)
print(f"After remove_empty_periods: {df_clean.shape}")
print(df_clean)
print()

# Test climate data
climate = ClimateData.objects.filter(
    health_zone__territory__province=province
).values('month').annotate(
    temperature=Avg('temperature'),
    precipitation=Avg('precipitation'),
    humidity=Avg('humidity')
).order_by('month')

climate_df = pd.DataFrame(list(climate))
print(f"Climate data shape: {climate_df.shape}")
print(f"Climate columns: {climate_df.columns.tolist()}")
print(f"Climate first dates:\n{climate_df.head()}")
print(f"Climate last dates:\n{climate_df.tail()}")

# Test forecast
pred = Prediction.objects.filter(
    disease_name='Paludisme',
    health_zone__territory__province=province
).order_by('-prediction_date').first()

if pred:
    forecast_data = {
        'predicted_cases': pred.predicted_cases,
        'prediction_date': pred.prediction_date,
        'reliability': 0.85,
        'health_zone': pred.health_zone.name if pred.health_zone else 'National'
    }
    print(f"\nForecast data: {forecast_data}")

# Try generating a chart
print("\n" + "="*60)
print("Testing chart generation...")
print("="*60)

try:
    gen = SmartGraphGenerator()
    if not df_clean.empty and forecast_data:
        print("Generating monthly forecast chart...")
        chart_b64 = gen.monthly_forecast_chart(df_clean, forecast_data, 'Paludisme')
        print(f"✓ Chart generated, base64 length: {len(chart_b64)} chars")
        print(f"✓ First 100 chars: {chart_b64[:100]}")
except Exception as e:
    print(f"✗ Error generating chart: {e}")
    import traceback
    traceback.print_exc()
