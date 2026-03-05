import os
import django
import sys
from django.db.models import Count, Sum

# Setup Django if not already
if not os.environ.get('DJANGO_SETTINGS_MODULE'):
    sys.path.append('/app')
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
    django.setup()

from analytics.models import Province, Territory, HealthZone, ClinicalData, Prediction

def audit():
    print("--- 🌍 AUDIT GÉOGRAPHIQUE ---")
    provinces = Province.objects.all()
    territories = Territory.objects.all()
    zones = HealthZone.objects.all()

    print(f"Provinces: {provinces.count()}")
    print(f"Territoires: {territories.count()}")
    print(f"Zones de Santé: {zones.count()}")

    # Check for empty levels
    empty_prov = provinces.annotate(t_count=Count('territories')).filter(t_count=0)
    if empty_prov.exists():
        print(f"🚨 Provinces sans territoires: {[p.name for p in empty_prov]}")
    else:
        print("✅ Toutes les provinces ont des territoires.")

    empty_terr = territories.annotate(z_count=Count('health_zones')).filter(z_count=0)
    if empty_terr.exists():
        print(f"⚠️ Territoires sans zones de santé: {empty_terr.count()}")
    else:
        print("✅ Tous les territoires ont des zones de santé.")

    # Check for coordinates
    zones_no_coords = zones.filter(latitude__isnull=True) | zones.filter(longitude__isnull=True)
    if zones_no_coords.count() > 0:
        print(f"🚨 Zones de santé sans coordonnées: {zones_no_coords.count()}")
    else:
        print("✅ Toutes les zones de santé ont des coordonnées.")

    print("\n--- 🦠 AUDIT CLINIQUE & PRÉDICTIF ---")
    diseases = ClinicalData.objects.values('disease_name').annotate(count=Count('id'), total_cases=Sum('cases_count')).order_by('-total_cases')
    print("Maladies enregistrées (Données Cliniques):")
    for d in diseases:
        print(f" - {d['disease_name']}: {d['count']} entrées, {d['total_cases']} cas au total")

    preds = Prediction.objects.values('disease_name').annotate(count=Count('id'), total_cases=Sum('predicted_cases')).order_by('-total_cases')
    print("\nPrédictions générées:")
    for p in preds:
        print(f" - {p['disease_name']}: {p['count']} prédictions, {p['total_cases']} cas prévus")

    clinical_range = ClinicalData.objects.aggregate(start=Count('month', distinct=True))
    print(f"\nNombre de mois distincts (Clinique): {clinical_range['start']}")

    neg_cases = ClinicalData.objects.filter(cases_count__lt=0)
    if neg_cases.exists():
        print(f"🚨 Données négatives détectées: {neg_cases.count()} records")
    else:
        print("✅ Pas de valeurs négatives dans les cas cliniques.")

if __name__ == "__main__":
    audit()
