"""
API pour données provinciales destinées au dashboard
"""

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.db.models import Avg, Sum, Count, Max, Min
from .models import Province, HealthZone, Prediction, ClimateData, ClinicalData
from datetime import datetime, timedelta
import json

@require_http_methods(["GET"])
def province_dashboard_api(request):
    """Retourne les données agrégées par province pour le dashboard"""
    province_name = request.GET.get('province')

    if province_name:
        try:
            province = Province.objects.get(name=province_name)
        except Province.DoesNotExist:
            return JsonResponse({'error': 'Province not found'}, status=404)
        provinces = [province]
    else:
        provinces = Province.objects.all()

    result = {}

    for prov in provinces:
        zones = HealthZone.objects.filter(territory__province=prov)
        zone_count = zones.count()

        # Prédictions récentes (toutes maladies confondues)
        recent_preds = Prediction.objects.filter(
            health_zone__in=zones,
            prediction_date__gte=datetime.now().replace(day=1)
        )

        # Agrégations
        if recent_preds.exists():
            total_cases = recent_preds.aggregate(total=Sum('predicted_cases'))['total'] or 0
            avg_prop = recent_preds.aggregate(avg=Avg('propagation_rate'))['avg'] or 0
            avg_reliability = recent_preds.aggregate(avg=Avg('influence_factors__reliability'))['avg'] or 50
        else:
            total_cases = avg_prop = avg_reliability = 0

        # Données climatiques récentes
        six_months_ago = datetime.now().replace(day=1) - timedelta(days=180)
        climate_data = ClimateData.objects.filter(
            health_zone__in=zones,
            month__gte=six_months_ago
        )

        if climate_data.exists():
            climate_stats = climate_data.aggregate(
                avg_temp=Avg('temperature'),
                avg_precip=Avg('precipitation'),
                avg_humidity=Avg('humidity')
            )
        else:
            climate_stats = {'avg_temp': None, 'avg_precip': None, 'avg_humidity': None}

        # Données cliniques historiques (pour graphique temporel)
        clinical_by_month = ClinicalData.objects.filter(
            health_zone__in=zones,
            month__gte=six_months_ago
        ).values('month').annotate(total_cases=Sum('cases_count')).order_by('month')

        months_list = []
        cases_list = []
        for item in clinical_by_month:
            months_list.append(item['month'].strftime('%Y-%m'))
            cases_list.append(item['total_cases'])

        # Top 5 zones par nombre de cas prédits
        top_zones_qs = recent_preds.values('health_zone__name').annotate(
            total_cases=Sum('predicted_cases')
        ).order_by('-total_cases')[:5]

        top_zones = [
            {
                'zone_name': item['health_zone__name'],
                'predicted_cases': round(item['total_cases'], 0)
            }
            for item in top_zones_qs
        ]

        result[prov.name] = {
            'zone_count': zone_count,
            'predictions': {
                'total_cases': round(total_cases, 0),
                'avg_propagation': round(avg_prop, 2),
                'avg_reliability': round(avg_reliability, 1),
                'count': recent_preds.count()
            },
            'climate': {
                'avg_temperature': round(climate_stats['avg_temp'], 2) if climate_stats['avg_temp'] else None,
                'avg_precipitation': round(climate_stats['avg_precip'], 2) if climate_stats['avg_precip'] else None,
                'avg_humidity': round(climate_stats['avg_humidity'], 1) if climate_stats['avg_humidity'] else None,
                'months_available': climate_data.values('month').distinct().count()
            },
            'historical_trend': {
                'months': months_list[-12:],  # Derniers 12 mois
                'cases': cases_list[-12:]
            },
            'top_zones': top_zones,
            'risk_level': 'high' if avg_prop > 5 else 'moderate' if avg_prop > 2 else 'low'
        }

    return JsonResponse(result)

@require_http_methods(["GET"])
def zone_list_api(request):
    """Retourne les zones d'une province"""
    province_name = request.GET.get('province')
    if not province_name:
        return JsonResponse({'error': 'province required'}, status=400)

    try:
        province = Province.objects.get(name=province_name)
    except Province.DoesNotExist:
        return JsonResponse({'error': 'Province not found'}, status=404)

    zones = HealthZone.objects.filter(territory__province=province).order_by('name')

    data = [{
        'id': zone.id,
        'name': zone.name,
        'code': zone.code,
        'territory': zone.territory.name if zone.territory else None,
        'population': zone.population,
        'latitude': zone.latitude,
        'longitude': zone.longitude
    } for zone in zones]

    return JsonResponse({
        'province': province_name,
        'zones': data,
        'count': len(data)
    })
