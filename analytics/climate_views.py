"""
API endpoints for climate and provincial analysis
"""
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.db.models import Avg, Max, Min, StdDev, Count, Sum
from .models import Province, HealthZone, ClimateData, Prediction, ClinicalData
from datetime import datetime, timedelta
import numpy as np
import json

@require_http_methods(["GET"])
def province_climate_api(request):
    """Get climate statistics for a province"""
    province_name = request.GET.get('province')
    if not province_name:
        return JsonResponse({'error': 'province parameter required'}, status=400)

    try:
        from .models import Province
        province = Province.objects.get(name=province_name)
    except Province.DoesNotExist:
        return JsonResponse({'error': 'Province not found'}, status=404)

    # Get zones in this province
    zones = HealthZone.objects.filter(territory__province=province)
    zone_ids = zones.values_list('id', flat=True)

    # Get recent climate data (last 6 months)
    six_months_ago = datetime.now().replace(day=1) - timedelta(days=180)
    climate_qs = ClimateData.objects.filter(
        health_zone_id__in=zone_ids,
        month__gte=six_months_ago
    )

    if not climate_qs.exists():
        return JsonResponse({'error': 'No climate data for this province'}, status=404)

    # Aggregate by month across all zones
    monthly_climate = climate_qs.values('month').annotate(
        avg_temp=Avg('temperature'),
        avg_precip=Avg('precipitation'),
        avg_humidity=Avg('humidity'),
        zones_count=Count('health_zone', distinct=True)
    ).order_by('month')

    # Format response
    months = []
    temps = []
    precips = []
    humidities = []

    for item in monthly_climate:
        months.append(item['month'].strftime('%Y-%m'))
        temps.append(round(item['avg_temp'], 2))
        precips.append(round(item['avg_precip'], 2))
        humidities.append(round(item['avg_humidity'], 1))

    # Calculate correlations with disease data if available
    disease_corrs = {}
    for disease in ['Paludisme', 'Tuberculose', 'VIH/SIDA']:
        # Get monthly cases for this province
        clinical_qs = ClinicalData.objects.filter(
            health_zone_id__in=zone_ids,
            disease_name=disease,
            month__gte=six_months_ago
        ).values('month').annotate(total_cases=Sum('cases_count')).order_by('month')

        if clinical_qs.exists():
            cases_by_month = {item['month'].strftime('%Y-%m'): item['total_cases'] for item in clinical_qs}
            disease_months = [m for m in months if m in cases_by_month]
            disease_cases = [cases_by_month.get(m, 0) for m in months]

            if len(disease_cases) > 2:
                # Correlate with temperature
                temp_vals = [t for m, t in zip(months, temps) if m in disease_months]
                if len(temp_vals) == len(disease_cases):
                    corr_temp = np.corrcoef(disease_cases, temp_vals)[0, 1] if len(disease_cases) > 1 else 0
                    corr_precip = np.corrcoef(disease_cases, [p for m, p in zip(months, precips) if m in disease_months])[0, 1] if len(disease_cases) > 1 else 0
                else:
                    corr_temp = corr_precip = 0
            else:
                corr_temp = corr_precip = 0

            disease_corrs[disease] = {
                'temperature_correlation': round(corr_temp, 3) if not np.isnan(corr_temp) else 0,
                'precipitation_correlation': round(corr_precip, 3) if not np.isnan(corr_precip) else 0,
                'data_points': len(disease_cases)
            }

    return JsonResponse({
        'province': province_name,
        'period': {
            'start': six_months_ago.strftime('%Y-%m'),
            'end': datetime.now().strftime('%Y-%m')
        },
        'monthly_data': {
            'months': months,
            'temperature': temps,
            'precipitation': precips,
            'humidity': humidities,
            'zones_covered': zones.count()
        },
        'disease_correlations': disease_corrs,
        'climate_explanation': _generate_climate_explanation(temps, precips)
    })

def _generate_climate_explanation(temps, precips):
    """Generate human-readable interpretation of climate patterns"""
    if not temps or not precips:
        return "Données climatiques insuffisantes pour analyse."

    avg_temp = np.mean(temps[-6:])  # Last 6 months
    avg_precip = np.mean(precips[-6:])

    interpretations = []

    if avg_temp > 28:
        interpretations.append("Température élevée (>28°C) favorable aux vecteurs (moustiques)")
    elif avg_temp > 24:
        interpretations.append("Température modérée (24-28°C) stabilisant les cycles de transmission")
    else:
        interpretations.append("Température fraîche (<24°C) pouvant ralentir les vecteurs")

    if avg_precip > 200:
        interpretations.append("Précipitations abondantes créant des gîtes larvaires (moustiques)")
    elif avg_precip > 100:
        interpretations.append("Précipitations modérées soutenant l'hydrologie locale")
    else:
        interpretations.append("Faibles précipitations limitant les sites de reproduction")

    return ". ".join(interpretations) + "."

@require_http_methods(["GET"])
def zone_propagation_heatmap_api(request):
    """Get propagation rates by health zone for heatmap"""
    province_name = request.GET.get('province')
    disease = request.GET.get('disease', 'Paludisme')

    # Build query
    zones_query = HealthZone.objects.all()
    if province_name:
        try:
            province = Province.objects.get(name=province_name)
            zones_query = zones_query.filter(territory__province=province)
        except Province.DoesNotExist:
            pass

    # Get latest predictions for each zone
    data = []
    for zone in zones_query:
        # Get latest prediction for this zone/disease
        pred = Prediction.objects.filter(
            health_zone=zone,
            disease_name=disease
        ).order_by('-prediction_date').first()

        if pred:
            # Calculate propagation from influence_factors if available
            propagation = pred.propagation_rate
            reliability = pred.influence_factors.get('reliability', 50) if pred.influence_factors else 50
            r2 = pred.influence_factors.get('r2', 0) if pred.influence_factors else 0

            data.append({
                'zone_id': zone.id,
                'zone_name': zone.name,
                'province': zone.territory.province.name if zone.territory and zone.territory.province else 'Unknown',
                'latitude': zone.latitude,
                'longitude': zone.longitude,
                'predicted_cases': float(pred.predicted_cases),
                'propagation': float(propagation),
                'reliability': int(reliability),
                'r2': float(r2),
                'prediction_date': pred.prediction_date.strftime('%Y-%m')
            })

    # Calculate statistics
    if data:
        avg_prop = np.mean([d['propagation'] for d in data])
        max_prop = np.max([d['propagation'] for d in data])
        min_prop = np.min([d['propagation'] for d in data])
    else:
        avg_prop = max_prop = min_prop = 0

    return JsonResponse({
        'disease': disease,
        'zones': data,
        'stats': {
            'total_zones': len(data),
            'avg_propagation': round(avg_prop, 2),
            'max_propagation': round(max_prop, 2),
            'min_propagation': round(min_prop, 2)
        }
    })

@require_http_methods(["GET"])
def provincial_predictions_api(request):
    """Get predictions aggregated by province"""
    province_name = request.GET.get('province')
    disease = request.GET.get('disease')

    # If no province, get all provinces
    if not province_name:
        provinces = Province.objects.all()
        result = {}
        for prov in provinces:
            result[prov.name] = _get_province_predictions(prov, disease)
        return JsonResponse(result)
    else:
        province = get_object_or_404(Province, name=province_name)
        data = _get_province_predictions(province, disease)
        return JsonResponse(data)

def _get_province_predictions(province, disease_filter=None):
    """Helper: get predictions aggregated by province"""
    zones = HealthZone.objects.filter(territory__province=province)

    # Get latest predictions (next 3 months)
    from datetime import datetime, timedelta
    from dateutil.relativedelta import relativedelta

    now = datetime.now().replace(day=1)
    next_months = [now + relativedelta(months=i) for i in range(1, 4)]

    result = {
        'province': province.name,
        'predictions': []
    }

    for month in next_months:
        month_data = {
            'month': month.strftime('%Y-%m'),
            'diseases': {}
        }

        # Get predictions for this month
        preds = Prediction.objects.filter(
            health_zone__in=zones,
            prediction_date__year=month.year,
            prediction_date__month=month.month
        )

        if disease_filter:
            preds = preds.filter(disease_name=disease_filter)

        # Aggregate by disease
        for disease in ['Paludisme', 'Tuberculose', 'VIH/SIDA', 'Choléra']:
            disease_preds = preds.filter(disease_name=disease)
            if disease_preds.exists():
                total_cases = sum(p.predicted_cases for p in disease_preds)
                avg_prop = np.mean([p.propagation_rate for p in disease_preds])
                zone_count = disease_preds.values('health_zone').distinct().count()

                month_data['diseases'][disease] = {
                    'total_cases': round(total_cases, 0),
                    'avg_propagation': round(avg_prop, 2),
                    'zones_affected': zone_count,
                    'predictions_count': disease_preds.count()
                }

        result['predictions'].append(month_data)

    return result

@require_http_methods(["GET"])
def disease_propagation_map_api(request):
    """Get map data with propagation rates for all zones"""
    disease = request.GET.get('disease', 'Paludisme')
    province = request.GET.get('province')

    zones = HealthZone.objects.filter(
        prediction__disease_name=disease
    ).distinct()

    if province:
        zones = zones.filter(territory__province__name=province)

    result = []
    for zone in zones:
        # Latest prediction
        pred = Prediction.objects.filter(
            health_zone=zone,
            disease_name=disease
        ).order_by('-prediction_date').first()

        if pred and zone.latitude and zone.longitude:
            result.append({
                'zone_name': zone.name,
                'province': zone.territory.province.name if zone.territory else 'N/A',
                'latitude': zone.latitude,
                'longitude': zone.longitude,
                'predicted_cases': float(pred.predicted_cases),
                'propagation': float(pred.propagation_rate),
                'reliability': pred.influence_factors.get('reliability', 50) if pred.influence_factors else 50,
                'r2': pred.influence_factors.get('r2', 0) if pred.influence_factors else 0
            })

    return JsonResponse({
        'disease': disease,
        'zones': result,
        'count': len(result)
    })
