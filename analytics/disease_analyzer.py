"""
Disease Analysis Engine - Generates high-quality interactive visualizations
"""
import json
from django.db.models import Sum, Count, Avg, F, Q
from .models import ClinicalData, Prediction, ClimateData, HealthZone
from scipy.stats import pearsonr
import numpy as np


class DiseaseAnalyzer:
    """Comprehensive disease analysis with climate correlation"""
    
    @staticmethod
    def get_timeline_data(disease_name, province_name=None):
        """Get monthly timeline data (no empty years)"""
        if province_name:
            qs = ClinicalData.objects.filter(
                disease_name=disease_name,
                health_zone__territory__province__name=province_name
            )
        else:
            qs = ClinicalData.objects.filter(disease_name=disease_name)
        
        data = qs.values('month').annotate(
            cases=Sum('cases_count'),
            deaths=Sum('deaths_count'),
            zones=Count('health_zone', distinct=True)
        ).order_by('month')
        
        # Get min/max to exclude empty periods
        if data.exists():
            min_month = data.first()['month']
            max_month = data.last()['month']
            data = data.filter(month__gte=min_month, month__lte=max_month)
        
        result = {
            'labels': [d['month'].strftime('%b %Y') for d in data],
            'cases': [float(d['cases'] or 0) for d in data],
            'deaths': [float(d['deaths'] or 0) for d in data],
            'zones': [int(d['zones'] or 0) for d in data],
        }
        return result
    
    @staticmethod
    def get_zone_distribution(province_name, disease_name, top_n=15):
        """Distribution by health zone (per capita)"""
        zones = HealthZone.objects.filter(
            territory__province__name=province_name,
            clinicaldata__disease_name=disease_name
        ).distinct().annotate(
            total_cases=Sum('clinicaldata__cases_count')
        ).order_by('-total_cases')[:top_n]
        
        result = {
            'labels': [z.name for z in zones],
            'cases': [float(z.total_cases or 0) for z in zones],
            'per_capita': [],
            'population': []
        }
        
        for zone in zones:
            if zone.population and zone.population > 0:
                pc = (zone.total_cases / zone.population) * 10000  # Per 10k
            else:
                pc = 0
            result['per_capita'].append(round(pc, 2))
            result['population'].append(zone.population or 0)
        
        return result
    
    @staticmethod
    def get_climate_impact(province_name, disease_name):
        """Correlation between climate and disease"""
        # Get historical clinical data
        zones = HealthZone.objects.filter(
            territory__province__name=province_name,
            climatedata__isnull=False,
            clinicaldata__disease_name=disease_name
        ).distinct()
        
        climate_disease_pairs = []
        
        for zone in zones:
            climate_data = ClimateData.objects.filter(
                health_zone=zone
            ).order_by('-month')[:36]  # Last 36 months
            
            for clim in climate_data:
                clinical = ClinicalData.objects.filter(
                    health_zone=zone,
                    disease_name=disease_name,
                    month__year=clim.month.year,
                    month__month=clim.month.month
                ).aggregate(total=Sum('cases_count'))
                
                cases = clinical['total'] or 0
                if cases > 0:
                    climate_disease_pairs.append({
                        'temperature': clim.temperature - 273.15,  # K to C
                        'precipitation': clim.precipitation,
                        'humidity': clim.humidity,
                        'cases': cases
                    })
        
        # Calculate correlations
        result = {
            'pairs_count': len(climate_disease_pairs),
            'correlations': {
                'temperature': None,
                'precipitation': None,
                'humidity': None
            },
            'current_conditions': None,
            'interpretation': ""
        }
        
        if len(climate_disease_pairs) >= 10:
            temps = np.array([p['temperature'] for p in climate_disease_pairs])
            precips = np.array([p['precipitation'] for p in climate_disease_pairs])
            humidities = np.array([p['humidity'] for p in climate_disease_pairs])
            cases = np.array([p['cases'] for p in climate_disease_pairs])
            
            try:
                r_temp, _ = pearsonr(temps, cases)
                result['correlations']['temperature'] = round(abs(r_temp), 3)
            except:
                pass
            
            try:
                r_precip, _ = pearsonr(precips, cases)
                result['correlations']['precipitation'] = round(abs(r_precip), 3)
            except:
                pass
            
            try:
                r_humid, _ = pearsonr(humidities, cases)
                result['correlations']['humidity'] = round(abs(r_humid), 3)
            except:
                pass
            
            # Current conditions
            latest = ClimateData.objects.filter(
                health_zone__territory__province__name=province_name
            ).order_by('-month').first()
            
            if latest:
                result['current_conditions'] = {
                    'temperature': round(latest.temperature - 273.15, 1),
                    'precipitation': round(latest.precipitation, 3),
                    'humidity': round(latest.humidity, 1)
                }
            
            # Generate interpretation
            corr_vals = [v for v in result['correlations'].values() if v is not None]
            if corr_vals:
                max_corr = max(corr_vals)
                if max_corr > 0.4:
                    result['interpretation'] = f"🌡️ Forte corrélation climatique détectée (r={max_corr})"
                elif max_corr > 0.2:
                    result['interpretation'] = f"📊 Corrélation modérée avec climat (r={max_corr})"
                else:
                    result['interpretation'] = "⛅ Facteurs climatiques peu influents"
        
        return result
    
    @staticmethod
    def get_predictions_comparison(province_name, disease_name):
        """Compare actual data vs predictions"""
        # Get recent predictions
        preds = Prediction.objects.filter(
            disease_name=disease_name,
            health_zone__territory__province__name=province_name,
            predicted_cases__gt=0
        ).order_by('-prediction_date')[:12]
        
        result = {
            'labels': [p.health_zone.name for p in preds],
            'predicted': [float(p.predicted_cases) for p in preds],
            'models': [(p.influence_factors or {}).get('model_used', 'IA') for p in preds],
            'reliability': [(p.influence_factors or {}).get('reliability', 0) for p in preds]
        }
        return result
    
    @staticmethod
    def get_summary_stats(province_name, disease_name):
        """Get summary statistics"""
        qs = ClinicalData.objects.filter(
            disease_name=disease_name,
            health_zone__territory__province__name=province_name
        )
        
        stats = qs.aggregate(
            total_cases=Sum('cases_count'),
            total_deaths=Sum('deaths_count'),
            affected_zones=Count('health_zone', distinct=True),
            avg_cases=Avg('cases_count')
        )
        
        total_cases = stats['total_cases'] or 0
        total_deaths = stats['total_deaths'] or 0
        
        return {
            'total_cases': int(total_cases),
            'total_deaths': int(total_deaths),
            'fatality_rate': round((total_deaths / total_cases * 100), 1) if total_cases > 0 else 0,
            'affected_zones': stats['affected_zones'] or 0,
            'avg_monthly': round(stats['avg_cases'] or 0, 0)
        }
