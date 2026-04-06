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
                        'temperature': clim.temperature,  # Already in Celsius
                        'precipitation': clim.precipitation,  # Already in mm
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
                    'temperature': round(latest.temperature, 1),  # Already Celsius
                    'precipitation': round(latest.precipitation, 1),  # Already mm
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
    
    @staticmethod
    def get_yearly_predictions(disease_name, province_name=None):
        """
        Récupère les prédictions progressives par année.
        Retourne un dict avec années comme clés et prédictions pour le mois 1 de l'année suivante.
        
        :param disease_name: Nom de la maladie
        :param province_name: Optionnel, province spécifique
        :returns: dict {year: {'date': ..., 'predicted_cases': ..., 'reliability': ...}}
        """
        from datetime import date
        
        # Récupérer toutes les prédictions pour cette maladie, groupées par année
        preds_qs = Prediction.objects.filter(disease_name=disease_name)
        
        if province_name:
            preds_qs = preds_qs.filter(health_zone__territory__province__name=province_name)
        
        # Grouper par year de prediction_date
        yearly_preds = {}
        
        for pred in preds_qs.order_by('prediction_date'):
            pred_year = pred.prediction_date.year
            
            # Sauvegarder avec le format: "2015-01" → prédiction pour 2015-01
            year_key = f"{pred_year}"
            
            if year_key not in yearly_preds:
                yearly_preds[year_key] = {
                    'date': pred.prediction_date.strftime('%Y-%m'),
                    'predicted_cases': 0,
                    'prediction_count': 0,
                    'reliability_sum': 0,
                    'reliability_count': 0
                }
            
            # Accumuler les cases
            yearly_preds[year_key]['predicted_cases'] += float(pred.predicted_cases or 0)
            yearly_preds[year_key]['prediction_count'] += 1
            
            # Extraire la reliability du JSON
            try:
                influence = json.loads(pred.influence_factors) if isinstance(pred.influence_factors, str) else pred.influence_factors
                reliability = float((influence or {}).get('reliability', 33.0))
            except:
                reliability = 33.0
            
            yearly_preds[year_key]['reliability_sum'] += reliability
            yearly_preds[year_key]['reliability_count'] += 1
        
        # Calculer les moyennes de reliability et nettoyer les champs temporaires
        for year_key in yearly_preds:
            count = yearly_preds[year_key].get('reliability_count', 1)
            if count > 0:
                yearly_preds[year_key]['reliability'] = round(
                    yearly_preds[year_key]['reliability_sum'] / count, 1
                )
            else:
                yearly_preds[year_key]['reliability'] = 33.0
            
            # Supprimer les champs temporaires
            del yearly_preds[year_key]['reliability_sum']
            del yearly_preds[year_key]['reliability_count']
        
        return yearly_preds
    
    @staticmethod
    def get_yearly_predictions_with_zone_details(disease_name, province_name=None):
        """
        Récupère les prédictions progressives par année AVEC détails par zone de santé.
        Retourne un dict imbriqué: {year: {zone_name: {'predicted_cases': ..., 'reliability': ...}}}
        
        :param disease_name: Nom de la maladie
        :param province_name: Optionnel, province spécifique
        :returns: dict {year: {zone: {'predicted_cases': ..., 'reliability': ..., 'count': ...}}}
        """
        from datetime import date
        
        # Récupérer toutes les prédictions pour cette maladie
        preds_qs = Prediction.objects.filter(disease_name=disease_name).select_related('health_zone')
        
        if province_name:
            preds_qs = preds_qs.filter(health_zone__territory__province__name=province_name)
        
        # Grouper par année et par zone de santé
        yearly_zone_preds = {}
        
        for pred in preds_qs.order_by('prediction_date', 'health_zone__name'):
            pred_year = pred.prediction_date.year
            zone_name = pred.health_zone.name if pred.health_zone else 'Unknown'
            
            if pred_year not in yearly_zone_preds:
                yearly_zone_preds[pred_year] = {}
            
            if zone_name not in yearly_zone_preds[pred_year]:
                yearly_zone_preds[pred_year][zone_name] = {
                    'predicted_cases': 0,
                    'count': 0,
                    'reliability_sum': 0,
                    'date': pred.prediction_date.strftime('%Y-%m')
                }
            
            # Incrémenter les valeurs
            yearly_zone_preds[pred_year][zone_name]['predicted_cases'] += float(pred.predicted_cases or 0)
            yearly_zone_preds[pred_year][zone_name]['count'] += 1
            
            # Récupérer la reliability de influence_factors
            try:
                influence = json.loads(pred.influence_factors) if isinstance(pred.influence_factors, str) else pred.influence_factors
                reliability = float(influence.get('reliability', 33.0))
            except:
                reliability = 33.0
            
            yearly_zone_preds[pred_year][zone_name]['reliability_sum'] += reliability
        
        # Calculer les moyennes de reliability
        for year in yearly_zone_preds:
            for zone in yearly_zone_preds[year]:
                count = yearly_zone_preds[year][zone]['count']
                if count > 0:
                    yearly_zone_preds[year][zone]['reliability'] = round(
                        yearly_zone_preds[year][zone]['reliability_sum'] / count, 1
                    )
                else:
                    yearly_zone_preds[year][zone]['reliability'] = 33.0
                
                # Supprimer les valeurs temporaires
                del yearly_zone_preds[year][zone]['reliability_sum']
                del yearly_zone_preds[year][zone]['count']
        
        return yearly_zone_preds

    @staticmethod
    def get_recurrence_percentage(health_zone, disease_name):
        """
        Calcule le pourcentage de récurrence (présence) d'une maladie dans une zone de santé.

        Méthode statistique :
        - Période totale = tous les mois entre la première et la dernière donnée DISPONIBLE
        - Si des mois sont manquants dans la base, on les estime comme "non occurrencs" (0 cas)
        - Cela évite des faux 100% quand seuls les mois avec cas sont enregistrés

        :param health_zone: Objet HealthZone
        :param disease_name: Nom de la maladie
        :returns: dict avec 'percentage', 'active_months', 'total_months_in_period', 'data_months',
                  'first_date', 'last_date', 'completeness', 'message'
        """
        from django.db.models import Count, Min, Max, Sum
        from datetime import datetime

        # Récupérer toutes les données cliniques pour cette zone et maladie
        qs = ClinicalData.objects.filter(
            health_zone=health_zone,
            disease_name=disease_name
        )

        # Nombre d'enregistrements disponibles
        data_months = qs.count()

        if data_months == 0:
            return {
                'percentage': 0.0,
                'active_months': 0,
                'total_months_in_period': 0,
                'data_months': 0,
                'completeness': 0.0,
                'first_date': None,
                'last_date': None,
                'message': "⚠️ Aucune donnée historique disponible"
            }

        # Compter les mois avec des cas > 0 (mois actifs) - PARMI LES DONNÉES DISPONIBLES
        active_months_from_data = qs.filter(cases_count__gt=0).count()

        # Obtenir la période couverte (première et dernière dates disponibles)
        dates = qs.aggregate(first=Min('month'), last=Max('month'))
        first_date = dates['first']
        last_date = dates['last']

        # Calculer le nombre total de mois dans la période calendaire
        if first_date and last_date:
            total_months_in_period = (last_date.year - first_date.year) * 12 + (last_date.month - first_date.month) + 1
        else:
            total_months_in_period = data_months

        # Estimer les mois manquants : on suppose que les mois non enregistrés sont des mois avec 0 cas
        missing_months = total_months_in_period - data_months

        # Nombre total de mois actifs = actifs parmi les données + (certains manquants pourraient être actifs, mais par prudence on les compte comme inactifs)
        # Par défaut, on considère que les mois manquants sont des mois sans cas (hypothèse conservatrice)
        estimated_active_months = active_months_from_data  # on ne rajoute pas d'actifs hypothétiques

        # Calculer le pourcentage de récurrence sur la période totale
        percentage = round((estimated_active_months / total_months_in_period) * 100, 1) if total_months_in_period > 0 else 0.0

        # Calculer le taux de complétude des données
        completeness = round((data_months / total_months_in_period) * 100, 1) if total_months_in_period > 0 else 0.0

        # Générer un message interprétatif avec contexte de complétude
        if completeness < 70:
            completeness_msg = f"⚠️ Données incomplètes ({completeness}% de la période)"
        else:
            completeness_msg = f"✅ Données suffisantes ({completeness}% de la période)"

        if percentage >= 80:
            message = f"{completeness_msg} - RÉCURRENCE ÉLEVÉE ({percentage}%) - Maladie très persistante"
        elif percentage >= 60:
            message = f"{completeness_msg} - RÉCURRENCE MODÉRÉE ({percentage}%) - Présence fréquente"
        elif percentage >= 40:
            message = f"{completeness_msg} - RÉCURRENCE FAIBLE ({percentage}%) - Apparitions occasionnelles"
        elif percentage >= 20:
            message = f"{completeness_msg} - RÉCURRENCE TRÈS FAIBLE ({percentage}%) - Maladie sporadique"
        else:
            message = f"{completeness_msg} - MALADIE RARE ({percentage}%) - Quasi-absence historique"

        return {
            'percentage': percentage,
            'active_months': estimated_active_months,
            'total_months_in_period': total_months_in_period,
            'data_months': data_months,
            'completeness': completeness,
            'first_date': first_date,
            'last_date': last_date,
            'message': message
        }

    @staticmethod
    def get_disease_recurrence_summary(disease_name, province_name=None):
        """
        Calcule un résumé de récurrence pour une maladie à l'échelle des zones.

        :param disease_name: Nom de la maladie
        :param province_name: Optionnel, province spécifique
        :returns: dict avec 'avg_recurrence', 'zones_count', 'max_recurrence', 'min_recurrence', 'zones_detail'
        """
        from django.db.models import Count, Sum, Avg, Max, Min, Q, FloatField
        from django.db.models.functions import Cast

        # Base queryset: zones with clinical data for this disease
        zones_qs = HealthZone.objects.filter(
            clinicaldata__disease_name=disease_name
        )
        if province_name:
            zones_qs = zones_qs.filter(territory__province__name=province_name)

        # Distinct zones
        zones_qs = zones_qs.distinct()

        # Annoter chaque zone avec ses statistiques de récurrence
        zones_with_stats = zones_qs.annotate(
            total_months=Count('clinicaldata', filter=Q(clinicaldata__disease_name=disease_name), distinct=True),
            active_months=Count('clinicaldata', filter=Q(clinicaldata__disease_name=disease_name, clinicaldata__cases_count__gt=0), distinct=True)
        ).filter(total_months__gt=0)

        # Calculer le pourcentage par zone en utilisant la méthode corrigée
        zones_detail = []
        percentages = []

        # Limiter à 100 zones pour performance
        for zone in zones_with_stats[:100]:
            recurrence_data = DiseaseAnalyzer.get_recurrence_percentage(zone, disease_name)
            pct = recurrence_data['percentage']
            percentages.append(pct)
            zones_detail.append({
                'zone_name': zone.name,
                'province': zone.province.name if zone.province else 'N/A',
                'percentage': pct,
                'active_months': recurrence_data['active_months'],
                'total_months_in_period': recurrence_data['total_months_in_period'],
                'completeness': recurrence_data['completeness']
            })

        # Calculer les moyennes
        avg_recurrence = round(sum(percentages) / len(percentages), 1) if percentages else 0.0
        max_recurrence = round(max(percentages), 1) if percentages else 0.0
        min_recurrence = round(min(percentages), 1) if percentages else 0.0

        return {
            'avg_recurrence': avg_recurrence,
            'zones_count': len(percentages),
            'max_recurrence': max_recurrence,
            'min_recurrence': min_recurrence,
            'zones_detail': sorted(zones_detail, key=lambda x: x['percentage'], reverse=True)[:20]  # Top 20
        }
