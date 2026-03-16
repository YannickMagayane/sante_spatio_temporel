"""
Enhanced disease analysis views with smart graphing and climate analysis
"""
import json
import pandas as pd
from datetime import datetime
from django.views.generic import TemplateView
from django.db.models import Sum, Avg
from .models import ClinicalData, Prediction, Province, HealthZone, ClimateData
from .disease_analyzer import DiseaseAnalyzer
from .smart_graphs import SmartGraphGenerator
import logging

logger = logging.getLogger(__name__)


class EnhancedDiseaseAnalysisView(TemplateView):
    """Enhanced disease analysis with monthly predictions and climate impact"""
    template_name = "analytics/disease_interactive.html"
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        disease_name = self.request.GET.get('disease', 'Paludisme')
        province_name = self.request.GET.get('province')
        
        context['disease'] = disease_name
        context['province'] = province_name
        
        # Get historical clinical data (with empty period filtering)
        historical_df = self._get_historical_data(disease_name, province_name)
        context['has_data'] = not historical_df.empty
        
        # Get next month forecast
        forecast_data = self._get_monthly_forecast(disease_name, province_name)
        context['forecast'] = forecast_data
        context['next_month'] = forecast_data.get('prediction_date', '').strftime('%B %Y') if forecast_data.get('prediction_date') else 'N/A'
        
        # Get climate data for impact analysis
        climate_df = self._get_climate_data(disease_name, province_name)
        context['has_climate_data'] = not climate_df.empty
        
        # Get correlation data
        climate_impact = self._analyze_climate_impact(historical_df, climate_df, disease_name)
        context['climate_impact'] = climate_impact
        
        # Generate smart graphs
        logger.info(f"DEBUG: Data before chart generation - hist:{len(historical_df)} rows, climate:{len(climate_df)} rows, has_forecast:{bool(forecast_data)}")
        charts = self._generate_smart_charts(
            disease_name=disease_name,
            province_name=province_name,
            historical_df=historical_df,
            forecast_data=forecast_data,
            climate_df=climate_df
        )
        logger.info(f"DEBUG: Charts generated: {list(charts.keys())}")
        context.update(charts)
        
        # Get zone-level predictions
        if province_name:
            zone_preds = self._get_zone_predictions(disease_name, province_name)
            context['zone_predictions'] = zone_preds
            # Calculate total for percentage display
            total_pred = sum(z.get('predicted_cases', 0) for z in zone_preds) if zone_preds else 0
            context['total_predicted_cases'] = total_pred
        
        # Get recommendations
        context['recommendations'] = self._get_recommendations(disease_name, province_name, forecast_data)
        
        # Get list of provinces for selector
        provinces = Province.objects.filter(
            territories__health_zones__clinicaldata__disease_name=disease_name
        ).distinct().order_by('name')
        context['provinces'] = provinces
        
        return context
    
    def _get_historical_data(self, disease_name, province_name=None):
        """Get historical clinical data, filtering empty periods"""
        query = ClinicalData.objects.filter(disease_name=disease_name)
        
        if province_name:
            query = query.filter(health_zone__territory__province__name=province_name)
        
        # Group by month
        data = query.values('month').annotate(
            cases_count=Sum('cases_count')
        ).order_by('month')
        
        df = pd.DataFrame(list(data))
        if df.empty:
            return pd.DataFrame()
        
        df['month'] = pd.to_datetime(df['month'])
        
        # Remove empty edge periods (but keep data in middle)
        has_data = (df['cases_count'].notna()) & (df['cases_count'] > 0)
        if has_data.any():
            first_idx = has_data.idxmax()
            last_idx = len(df) - 1 - has_data[::-1].idxmax()
            df = df.iloc[first_idx:last_idx + 1]
        
        return df.reset_index(drop=True)
    
    def _get_monthly_forecast(self, disease_name, province_name=None):
        """Get next month forecast"""
        query = Prediction.objects.filter(disease_name=disease_name)
        
        if province_name:
            query = query.filter(health_zone__territory__province__name=province_name)
        
        # Get most recent prediction
        latest = query.order_by('-prediction_date').first()
        
        if latest:
            return {
                'predicted_cases': latest.predicted_cases,
                'prediction_date': latest.prediction_date,
                'reliability': self._calculate_reliability(latest),
                'health_zone': latest.health_zone.name if latest.health_zone else 'National'
            }
        
        return {}
    
    def _get_climate_data(self, disease_name, province_name=None):
        """Get climate data for the same period as historical data"""
        query = ClimateData.objects.all()
        
        if province_name:
            query = query.filter(
                health_zone__territory__province__name=province_name
            )
        
        # Group by month and average across zones
        data = query.values('month').annotate(
            temperature=Avg('temperature'),
            precipitation=Avg('precipitation'),
            humidity=Avg('humidity')
        ).order_by('month')
        
        df = pd.DataFrame(list(data))
        if df.empty:
            return pd.DataFrame()
        
        df['month'] = pd.to_datetime(df['month'])
        return df.rename(columns={'month': 'date'})
    
    def _analyze_climate_impact(self, historical_df, climate_df, disease_name):
        """Analyze how climate factors correlate with disease"""
        impact = {
            'correlations': {},
            'interpretation': '',
            'recommendations': []
        }
        
        if historical_df.empty or climate_df.empty:
            return impact
        
        # Merge data
        merged = pd.merge(
            historical_df[['month', 'cases_count']],
            climate_df[['date', 'temperature', 'precipitation', 'humidity']],
            left_on='month', right_on='date', how='inner'
        )
        
        if merged.empty or len(merged) < 3:
            return impact
        
        # Calculate correlations
        if 'temperature' in merged.columns:
            corr_temp = merged['temperature'].corr(merged['cases_count'])
            impact['correlations']['temperature'] = float(corr_temp) if not pd.isna(corr_temp) else 0
        
        if 'precipitation' in merged.columns:
            corr_precip = merged['precipitation'].corr(merged['cases_count'])
            impact['correlations']['precipitation'] = float(corr_precip) if not pd.isna(corr_precip) else 0
        
        if 'humidity' in merged.columns:
            corr_humid = merged['humidity'].corr(merged['cases_count'])
            impact['correlations']['humidity'] = float(corr_humid) if not pd.isna(corr_humid) else 0
        
        # Generate interpretation
        impact['interpretation'] = self._interpret_climate_impact(
            disease_name, impact['correlations']
        )
        
        return impact
    
    def _interpret_climate_impact(self, disease_name, correlations):
        """Generate interpretation text based on climate correlations"""
        interpretation = ""
        
        # Disease-specific climate relationships
        climate_profiles = {
            'Paludisme': {
                'temp_trigger': (18, 32),  # Malaria mosquito optimal temp
                'precip_risk': True,
                'humidity_risk': True,
                'description': 'Le paludisme est fortement lié à la température, aux précipitations et à l\'humidité'
            },
            'Choléra': {
                'temp_trigger': (15, 30),
                'precip_risk': True,
                'humidity_risk': False,
                'description': 'Le choléra augmente lors de précipitations (contamination hydrique) et de températures chaudes'
            },
            'Rougeole': {
                'temp_trigger': None,
                'precip_risk': False,
                'humidity_risk': True,
                'description': 'La rougeole survit mieux dans les environnements secs et frais'
            }
        }
        
        profile = climate_profiles.get(disease_name, {
            'description': 'Relations climatiques complexes avec la transmission'
        })
        
        interpretation = profile.get('description', '')
        
        # Add correlation-based insights
        corr = correlations
        if corr.get('temperature', 0) > 0.5:
            interpretation += " | ⬆️ Forte corrélation avec température"
        if corr.get('precipitation', 0) > 0.5:
            interpretation += " | 💧 Forte corrélation avec précipitations"
        
        return interpretation
    
    def _get_zone_predictions(self, disease_name, province_name):
        """Get top 15 zone predictions for province"""
        zones = HealthZone.objects.filter(
            territory__province__name=province_name,
            clinicaldata__disease_name=disease_name
        ).distinct()
        
        zone_data = []
        for zone in zones:
            pred = Prediction.objects.filter(
                disease_name=disease_name,
                health_zone=zone
            ).order_by('-prediction_date').first()
            
            if pred:
                zone_data.append({
                    'zone_name': zone.name,
                    'predicted_cases': pred.predicted_cases,
                    'prediction_date': pred.prediction_date.strftime('%Y-%m-%d')
                })
        
        # Sort and return top 15
        return sorted(zone_data, key=lambda x: x['predicted_cases'], reverse=True)[:15]
    
    def _generate_smart_charts(self, disease_name, province_name, historical_df, 
                               forecast_data, climate_df):
        """Generate all smart charts"""
        charts = {}
        gen = SmartGraphGenerator()
        
        logger.info(f"CHART GEN START: hist_empty={historical_df.empty}, hist_len={len(historical_df)}, forecast={bool(forecast_data)}")
        logger.info(f"  hist_cols={list(historical_df.columns) if not historical_df.empty else 'N/A'}")
        logger.info(f"  climate_empty={climate_df.empty}, climate_len={len(climate_df)}, climate_cols={list(climate_df.columns)}")
        
        try:
            # 1. Monthly forecast chart
            if not historical_df.empty and forecast_data:
                logger.info(f"Generating monthly_forecast chart...")
                charts['monthly_forecast'] = gen.monthly_forecast_chart(
                    historical_df, forecast_data, disease_name
                )
                logger.info(f"✓ monthly_forecast generated")
            else:
                logger.info(f"SKIP monthly_forecast: hist_empty={historical_df.empty}, has_forecast={bool(forecast_data)}")
        except Exception as e:
            logger.error(f"Monthly forecast chart error: {e}", exc_info=True)
        
        try:
            # 2. Climate impact chart
            if not historical_df.empty and not climate_df.empty:
                logger.info(f"Generating climate_impact chart...")
                charts['climate_impact'] = gen.climate_impact_chart(
                    climate_df, historical_df, disease_name, province_name
                )
                logger.info(f"✓ climate_impact generated")
            else:
                logger.info(f"SKIP climate_impact: hist_empty={historical_df.empty}, climate_empty={climate_df.empty}")
        except Exception as e:
            logger.error(f"Climate impact chart error: {e}", exc_info=True)
        
        try:
            # 3. With/Without climate comparison
            if not historical_df.empty:
                logger.info(f"Generating climate_comparison chart...")
                charts['climate_comparison'] = gen.comparison_with_without_climate(
                    historical_df, climate_df, disease_name, province_name
                )
                logger.info(f"✓ climate_comparison generated")
            else:
                logger.info(f"SKIP climate_comparison: hist_empty={historical_df.empty}")
        except Exception as e:
            logger.error(f"Climate comparison chart error: {e}", exc_info=True)
        
        logger.info(f"CHART GEN END: generated {list(charts.keys())}")
        return charts
    
    def _get_recommendations(self, disease_name, province_name, forecast_data):
        """Get actionable recommendations based on forecast"""
        recs = {
            'Paludisme': {
                'icon': '🦟',
                'title': 'Prédictions de Paludisme - Actions Requises',
                'actions': [
                    '🏥 Assurer les stocks de CTA (Combinaisons Thérapeutiques à base d\'Artémisinine)',
                    '💊 Préparer les RDT (Rapid Diagnostic Tests) pour le diagnostic',
                    '🪵 Déployer les moustiquaires imprégnées si transmission augmente',
                    '📊 Renforcer la surveillance communautaire'
                ]
            },
            'Choléra': {
                'icon': '💧',
                'title': 'Alerte Choléra - Mesures d\'Urgence',
                'actions': [
                    '🚰 Préparer les kit de purification d\'eau',
                    '🏥 Mettre en place les centres de traitement du choléra',
                    '💉 Organiser les campagnes de vaccination orale du choléra',
                    '🧤 Renforcer les protocoles d\'hygiène et assainissement'
                ]
            }
        }
        
        base_rec = recs.get(disease_name, {
            'icon': '⚕️',
            'title': f'Actions pour {disease_name}',
            'actions': [
                '📊 Renforcer la surveillance',
                '💊 Assurer la disponibilité des traitements',
                '🏥 Adapter les capacités',
                '🧪 Confirmer les cas en laboratoire'
            ]
        })
        
        # Add forecast-based risk level
        if forecast_data:
            predicted = forecast_data.get('predicted_cases', 0)
            
            if predicted > 5000:
                base_rec['risk_level'] = '🔴 CRITIQUE'
                base_rec['urgency'] = 'Action immédiate requise'
            elif predicted > 1000:
                base_rec['risk_level'] = '🟠 ÉLEVÉ'
                base_rec['urgency'] = 'Activation du plan de réponse'
            elif predicted > 100:
                base_rec['risk_level'] = '🟡 MODÉRÉ'
                base_rec['urgency'] = 'Préparation accrue'
            else:
                base_rec['risk_level'] = '🟢 BAS'
                base_rec['urgency'] = 'Surveillance de routine'
        
        return base_rec
    
    def _calculate_reliability(self, prediction):
        """Calculate reliability score for prediction - based on available fields"""
        reliability = 0.75  # Base reliability
        
        # Check if we have contextual risk data
        if prediction.contextual_risk:
            reliability += 0.15
        
        # Increase for newer predictions
        from datetime import datetime, timedelta
        age = datetime.now().date() - prediction.prediction_date
        if age <= timedelta(days=30):
            reliability += 0.1
        
        return min(reliability, 1.0)
