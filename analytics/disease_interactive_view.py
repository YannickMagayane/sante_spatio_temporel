
"""
Modern interactive disease analysis views with climate impact
"""
import json
import base64
import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from django.views.generic import TemplateView
from .disease_analyzer import DiseaseAnalyzer

# Matplotlib styling
plt.style.use('dark_background')
sns.set_palette("husl")


# ====== NEW INTERACTIVE DISEASE ANALYSIS VIEW ======
class DiseaseInteractiveView(TemplateView):
    """Modern interactive disease analysis with Chart.js & climate impact"""
    template_name = "analytics/disease_interactive.html"
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        disease_name = self.request.GET.get('disease', 'Paludisme')
        province_name = self.request.GET.get('province')
        
        context['disease'] = disease_name
        context['province'] = province_name
        
        # Get timeline data (no empty years)
        timeline_data = DiseaseAnalyzer.get_timeline_data(disease_name, province_name)
        context['timeline_data'] = json.dumps(timeline_data)
        
        # Initialize default values for optional fields
        context['zone_distribution'] = json.dumps({'labels': [], 'cases': []})
        context['predictions_data'] = json.dumps({'labels': [], 'predicted': [], 'confidence_upper': [], 'confidence_lower': []})
        context['provincial_predictions'] = json.dumps({'labels': [], 'predicted': []})
        context['zone_predictions'] = json.dumps({'labels': [], 'predicted': []})
        context['climate_impact'] = {
            'correlations': {'temperature': 0, 'precipitation': 0, 'humidity': 0},
            'current_conditions': None,
            'interpretation': ""
        }
        context['stats'] = None
        context['recommendations'] = self._get_recommendations(disease_name, None)
        
        # National-level predictions across all provinces
        national_preds = self._get_national_predictions(disease_name)
        context['provincial_predictions'] = json.dumps(national_preds)
        
        # Only show detailed analysis if province is selected
        if province_name:
            # Get summary stats
            stats = DiseaseAnalyzer.get_summary_stats(province_name, disease_name)
            context['stats'] = stats
            
            # Get zone distribution
            zone_dist = DiseaseAnalyzer.get_zone_distribution(province_name, disease_name)
            context['zone_distribution'] = json.dumps(zone_dist)
            
            # Get zone-level predictions
            zone_preds = self._get_zone_predictions(province_name, disease_name)
            context['zone_predictions'] = json.dumps(zone_preds)
            
            # Get climate impact
            climate_impact = DiseaseAnalyzer.get_climate_impact(province_name, disease_name)
            # Convert numpy types to Python floats for JSON serialization
            if climate_impact.get('correlations'):
                climate_impact['correlations'] = {
                    'temperature': float(climate_impact['correlations'].get('temperature', 0) or 0),
                    'precipitation': float(climate_impact['correlations'].get('precipitation', 0) or 0),
                    'humidity': float(climate_impact['correlations'].get('humidity', 0) or 0)
                }
            context['climate_impact'] = climate_impact
            
            # Get predictions
            predictions = DiseaseAnalyzer.get_predictions_comparison(province_name, disease_name)
            context['predictions_data'] = json.dumps(predictions)
            
            # Get intelligent recommendations for this province
            context['recommendations'] = self._get_recommendations(disease_name, province_name, stats)
        
        # Get all provinces affected by this disease
        from .models import Province
        provinces = Province.objects.filter(
            territories__health_zones__clinicaldata__disease_name=disease_name
        ).distinct().order_by('name')
        context['provinces'] = provinces
        
        # Generate Matplotlib Charts (Beautiful visualizations)
        national_preds = self._get_national_predictions(disease_name) if 'provincial_predictions' not in context else json.loads(context.get('provincial_predictions', '{}'))
        
        if not province_name:
            zone_dist = {'labels': [], 'cases': []}
            zone_preds = {'labels': [], 'predicted': []}
            pred_data = {'labels': [], 'predicted': [], 'confidence_upper': [], 'confidence_lower': []}
            climate_impact_data = {'correlations': {'temperature': 0, 'precipitation': 0, 'humidity': 0}}
        else:
            zone_dist = json.loads(context.get('zone_distribution', '{}'))
            zone_preds = json.loads(context.get('zone_predictions', '{}'))
            pred_data = json.loads(context.get('predictions_data', '{}'))
            climate_impact_data = context.get('climate_impact', {})
        
        charts = self._generate_matplotlib_charts(
            disease_name, 
            province_name,
            timeline_data=timeline_data,
            provincial_preds=national_preds,
            zone_dist=zone_dist,
            zone_preds=zone_preds,
            pred_data=pred_data,
            climate_impact=climate_impact_data
        )
        
        context.update(charts)
        
        return context
    
    def _get_national_predictions(self, disease_name):
        """Get predictions aggregated by province"""
        from .models import Province, Prediction
        from django.db.models import Sum
        
        provinces = Province.objects.filter(
            territories__health_zones__clinicaldata__disease_name=disease_name
        ).distinct()
        
        labels = []
        predicted = []
        
        for prov in provinces[:15]:  # Top 15 provinces
            preds = Prediction.objects.filter(
                disease_name=disease_name,
                health_zone__territory__province=prov
            ).aggregate(total=Sum('predicted_cases'))
            
            if preds['total']:
                labels.append(prov.name)
                predicted.append(preds['total'])
        
        return {'labels': labels, 'predicted': predicted}
    
    def _get_zone_predictions(self, province_name, disease_name):
        """Get predictions by health zone"""
        from .models import HealthZone, Prediction, Province
        from django.db.models import Sum
        
        province = Province.objects.get(name=province_name)
        zones = HealthZone.objects.filter(
            territory__province=province,
            clinicaldata__disease_name=disease_name
        ).distinct()
        
        labels = []
        predicted = []
        
        for zone in zones[:10]:  # Top 10 zones
            preds = Prediction.objects.filter(
                disease_name=disease_name,
                health_zone=zone
            ).aggregate(total=Sum('predicted_cases'))
            
            if preds['total']:
                labels.append(zone.name[:25])
                predicted.append(preds['total'])
        
        return {'labels': labels, 'predicted': predicted}
    
    def _get_recommendations(self, disease_name, province_name=None, stats=None):
        """Generate intelligent recommendations based on disease data"""
        recommendations = {
            'Paludisme': {
                'icon': '🦟',
                'title': 'Stratégie de Contrôle du Paludisme',
                'actions': [
                    '🪵 Déployer les moustiquaires dans les zones à forte transmission (>100 cas/zone)',
                    '💊 Assurer les niveaux de stock de combinaisons thérapeutiques à base d\'artémisinine (CTA)',
                    '🏥 Renforcer la surveillance dans les zones frontalières et aux pics saisonniers',
                    '💉 Prioriser les femmes enceintes et les enfants <5 ans avec un traitement préventif'
                ]
            },
            'Choléra': {
                'icon': '💧',
                'title': 'Réponse aux Épidémies de Choléra',
                'actions': [
                    '🚰 Établir des centres de purification d\'eau dans les zones d\'épidémie',
                    '🏥 Pré-positionner des CTC (Centres de Traitement du Choléra) à moins de 5km des foyers',
                    '📢 Lancer l\'éducation communautaire sur la transmission hydrique',
                    '💉 Initier les campagnes OVC (Vaccin Oral du Choléra) dans les populations à haut risque'
                ]
            },
            'VIH/SIDA': {
                'icon': '🔴',
                'title': 'Prévention et Traitement du VIH/SIDA',
                'actions': [
                    '🧪 Intensifier les services de CDV (Conseil et Dépistage Volontaire)',
                    '💊 Assurer l\'accès universel à la thérapie antirétrovirale (TAR)',
                    '👶 Mettre en œuvre la PTME (Prévention de la Transmission Mère-Enfant)',
                    '📚 Renforcer la sensibilisation communautaire et la réduction de la stigmatisation'
                ]
            },
            'Rougeole': {
                'icon': '💉',
                'title': 'Campagne d\'Élimination de la Rougeole',
                'actions': [
                    '📋 Atteindre ≥95% de couverture vaccinale dans toutes les zones',
                    '🎯 Mener des campagnes de rattrapage dans les populations précédemment manquées',
                    '⚠️ Mettre en œuvre des protocoles de réponse rapide aux cas confirmés',
                    '📊 Renforcer la surveillance basée sur les cas et la confirmation en laboratoire'
                ]
            },
            'Hépatite': {
                'icon': '🔬',
                'title': 'Stratégie de Prévention de l\'Hépatite',
                'actions': [
                    '💉 Vaccination universelle des nouveau-nés contre l\'hépatite B dans toutes les structures',
                    '🧪 Dépister tous les donneurs de sang et mettre en place les protocoles TPHA',
                    '📚 Promotion de l\'hygiène et de l\'assainissement (WASH) pour prévenir HAV/HEV',
                    '🏥 Assurer l\'accès à la thérapie antivirale dans les centres tertiaires'
                ]
            },
            'Coqueluche': {
                'icon': '🫁',
                'title': 'Programme de Contrôle de la Coqueluche',
                'actions': [
                    '👶 Assurer la série complète de vaccination DTC chez les nourrissons',
                    '⚠️ Renforcer les adolescents et adultes avec Tdca',
                    '📢 Identification précoce et isolement des cas suspects',
                    '💊 Mettre en œuvre la prophylaxie post-exposition (PPE) pour les contacts'
                ]
            }
        }
        
        base_rec = recommendations.get(disease_name, {
            'icon': '⚕️',
            'title': f'Stratégie de Gestion du {disease_name}',
            'actions': [
                '📊 Renforcer la détection et le signalement des cas',
                '🏥 Assurer l\'accès au diagnostic et au traitement',
                '📚 Lancer des campagnes de sensibilisation en santé publique',
                '🔬 Soutenir les initiatives de surveillance et de recherche'
            ]
        })
        
        if stats and province_name:
            # Add risk-stratified recommendations
            total_cases = stats.get('total_cases', 0)
            if total_cases > 10000:
                base_rec['risk_level'] = '🔴 TRÈS ÉLEVÉ'
                base_rec['risk_action'] = '⚡ URGENT: Déclarer d\'un urgence sanitaire. Mobiliser toutes les ressources immédiatement.'
            elif total_cases > 1000:
                base_rec['risk_level'] = '🟠 ÉLEVÉ'
                base_rec['risk_action'] = 'Activer le plan de réponse. Augmenter l\'engagement communautaire et la surveillance.'
            elif total_cases > 100:
                base_rec['risk_level'] = '🟡 MODÉRÉ'
                base_rec['risk_action'] = 'Surveillance renforcée. Préparer les ressources d\'urgence.'
            else:
                base_rec['risk_level'] = '🟢 BAS'
                base_rec['risk_action'] = 'Maintenir la surveillance de routine. Poursuivre les programmes de prévention.'
        
        return base_rec
    
    def _generate_matplotlib_charts(self, disease_name, province_name=None, timeline_data=None, 
                                   provincial_preds=None, zone_dist=None, zone_preds=None, 
                                   pred_data=None, climate_impact=None):
        """Generate beautiful matplotlib charts and return as base64 encoded images"""
        charts = {}
        
        try:
            # 1️⃣ TIMELINE CHART - Show all years with date range
            if timeline_data and timeline_data.get('labels'):
                fig, ax = plt.subplots(figsize=(16, 6), facecolor='#0f172a', edgecolor='none')
                ax.set_facecolor('#1e293b')
                
                labels = timeline_data['labels']
                cases = timeline_data['cases']
                
                # Show date range in title
                date_range = f"{labels[0]} - {labels[-1]}" if labels else "N/A"
                
                ax.fill_between(range(len(labels)), cases, alpha=0.3, color='#ef4444', label='Cas Confirmés')
                ax.plot(range(len(labels)), cases, color='#ef4444', linewidth=3, marker='o', 
                       markersize=5, markerfacecolor='#ef4444', markeredgecolor='white', markeredgewidth=1.5)
                
                ax.set_xlabel('Temps', fontsize=13, color='#e2e8f0', fontweight='bold')
                ax.set_ylabel('Nombre de Cas', fontsize=13, color='#e2e8f0', fontweight='bold')
                ax.set_title(f'Évolution Temporelle - {disease_name} | Période: {date_range}', 
                           fontsize=15, color='#e2e8f0', fontweight='bold', pad=20)
                ax.grid(True, alpha=0.25, color='#64748b', linestyle='--', linewidth=0.8)
                ax.legend(loc='upper left', framealpha=0.9, facecolor='#1e293b', edgecolor='#64748b', fontsize=11)
                
                # Styling
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#64748b')
                ax.spines['bottom'].set_color('#64748b')
                ax.tick_params(colors='#cbd5e1', labelsize=10)
                
                # Every nth label to avoid crowding
                step = max(1, len(labels) // 15)
                ax.set_xticks(range(0, len(labels), step))
                ax.set_xticklabels([labels[i] if i < len(labels) else '' for i in range(0, len(labels), step)], 
                                  rotation=45, ha='right', fontsize=10)
                
                # Add total cases annotation
                max_cases = max(cases) if cases else 0
                ax.text(0.98, 0.97, f'Max: {int(max_cases):,} cas', 
                       transform=ax.transAxes, fontsize=11, color='#60a5fa',
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='#1e293b', alpha=0.8, edgecolor='#60a5fa'))
                
                plt.tight_layout()
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', facecolor='#0f172a', dpi=100, bbox_inches='tight')
                img_buffer.seek(0)
                charts['timeline'] = base64.b64encode(img_buffer.getvalue()).decode()
                plt.close()
        except Exception as e:
            print(f"Timeline chart error: {e}")
        
        try:
            # 2️⃣ PROVINCIAL PREDICTIONS CHART - Timeline predictions by province
            if provincial_preds and provincial_preds.get('labels'):
                fig, ax = plt.subplots(figsize=(15, 7), facecolor='#0f172a', edgecolor='none')
                ax.set_facecolor('#1e293b')
                
                labels = provincial_preds['labels'][:15]
                predicted = provincial_preds['predicted'][:15]
                
                # Sort by predicted cases
                sorted_data = sorted(zip(labels, predicted), key=lambda x: x[1], reverse=True)
                labels_sorted, predicted_sorted = zip(*sorted_data)
                
                # Professional colors with gradient
                colors = ['#ef4444', '#f97316', '#eab308', '#3b82f6', '#6366f1', '#8b5cf6', 
                         '#a855f7', '#ec4899', '#f43f5e', '#14b8a6', '#22c55e', '#06b6d4', 
                         '#0ea5e9', '#6366f1', '#8b5cf6'][:len(labels_sorted)]
                
                bars = ax.barh(labels_sorted, predicted_sorted, color=colors, edgecolor='#1f2937', 
                              linewidth=2, height=0.75)
                
                # Add value labels with commas and percentages
                total = sum(predicted_sorted)
                for i, (bar, val) in enumerate(zip(bars, predicted_sorted)):
                    pct = (val/total)*100 if total > 0 else 0
                    ax.text(val, bar.get_y() + bar.get_height()/2, 
                           f' {int(val):,} ({pct:.1f}%)', 
                           va='center', ha='left', fontsize=10, color='#e2e8f0', fontweight='bold')
                
                ax.set_xlabel('Cas Prédits (Prévisions IA)', fontsize=13, color='#e2e8f0', fontweight='bold')
                ax.set_title(f'Cas Prédits par Province (Top 15) | Total Prévu: {int(total):,} cas', 
                            fontsize=14, color='#e2e8f0', fontweight='bold', pad=20)
                ax.grid(True, alpha=0.2, axis='x', color='#64748b', linestyle='--')
                
                # Styling
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#64748b')
                ax.spines['bottom'].set_color('#64748b')
                ax.tick_params(colors='#cbd5e1', labelsize=11)
                
                # Add legend annotation
                legend_text = 'Modèle: CNN-LSTM Ensemble\nDonnées: 16+ ans | Prédictions: Futures'
                ax.text(0.98, 0.02, legend_text, 
                       transform=ax.transAxes, fontsize=9, color='#cbd5e1',
                       verticalalignment='bottom', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='#1e293b', alpha=0.8, edgecolor='#64748b'),
                       style='italic')
                
                plt.tight_layout()
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', facecolor='#0f172a', dpi=100, bbox_inches='tight')
                img_buffer.seek(0)
                charts['provincial'] = base64.b64encode(img_buffer.getvalue()).decode()
                plt.close()
        except Exception as e:
            print(f"Provincial predictions chart error: {e}")
        
        try:
            # 3️⃣ ZONE DISTRIBUTION CHART - All zones clearly visible
            if zone_dist and zone_dist.get('labels'):
                # Calculate dimensions based on number of zones
                num_zones = len(zone_dist['labels'])
                fig_height = max(10, num_zones * 0.45)  # Dynamic height - plus visible
                
                fig, ax = plt.subplots(figsize=(18, fig_height), facecolor='#0f172a', edgecolor='none')
                ax.set_facecolor('#1e293b')
                
                labels = zone_dist['labels']
                cases = zone_dist['cases']
                
                # Sort by cases
                sorted_data = sorted(zip(labels, cases), key=lambda x: x[1], reverse=True)
                labels_sorted, cases_sorted = zip(*sorted_data)
                
                colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(labels_sorted)))
                
                bars = ax.barh(labels_sorted, cases_sorted, color=colors, edgecolor='#1f2937', 
                              linewidth=1.5, height=0.8)
                
                # Add detailed value labels and percentages
                total = sum(cases_sorted)
                for i, (bar, val) in enumerate(zip(bars, cases_sorted)):
                    pct = (val/total)*100 if total > 0 else 0
                    ax.text(val, bar.get_y() + bar.get_height()/2, 
                           f' {int(val):,} ({pct:.1f}%)', 
                           va='center', ha='left', fontsize=9, color='#e2e8f0', fontweight='bold')
                
                ax.set_xlabel('Nombre de Cas (Historique Complet)', fontsize=12, color='#e2e8f0', fontweight='bold')
                ax.set_title(f'Distribution Géographique par Zone de Santé - {disease_name} | Total: {int(total):,} cas', 
                            fontsize=14, color='#e2e8f0', fontweight='bold', pad=20)
                ax.grid(True, alpha=0.2, axis='x', color='#64748b', linestyle='--')
                
                # Styling
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#64748b')
                ax.spines['bottom'].set_color('#64748b')
                ax.tick_params(colors='#cbd5e1', labelsize=9)
                
                plt.tight_layout()
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', facecolor='#0f172a', dpi=100, bbox_inches='tight')
                img_buffer.seek(0)
                charts['zone_dist'] = base64.b64encode(img_buffer.getvalue()).decode()
                plt.close()
        except Exception as e:
            print(f"Zone distribution chart error: {e}")
        
        try:
            # 4️⃣ ZONE PREDICTIONS CHART - Lignes de Tendance: Données Historiques (Bleu) + Prédictions (Rouge)
            if zone_preds and zone_preds.get('labels'):
                num_zones = len(zone_preds['labels'])
                fig_height = max(12, num_zones * 0.45)
                
                fig, ax = plt.subplots(figsize=(18, fig_height), facecolor='#0f172a', edgecolor='none')
                ax.set_facecolor('#1e293b')
                
                labels = zone_preds['labels']
                predicted = zone_preds['predicted']
                historical = zone_preds.get('historical_avg', [])
                
                # Create individual line plots for each zone showing trend
                x_zones = np.arange(len(labels))
                
                # Plot historical data as blue lines
                if historical and len(historical) == len(predicted):
                    ax.plot(x_zones, historical, color='#60a5fa', linewidth=2.5, marker='o',
                           markersize=5, markerfacecolor='#60a5fa', markeredgecolor='white',
                           markeredgewidth=1, label='Tendance Historique', alpha=0.8, linestyle='-')
                
                # Plot predictions as red lines
                ax.plot(x_zones, predicted, color='#ef4444', linewidth=3.5, marker='s',
                       markersize=6, markerfacecolor='#fca5a5', markeredgecolor='#ffffff',
                       markeredgewidth=1.5, label='Tendance Prédite (IA)', alpha=0.95, linestyle='-')
                
                # Fill between for clarity
                if historical and len(historical) == len(predicted):
                    ax.fill_between(x_zones, historical, predicted, alpha=0.1, color='#3b82f6')
                
                # Add value labels at key points
                step = max(1, len(labels)//8)
                for i in range(0, len(labels), step):
                    if historical and len(historical) > i:
                        ax.text(i, historical[i], f'{int(historical[i]):,}', 
                               ha='center', va='bottom', fontsize=7, color='#60a5fa', fontweight='bold')
                    ax.text(i, predicted[i], f'{int(predicted[i]):,}', 
                           ha='center', va='bottom', fontsize=7, color='#ef4444', fontweight='bold')
                
                ax.set_xlabel('Zones de Santé - Évolution de la Tendance (Province: ' + province_name + ')', 
                             fontsize=13, color='#e2e8f0', fontweight='bold')
                ax.set_ylabel('Nombre de Cas', fontsize=13, color='#e2e8f0', fontweight='bold')
                ax.set_title(f'Tendances: Données Historiques (Bleu) vs Prédictions IA (Rouge) | Province: {province_name}', 
                            fontsize=15, color='#e2e8f0', fontweight='bold', pad=20)
                
                ax.set_xticks(x_zones[::step])
                ax.set_xticklabels(labels[::step], rotation=45, ha='right', fontsize=9)
                ax.grid(True, alpha=0.25, axis='y', color='#64748b', linestyle='--', linewidth=0.8)
                ax.legend(loc='upper right', framealpha=0.95, facecolor='#1e293b', edgecolor='#64748b', fontsize=12, ncol=2)
                
                # Styling
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#64748b')
                ax.spines['bottom'].set_color('#64748b')
                ax.tick_params(colors='#cbd5e1', labelsize=10)
                
                plt.tight_layout()
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', facecolor='#0f172a', dpi=100, bbox_inches='tight')
                img_buffer.seek(0)
                charts['zone_pred'] = base64.b64encode(img_buffer.getvalue()).decode()
                plt.close()
        except Exception as e:
            print(f"Zone predictions chart error: {e}")
        
        try:
            # 5️⃣ PREDICTIONS WITH CONFIDENCE INTERVALS - Données Historiques + Prédictions (Ligne Rouge)
            if pred_data and pred_data.get('labels'):
                fig, ax = plt.subplots(figsize=(18, 7), facecolor='#0f172a', edgecolor='none')
                ax.set_facecolor('#1e293b')
                
                labels = pred_data['labels']
                predicted = pred_data['predicted']
                conf_upper = pred_data.get('confidence_upper', predicted)
                conf_lower = pred_data.get('confidence_lower', predicted)
                historical = pred_data.get('historical', [])  # Données historiques
                
                x_pos = np.arange(len(labels))
                
                # SECTION 1: DONNÉES HISTORIQUES (Bleu clair)
                if historical:
                    historical_x = np.arange(len(historical))
                    ax.plot(historical_x, historical, color='#60a5fa', linewidth=2.5, marker='o',
                           markersize=4, markerfacecolor='#60a5fa', markeredgecolor='white',
                           markeredgewidth=1, label='Données Historiques', alpha=0.8, linestyle='-')
                
                # SECTION 2: POINT DE SÉPARATION (ligne pointillée grise)
                separation_point = len(historical) if historical else 0
                if separation_point > 0:
                    ax.axvline(x=separation_point-0.5, color='#64748b', linestyle='--', 
                              linewidth=2, alpha=0.5, label='Début des Prédictions')
                
                # SECTION 3: PRÉDICTIONS FUTURES EN ROUGE
                pred_x = np.arange(separation_point, separation_point + len(predicted))
                
                # Confidence interval (autour de la ligne rouge)
                ax.fill_between(pred_x, conf_lower, conf_upper, alpha=0.2, color='#ef4444', 
                               label='Intervalle de Confiance 95%')
                
                # Main RED prediction line
                ax.plot(pred_x, predicted, color='#ef4444', linewidth=4, marker='s',
                       markersize=7, markerfacecolor='#fca5a5', markeredgecolor='#ffffff',
                       markeredgewidth=2, label='Prédiction IA (ROUGE)', zorder=5, linestyle='-')
                
                # Add value labels on prediction points (not too many)
                step = max(1, len(predicted)//8)
                for i in range(0, len(predicted), step):
                    ax.text(pred_x[i], predicted[i], f'{int(predicted[i]):,}', 
                           ha='center', va='bottom', fontsize=9, color='#fca5a5', fontweight='bold')
                
                # Configure x-axis
                all_labels = (list(range(len(historical))) if historical else []) + list(labels)
                ax.set_xlabel('Mois (Historique → Prédictions Futures)', fontsize=13, 
                             color='#e2e8f0', fontweight='bold')
                ax.set_ylabel('Nombre de Cas', fontsize=13, color='#e2e8f0', fontweight='bold')
                ax.set_title('Analyse Complète: Données Historiques vs Prédictions IA (Ligne Rouge) | Intervalles de Confiance 95%', 
                            fontsize=14, color='#e2e8f0', fontweight='bold', pad=20)
                
                # Set ticks
                total_x = len(historical) + len(predicted) if historical else len(predicted)
                ax.set_xticks(np.arange(0, total_x, max(1, total_x//15)))
                if historical:
                    tick_labels = list(range(len(historical))) + list(labels[::max(1, len(labels)//7)])
                else:
                    tick_labels = labels[::max(1, len(labels)//15)]
                ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
                
                ax.grid(True, alpha=0.25, axis='y', color='#64748b', linestyle='--', linewidth=0.8)
                ax.legend(loc='upper left', framealpha=0.95, facecolor='#1e293b', 
                         edgecolor='#ef4444', fontsize=11, ncol=2)
                
                # Styling
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#64748b')
                ax.spines['bottom'].set_color('#64748b')
                ax.tick_params(colors='#cbd5e1', labelsize=10)
                
                # Add stats box
                max_pred = max(predicted) if predicted else 0
                min_pred = min(predicted) if predicted else 0
                stats_text = f'Prédictions | Max: {int(max_pred):,} | Min: {int(min_pred):,} | Moyenne: {int(np.mean(predicted)):,}'
                ax.text(0.98, 0.05, stats_text, 
                       transform=ax.transAxes, fontsize=10, color='#fca5a5',
                       verticalalignment='bottom', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='#1e293b', alpha=0.9, edgecolor='#ef4444', linewidth=2))
                
                plt.tight_layout()
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', facecolor='#0f172a', dpi=100, bbox_inches='tight')
                img_buffer.seek(0)
                charts['predictions'] = base64.b64encode(img_buffer.getvalue()).decode()
                plt.close()
        except Exception as e:
            print(f"Predictions chart error: {e}")
        
        try:
            # 6️⃣ CLIMATE CORRELATION & INFLUENCE - Comprendre comment le climat affecte la maladie
            if climate_impact and climate_impact.get('correlations'):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor='#0f172a', edgecolor='none')
                ax1.set_facecolor('#1e293b')
                ax2.set_facecolor('#1e293b')
                
                corr_dict = climate_impact['correlations']
                variables = ['🌡️ Température', '💧 Précipitations', '💨 Humidité']
                correlations = [corr_dict.get('temperature', 0), corr_dict.get('precipitation', 0), 
                              corr_dict.get('humidity', 0)]
                
                # GAUCHE: Barres de corrélation avec couleurs
                colors_corr = []
                for corr in correlations:
                    if corr > 0.5:
                        colors_corr.append('#ef4444')  # Rouge = forte corrélation positive
                    elif corr > 0.3:
                        colors_corr.append('#f97316')  # Orange = corrélation modérée positive
                    elif corr > 0:
                        colors_corr.append('#eab308')  # Jaune = corrélation faible positive
                    elif corr > -0.3:
                        colors_corr.append('#3b82f6')  # Bleu = corrélation faible/négative
                    else:
                        colors_corr.append('#1e40af')  # Bleu foncé = forte corrélation négative
                
                bars = ax1.barh(variables, correlations, color=colors_corr, edgecolor='#1f2937', 
                               linewidth=2.5, height=0.6)
                
                # Add value labels
                for bar, val in zip(bars, correlations):
                    label_color = '#ffffff'
                    ax1.text(val, bar.get_y() + bar.get_height()/2, f'  r={val:.3f}', 
                            va='center', ha='left' if val >= 0 else 'right', fontsize=11, 
                            color=label_color, fontweight='bold')
                
                # Add reference line at r=0
                ax1.axvline(x=0, color='#64748b', linestyle='-', linewidth=2, alpha=0.7)
                ax1.axvline(x=0.3, color='#64748b', linestyle=':', linewidth=1.5, alpha=0.4, label='Seuil significatif')
                ax1.axvline(x=-0.3, color='#64748b', linestyle=':', linewidth=1.5, alpha=0.4)
                
                ax1.set_xlabel('Coefficient de Corrélation de Pearson (r)', fontsize=12, 
                              color='#e2e8f0', fontweight='bold')
                ax1.set_title('Influence des Facteurs Climatiques sur la Maladie', fontsize=13, 
                             color='#e2e8f0', fontweight='bold', pad=15)
                ax1.set_xlim(-1, 1)
                ax1.grid(True, alpha=0.2, axis='x', color='#64748b', linestyle='--')
                ax1.legend(loc='lower right', fontsize=9, framealpha=0.8, facecolor='#1e293b', edgecolor='#64748b')
                
                # Styling left plot
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                ax1.spines['left'].set_color('#64748b')
                ax1.spines['bottom'].set_color('#64748b')
                ax1.tick_params(colors='#cbd5e1', labelsize=10)
                
                # DROITE: Légende scientifique et interprétation détaillée
                ax2.axis('off')
                
                interpretation_text = "INTERPRÉTATION DES CORRÉLATIONS:\n\n"
                
                interp_mapping = {
                    'Température': {
                        'positive': 'Températures plus chaudes = PLUS de cas',
                        'negative': 'Températures plus chaudes = MOINS de cas',
                        'weak': 'Température a peu d\'effet'
                    },
                    'Précipitations': {
                        'positive': 'Plus de pluie = PLUS de cas (vecteurs)',
                        'negative': 'Plus de pluie = MOINS de cas',
                        'weak': 'Les précipitations ont peu d\'effet'
                    },
                    'Humidité': {
                        'positive': 'Humidité plus élevée = PLUS de cas',
                        'negative': 'Humidité plus élevée = MOINS de cas',
                        'weak': 'L\'humidité a peu d\'effet'
                    }
                }
                
                for var, corr in zip(variables, correlations):
                    var_name = var.split(' ')[1]  # Extract name without emoji
                    
                    if abs(corr) > 0.5:
                        strength = "FORTE"
                        emoji = "🔴"
                    elif abs(corr) > 0.3:
                        strength = "MODÉRÉE"
                        emoji = "🟠"
                    elif abs(corr) > 0.1:
                        strength = "FAIBLE"
                        emoji = "🟡"
                    else:
                        strength = "TRÈS FAIBLE/NULLE"
                        emoji = "⚪"
                    
                    effect = 'positive' if corr > 0 else 'negative' if corr < -0.1 else 'weak'
                    direction = "↑ POSITIVE" if corr > 0 else "↓ NÉGATIVE" if corr < -0.1 else "→ NULLE"
                    
                    interpretation_text += f"{emoji} {var_name}:\n"
                    interpretation_text += f"   Force: {strength} ({abs(corr):.3f})\n"
                    interpretation_text += f"   Direction: {direction}\n"
                    interpretation_text += f"   📍 {interp_mapping[var_name].get(effect, 'Pas de lien clair')}\n\n"
                
                interpretation_text += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                interpretation_text += "SEUILS STATISTIQUES:\n"
                interpretation_text += "• |r| > 0.5 → Forte corrélation\n"
                interpretation_text += "• |r| > 0.3 → Corrélation significative\n"
                interpretation_text += "• |r| < 0.1 → Corrélation négligeable\n"
                
                # Add interpretation text
                ax2.text(0.05, 0.95, interpretation_text, transform=ax2.transAxes, 
                        fontsize=9.5, color='#cbd5e1', verticalalignment='top',
                        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#1e293b', 
                        alpha=0.8, edgecolor='#64748b', linewidth=1.5, pad=1))
                
                plt.tight_layout()
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', facecolor='#0f172a', dpi=100, bbox_inches='tight')
                img_buffer.seek(0)
                charts['climate'] = base64.b64encode(img_buffer.getvalue()).decode()
                plt.close()
        except Exception as e:
            print(f"Climate correlation chart error: {e}")
        
        return charts
        """Generate intelligent recommendations based on disease data"""
        recommendations = {
            'Paludisme': {
                'icon': '🦟',
                'title': 'Malaria Control Strategy',
                'actions': [
                    '🪵 Deploy bed nets in high-transmission zones (>100 cases/zone)',
                    '💊 Ensure artemisinin-based combination therapy (ACT) stock levels',
                    '🏥 Strengthen surveillance in border areas and seasonal peaks',
                    '💉 Prioritize pregnant women and children <5 years with preventive treatment'
                ]
            },
            'Choléra': {
                'icon': '💧',
                'title': 'Cholera Outbreak Response',
                'actions': [
                    '🚰 Establish water purification centers in outbreak zones',
                    '🏥 Pre-position CTCs (Cholera Treatment Centers) within 5km of hotspots',
                    '📢 Launch community education on waterborne transmission',
                    '💉 Initiate OCV (Oral Cholera Vaccine) campaigns in high-risk populations'
                ]
            },
            'VIH/SIDA': {
                'icon': '🔴',
                'title': 'HIV/AIDS Prevention & Treatment',
                'actions': [
                    '🧪 Scale up VCT (Voluntary Counseling & Testing) services',
                    '💊 Ensure universal access to antiretroviral therapy (ART)',
                    '👶 Implement PMTCT (Prevention of Mother-to-Child Transmission)',
                    '📚 Strengthen community awareness & stigma reduction programs'
                ]
            },
            'Rougeole': {
                'icon': '💉',
                'title': 'Measles Elimination Campaign',
                'actions': [
                    '📋 Achieve ≥95% vaccination coverage in all catchment areas',
                    '🎯 Conduct catch-up campaigns in previously missed populations',
                    '⚠️ Implement rapid response protocols for confirmed cases',
                    '📊 Strengthen case-based surveillance and lab confirmation'
                ]
            },
            'Hépatite': {
                'icon': '🔬',
                'title': 'Hepatitis Prevention Strategy',
                'actions': [
                    '💉 Universal newborn HBV vaccination in all health facilities',
                    '🧪 Screen all blood donors and implement TPHA protocols',
                    '📚 WASH promotion for HAV/HEV transmission prevention',
                    '🏥 Ensure access to antiviral therapy in tertiary centers'
                ]
            },
            'Coqueluche': {
                'icon': '🫁',
                'title': 'Pertussis Control Program',
                'actions': [
                    '👶 Ensure complete DPT vaccination series in infants',
                    '⚠️ Boost adolescents and adults with Tdap',
                    '📢 Early identification & isolation of suspected cases',
                    '💊 Implement post-exposure prophylaxis (PEP) for contacts'
                ]
            }
        }
        
        base_rec = recommendations.get(disease_name, {
            'icon': '⚕️',
            'title': f'{disease_name} Management Strategy',
            'actions': [
                '📊 Strengthen case detection and reporting',
                '🏥 Ensure access to diagnosis and treatment',
                '📚 Launch public health awareness campaigns',
                '🔬 Support surveillance and research initiatives'
            ]
        })
        
        if stats and province_name:
            # Add risk-stratified recommendations
            total_cases = stats.get('total_cases', 0)
            if total_cases > 10000:
                base_rec['risk_level'] = '🔴 VERY HIGH'
                base_rec['risk_action'] = 'URGENT: Declare health emergency. Mobilize all resources immediately.'
            elif total_cases > 1000:
                base_rec['risk_level'] = '🟠 HIGH'
                base_rec['risk_action'] = 'Activate response plan. Increase community engagement & surveillance.'
            elif total_cases > 100:
                base_rec['risk_level'] = '🟡 MODERATE'
                base_rec['risk_action'] = 'Enhanced surveillance. Prepare contingency resources.'
            else:
                base_rec['risk_level'] = '🟢 LOW'
                base_rec['risk_action'] = 'Maintain routine surveillance. Continue prevention programs.'
        
        return base_rec
