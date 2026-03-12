"""
Province-level analysis views with comprehensive visualizations
"""
from django.http import HttpResponse, JsonResponse
from django.views.generic import TemplateView, DetailView
from django.shortcuts import get_object_or_404
from .models import Province, Territory, HealthZone, ClinicalData, Prediction
from .disease_analyzer import DiseaseAnalyzer
from django.db.models import Sum, Avg, Max, Min, Count, Q, F
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np
import io
import base64
from scipy import stats

# Configure matplotlib for non-interactive backend
plt.switch_backend('Agg')


def generate_graph_base64(fig):
    """Convert matplotlib figure to base64 for inline HTML"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{image_base64}"


class ProvinceAnalysisView(TemplateView):
    """Detailed province analysis with multiple graphs and metrics"""
    template_name = "analytics/province_analysis.html"
    
    def get_context_data(self, province_name=None, **kwargs):
        context = super().get_context_data(**kwargs)
        
        if not province_name:
            province_name = self.request.GET.get('province')
        
        province = get_object_or_404(Province, name=province_name)
        context['province'] = province
        
        # Get all zones for this province
        zones = HealthZone.objects.filter(territory__province=province).order_by('name')
        context['zones'] = zones
        context['zones_count'] = zones.count()
        
        # Get all territories for this province
        territories = Territory.objects.filter(province=province)
        context['territories_count'] = territories.count()
        
        # Clinical data for this province
        clinical_data = ClinicalData.objects.filter(health_zone__territory__province=province)
        context['clinical_records_count'] = clinical_data.count()
        
        # Get diseases for this province
        diseases = clinical_data.values_list('disease_name', flat=True).distinct()
        context['diseases'] = sorted(diseases)
        
        # Generate graphs - OPTIMIZED
        context['graphs'] = {}
        
        # 1. Generate graphs ONLY for top 5 diseases
        top_diseases = clinical_data.values('disease_name').annotate(
            total_cases=Sum('cases_count')
        ).order_by('-total_cases')[:5]
        
        top_disease_names = [d['disease_name'] for d in top_diseases]
        
        for disease in top_disease_names:
            context['graphs'][disease] = {
                'timeseries': self._generate_timeseries_graph(province, disease),
                'baseline': self._generate_baseline_graph(province, disease),
                'quality': self._generate_data_quality_graph(province, disease),
                'zones_affected': self._get_zones_affected(province, disease),
                'stats': self._get_disease_stats(province, disease),
            }
        
        # 2. Correlation analysis
        context['correlation_graph'] = self._generate_correlation_graph(province)
        
        # 3. Data completeness dashboard
        context['data_dashboard'] = self._get_data_completeness_dashboard(province)
        
        # 4. Prediction vs actual data
        context['prediction_graph'] = self._generate_prediction_vs_actual(province)
        
        # 5. Zone-level heatmap
        context['zones_heatmap'] = self._generate_zones_heatmap(province)
        
        # Summary note for users
        context['total_diseases'] = len(diseases)
        context['displayed_diseases'] = len(top_disease_names)
        
        return context
    
    def _generate_timeseries_graph(self, province, disease):
        """Generate time series graph for a specific disease"""
        clinical_data = ClinicalData.objects.filter(
            health_zone__territory__province=province,
            disease_name=disease
        ).order_by('month').values('month', 'cases_count', 'deaths_count')
        
        if not clinical_data.exists():
            return None
        
        df = pd.DataFrame(list(clinical_data))
        df['month'] = pd.to_datetime(df['month'])
        
        fig, ax = plt.subplots(figsize=(12, 5), facecolor='white')
        ax.plot(df['month'], df['cases_count'], linewidth=2.5, color='#ef4444', marker='o', markersize=4, label='Cas confirmés')
        
        # Add deaths on secondary axis if available
        if df['deaths_count'].max() > 0:
            ax2 = ax.twinx()
            ax2.plot(df['month'], df['deaths_count'], linewidth=2, color='#7f1d1d', marker='x', markersize=6, label='Décès', linestyle='--')
            ax2.set_ylabel('Décès', fontweight='bold')
            ax2.grid(True, alpha=0.2)
        
        ax.set_title(f'Évolution temporelle - {disease} en {province.name}', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Mois/Année', fontweight='bold')
        ax.set_ylabel('Nombre de cas', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45, ha='right')
        ax.legend(loc='upper left')
        
        return generate_graph_base64(fig)
    
    def _generate_baseline_graph(self, province, disease):
        """Generate baseline comparison graph"""
        clinical_data = ClinicalData.objects.filter(
            health_zone__territory__province=province,
            disease_name=disease
        ).order_by('month').values('month', 'cases_count')
        
        if clinical_data.count() < 2:
            return None
        
        df = pd.DataFrame(list(clinical_data))
        df['month'] = pd.to_datetime(df['month'])
        df['year'] = df['month'].dt.year
        df['month_day'] = df['month'].dt.dayofyear
        
        fig, ax = plt.subplots(figsize=(12, 5), facecolor='white')
        
        # Calculate baseline (mean) and plot confidence band
        baseline = df['cases_count'].mean()
        std_dev = df['cases_count'].std()
        
        # Plot actual data
        ax.plot(df['month'], df['cases_count'], linewidth=2, color='#0ea5e9', marker='o', label='Données actuelles')
        
        # Add baseline
        ax.axhline(y=baseline, color='#fbbf24', linestyle='--', linewidth=2, label=f'Baseline (moy: {baseline:.0f})')
        
        # Add confidence band
        ax.fill_between(df['month'], baseline - std_dev, baseline + std_dev, alpha=0.2, color='#fbbf24', label='Bande de confiance (±1σ)')
        
        ax.set_title(f'Analyse Baseline - {disease} en {province.name}', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Mois/Année', fontweight='bold')
        ax.set_ylabel('Nombre de cas', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45, ha='right')
        ax.legend()
        
        return generate_graph_base64(fig)
    
    def _generate_data_quality_graph(self, province, disease):
        """Generate data quality and missing data visualization"""
        zones = HealthZone.objects.filter(territory__province=province)
        
        quality_data = []
        for zone in zones:
            clinic_count = ClinicalData.objects.filter(
                health_zone=zone,
                disease_name=disease
            ).count()
            
            # Calculate coverage percentage (12 months = 100% if present)
            coverage = min(100, (clinic_count / 12) * 100)
            quality_data.append({
                'zone': zone.name[:15],  # Shorten name
                'coverage': coverage,
                'records': clinic_count
            })
        
        if not quality_data:
            return None
        
        df_quality = pd.DataFrame(quality_data).sort_values('coverage', ascending=True).tail(15)
        
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        colors = ['#ef4444' if x < 50 else '#fbbf24' if x < 80 else '#10b981' for x in df_quality['coverage']]
        
        bars = ax.barh(df_quality['zone'], df_quality['coverage'], color=colors, height=0.7)
        ax.set_xlabel('Couverture de données (%)', fontweight='bold')
        ax.set_title(f'Qualité des données - {disease} en {province.name} (Top 15 zones)', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlim(0, 100)
        
        # Add percentage labels
        for i, (idx, row) in enumerate(df_quality.iterrows()):
            ax.text(row['coverage'] + 2, i, f"{row['coverage']:.0f}%", va='center', fontsize=9, fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='x')
        
        return generate_graph_base64(fig)
    
    def _generate_correlation_graph(self, province):
        """Generate correlation matrix between diseases and climate factors"""
        # Get all clinical data for province
        all_clinical = ClinicalData.objects.filter(
            health_zone__territory__province=province
        ).values('month', 'disease_name', 'cases_count')
        
        df = pd.DataFrame(list(all_clinical))
        if df.empty or len(df) < 2:
            return None
        
        # Pivot to get diseases as columns
        df_pivot = df.pivot_table(index='month', columns='disease_name', values='cases_count', fill_value=0)
        
        if df_pivot.shape[1] < 2:  # Need at least 2 diseases
            return None
        
        # Calculate correlation
        corr_matrix = df_pivot.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        
        im = ax.imshow(corr_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(corr_matrix.columns)))
        ax.set_yticks(np.arange(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.columns)
        
        # Add correlation values
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold', fontsize=10)
        
        ax.set_title(f'Matrice de corrélation - Maladies en {province.name}', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel('Maladie', fontweight='bold')
        ax.set_xlabel('Maladie', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Coefficient de corrélation', fontweight='bold')
        
        return generate_graph_base64(fig)
    
    def _generate_prediction_vs_actual(self, province):
        """Compare predictions with actual historical data"""
        predictions = Prediction.objects.filter(health_zone__territory__province=province)
        
        if not predictions.exists():
            return None
        
        # Group by disease and get average predictions
        pred_data = predictions.values('disease_name', 'prediction_date').annotate(
            avg_predicted=Avg('predicted_cases')
        ).order_by('disease_name')
        
        # Get actual clinical data for comparison
        clinical_data = ClinicalData.objects.filter(
            health_zone__territory__province=province
        ).values('disease_name').annotate(
            avg_cases=Avg('cases_count')
        ).order_by('disease_name')
        
        diseases = list(set([p['disease_name'] for p in pred_data]))
        
        if not diseases:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 5), facecolor='white')
        
        x = np.arange(len(diseases))
        width = 0.35
        
        pred_values = []
        actual_values = []
        
        for disease in diseases:
            pred_val = predictions.filter(disease_name=disease).aggregate(Avg('predicted_cases'))['predicted_cases__avg'] or 0
            actual_val = ClinicalData.objects.filter(
                health_zone__territory__province=province,
                disease_name=disease
            ).aggregate(Avg('cases_count'))['cases_count__avg'] or 0
            
            pred_values.append(pred_val)
            actual_values.append(actual_val)
        
        bars1 = ax.bar(x - width/2, actual_values, width, label='Données historiques', color='#0ea5e9')
        bars2 = ax.bar(x + width/2, pred_values, width, label='Prédictions', color='#f43f5e')
        
        ax.set_xlabel('Maladies', fontweight='bold')
        ax.set_ylabel('Moyenne de cas', fontweight='bold')
        ax.set_title(f'Comparaison Prédictions vs Données actuelles - {province.name}', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(diseases, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        return generate_graph_base64(fig)
    
    def _generate_zones_heatmap(self, province):
        """Generate heatmap of disease intensity by zone"""
        zones = HealthZone.objects.filter(territory__province=province)
        diseases = ClinicalData.objects.filter(
            health_zone__territory__province=province
        ).values_list('disease_name', flat=True).distinct()
        
        if not zones.exists() or not diseases.exists():
            return None
        
        heatmap_data = []
        zone_names = []
        
        for zone in zones[:20]:  # Top 20 zones
            zone_names.append(zone.name[:12])
            row = []
            for disease in diseases:
                avg_cases = ClinicalData.objects.filter(
                    health_zone=zone,
                    disease_name=disease
                ).aggregate(Avg('cases_count'))['cases_count__avg'] or 0
                row.append(avg_cases)
            heatmap_data.append(row)
        
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
        
        im = ax.imshow(np.array(heatmap_data), cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(np.arange(len(diseases)))
        ax.set_yticks(np.arange(len(zone_names)))
        ax.set_xticklabels(diseases, rotation=45, ha='right')
        ax.set_yticklabels(zone_names)
        
        ax.set_title(f'Intensité des maladies par zone - {province.name}', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Maladie', fontweight='bold')
        ax.set_ylabel('Zone de santé', fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Moyenne de cas', fontweight='bold')
        
        return generate_graph_base64(fig)
    
    def _get_zones_affected(self, province, disease):
        """Get list of zones affected by a disease"""
        zones = HealthZone.objects.filter(
            territory__province=province,
            clinicaldata__disease_name=disease
        ).distinct().values_list('name', flat=True)
        
        return sorted(zones)[:10]  # Top 10
    
    def _get_disease_stats(self, province, disease):
        """Get statistics for a specific disease in province"""
        data = ClinicalData.objects.filter(
            health_zone__territory__province=province,
            disease_name=disease
        ).aggregate(
            total_cases=Sum('cases_count'),
            total_deaths=Sum('deaths_count'),
            avg_cases=Avg('cases_count'),
            max_cases=Max('cases_count'),
            min_cases=Min('cases_count'),
            record_count=Count('id')
        )
        
        # Calculate fatality rate
        fatality_rate = 0
        if data['total_cases'] and data['total_cases'] > 0:
            fatality_rate = (data['total_deaths'] / data['total_cases']) * 100
        
        return {
            'total_cases': data['total_cases'] or 0,
            'total_deaths': data['total_deaths'] or 0,
            'avg_cases': round(data['avg_cases'] or 0, 1),
            'max_cases': data['max_cases'] or 0,
            'fatality_rate': round(fatality_rate, 1),
            'data_points': data['record_count']
        }
    
    def _get_data_completeness_dashboard(self, province):
        """Calculate data completeness metrics"""
        zones = HealthZone.objects.filter(territory__province=province).count()
        diseases = ClinicalData.objects.filter(
            health_zone__territory__province=province
        ).values_list('disease_name', flat=True).distinct().count()
        
        # Calculate time span
        date_range = ClinicalData.objects.filter(
            health_zone__territory__province=province
        ).aggregate(
            min_date=Min('month'),
            max_date=Max('month')
        )
        
        months_span = 0
        if date_range['min_date'] and date_range['max_date']:
            months_span = (date_range['max_date'] - date_range['min_date']).days // 30
        
        total_records = ClinicalData.objects.filter(
            health_zone__territory__province=province
        ).count()
        
        # Expected records = zones * diseases * months
        expected_records = max(zones * diseases * months_span, 1)
        completeness = min(100, (total_records / expected_records) * 100)
        
        return {
            'zones': zones,
            'diseases': diseases,
            'months_span': months_span,
            'total_records': total_records,
            'completeness_percent': round(completeness, 1),
            'data_start': date_range['min_date'].strftime('%b %Y') if date_range['min_date'] else 'N/A',
            'data_end': date_range['max_date'].strftime('%b %Y') if date_range['max_date'] else 'N/A',
        }


class DiseaseDetailView(TemplateView):
    """Disease-specific detailed analysis with Chart.js"""
    template_name = "analytics/disease_detail.html"
    
    def get_context_data(self, disease_name=None, **kwargs):
        context = super().get_context_data(**kwargs)
        
        disease_name = disease_name or self.request.GET.get('disease')
        province_name = self.request.GET.get('province')  # Optional province filter
        context['disease'] = disease_name
        context['province'] = province_name
        
        # Get provinces affected by this disease
        if province_name:
            province = get_object_or_404(Province, name=province_name)
            provinces_affected = Province.objects.filter(pk=province.pk)
        else:
            provinces_affected = Province.objects.filter(
                territories__health_zones__clinicaldata__disease_name=disease_name
            ).distinct()
        
        context['provinces'] = provinces_affected
        context['provinces_count'] = provinces_affected.count()
        
        # Global disease statistics (CASES ONLY - no deaths)
        if province_name:
            global_stats = ClinicalData.objects.filter(
                disease_name=disease_name,
                health_zone__territory__province__name=province_name
            ).aggregate(
                total_cases=Sum('cases_count'),
                affected_zones=Count('health_zone', distinct=True)
            )
        else:
            global_stats = ClinicalData.objects.filter(
                disease_name=disease_name
            ).aggregate(
                total_cases=Sum('cases_count'),
                affected_zones=Count('health_zone', distinct=True)
            )
        
        total_cases = global_stats['total_cases'] or 0
        
        context['global_stats'] = {
            'total_cases': total_cases,
            'affected_zones': global_stats['affected_zones'] or 0,
        }
        
        # Generate Chart.js JSON data
        context['timeline_data_json'] = self._get_timeline_data_json(disease_name, province_name)
        context['province_comparison_json'] = self._get_province_comparison_json(disease_name)
        context['zones_distribution_json'] = self._get_zones_distribution_json(disease_name, province_name)
        context['prediction_data_json'] = self._get_prediction_data_json(disease_name, province_name)
        
        # Province-level analysis (if selected)
        if province_name:
            province_obj = get_object_or_404(Province, name=province_name)
            context['province_analysis'] = {
                'territory_breakdown_json': self._get_territory_breakdown_json(province_obj, disease_name),
                'monthly_trend_json': self._get_monthly_trend_json(province_obj, disease_name)
            }
        else:
            context['province_analysis'] = {
                'territory_breakdown_json': json.dumps({'labels': [], 'cases': []}),
                'monthly_trend_json': json.dumps({'labels': [], 'cases': []})
            }
        
        return context
    
    def _get_timeline_data_json(self, disease_name, province_name=None):
        """Timeline data for Chart.js - CASES ONLY"""
        if province_name:
            data = ClinicalData.objects.filter(
                disease_name=disease_name,
                health_zone__territory__province__name=province_name
            ).values('month').annotate(
                cases=Sum('cases_count')
            ).order_by('month')
        else:
            data = ClinicalData.objects.filter(
                disease_name=disease_name
            ).values('month').annotate(
                cases=Sum('cases_count')
            ).order_by('month')
        
        if not data.exists():
            return json.dumps({'labels': [], 'cases': []})
        
        labels = [str(d['month'].strftime('%b %Y')) for d in data]
        cases = [d['cases'] for d in data]
        
        return json.dumps({'labels': labels, 'cases': cases})
    
    def _get_province_comparison_json(self, disease_name):
        """Province comparison - CASES ONLY"""
        provinces = Province.objects.filter(
            territories__health_zones__clinicaldata__disease_name=disease_name
        ).distinct().annotate(
            total_cases=Sum('territories__health_zones__clinicaldata__cases_count')
        ).order_by('-total_cases')[:15]
        
        names = [p.name for p in provinces]
        cases = [p.total_cases or 0 for p in provinces]
        
        return json.dumps({'labels': names, 'cases': cases})
    
    def _get_zones_distribution_json(self, disease_name, province_name=None):
        """Zones distribution - CASES ONLY"""
        if province_name:
            zones = HealthZone.objects.filter(
                territory__province__name=province_name,
                clinicaldata__disease_name=disease_name
            ).distinct().annotate(
                total_cases=Sum('clinicaldata__cases_count')
            ).order_by('-total_cases')[:20]
        else:
            zones = HealthZone.objects.filter(
                clinicaldata__disease_name=disease_name
            ).distinct().annotate(
                total_cases=Sum('clinicaldata__cases_count')
            ).order_by('-total_cases')[:20]
        
        names = [z.name[:20] for z in zones]
        cases = [z.total_cases or 0 for z in zones]
        
        return json.dumps({'labels': names, 'cases': cases})
    
    def _get_territory_breakdown_json(self, province, disease_name):
        """Territory breakdown within province - CASES ONLY"""
        territories = Territory.objects.filter(
            province=province,
            health_zones__clinicaldata__disease_name=disease_name
        ).distinct().annotate(
            total_cases=Sum('health_zones__clinicaldata__cases_count')
        ).order_by('-total_cases')
        
        names = [t.name for t in territories]
        cases = [t.total_cases or 0 for t in territories]
        
        return json.dumps({'labels': names, 'cases': cases})
    
    def _get_monthly_trend_json(self, province, disease_name):
        """Monthly trend for province - CASES ONLY"""
        data = ClinicalData.objects.filter(
            disease_name=disease_name,
            health_zone__territory__province=province
        ).values('month').annotate(
            cases=Sum('cases_count')
        ).order_by('month')
        
        labels = [d['month'].strftime('%b %Y') for d in data]
        cases = [d['cases'] for d in data]
        
        return json.dumps({'labels': labels, 'cases': cases})
    
    def _get_prediction_data_json(self, disease_name, province_name=None):
        """Get prediction data for disease - CASES ONLY"""
        if province_name:
            predictions = Prediction.objects.filter(
                disease_name=disease_name,
                health_zone__territory__province__name=province_name
            ).order_by('prediction_date')
        else:
            predictions = Prediction.objects.filter(
                disease_name=disease_name
            ).order_by('prediction_date')
        
        if not predictions.exists():
            return json.dumps({'labels': [], 'predicted': [], 'confidence_upper': [], 'confidence_lower': []})
        
        labels = [p.prediction_date.strftime('%b %Y') for p in predictions]
        predicted = [p.predicted_cases or 0 for p in predictions]
        
        # Extract confidence intervals from influence_factors
        confidence_upper = []
        confidence_lower = []
        for p in predictions:
            factors = p.influence_factors or {}
            upper = factors.get('confidence_interval_upper', p.predicted_cases or 0)
            lower = factors.get('confidence_interval_lower', p.predicted_cases or 0)
            confidence_upper.append(upper)
            confidence_lower.append(lower)
        
        return json.dumps({
            'labels': labels,
            'predicted': predicted,
            'confidence_upper': confidence_upper,
            'confidence_lower': confidence_lower
        })
    
    def _generate_global_timeline(self, disease_name, province_name=None):
        """Generate global or provincial timeline for disease"""
        if province_name:
            data = ClinicalData.objects.filter(
                disease_name=disease_name,
                health_zone__territory__province__name=province_name
            ).values('month').annotate(
                cases=Sum('cases_count'),
                deaths=Sum('deaths_count')
            ).order_by('month')
        else:
            data = ClinicalData.objects.filter(
                disease_name=disease_name
            ).values('month').annotate(
                cases=Sum('cases_count'),
                deaths=Sum('deaths_count')
            ).order_by('month')
        
        if not data.exists():
            return None
        
        df = pd.DataFrame(list(data))
        df['month'] = pd.to_datetime(df['month'])
        
        fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')
        ax.fill_between(df['month'], df['cases'], alpha=0.3, color='#ef4444')
        ax.plot(df['month'], df['cases'], linewidth=3, color='#ef4444', marker='o')
        
        ax.set_title(f'Évolution nationale - {disease_name}', fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Mois/Année', fontweight='bold')
        ax.set_ylabel('Nombre total de cas', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45, ha='right')
        
        return generate_graph_base64(fig)
    
    def _generate_province_comparison(self, disease_name):
        """Compare disease burden across provinces"""
        provinces = Province.objects.filter(
            territories__health_zones__clinicaldata__disease_name=disease_name
        ).distinct().annotate(
            total_cases=Sum('territories__health_zones__clinicaldata__cases_count')
        ).order_by('-total_cases')[:12]
        
        prov_names = [p.name for p in provinces]
        prov_cases = [p.total_cases or 0 for p in provinces]
        
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        bars = ax.barh(prov_names, prov_cases, color='#f43f5e')
        
        ax.set_xlabel('Nombre total de cas', fontweight='bold')
        ax.set_title(f'Charge morbide de {disease_name} par province', fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(prov_cases):
            ax.text(v + max(prov_cases)*0.01, i, f'{int(v)}', va='center', fontweight='bold')
        
        return generate_graph_base64(fig)
    
    def _generate_per_capita_graph(self, province, disease_name):
        """Generate per-capita cases graph by zone"""
        zones = HealthZone.objects.filter(
            territory__province=province,
            clinicaldata__disease_name=disease_name
        ).distinct().annotate(
            total_cases=Sum('clinicaldata__cases_count')
        ).order_by('-total_cases')[:15]
        
        if not zones.exists():
            return None
        
        zone_names = [z.name[:15] for z in zones]
        per_capita = []
        
        for zone in zones:
            if zone.population and zone.population > 0:
                pc = (zone.total_cases / zone.population) * 100000
            else:
                pc = 0
            per_capita.append(pc)
        
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
        colors = ['#dc2626' if x > 100 else '#f59e0b' if x > 50 else '#10b981' for x in per_capita]
        bars = ax.barh(zone_names, per_capita, color=colors)
        
        ax.set_xlabel('Cas pour 100,000 habitants', fontweight='bold')
        ax.set_title(f'Taux d\'incidence - {disease_name} dans {province.name}', fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, v in enumerate(per_capita):
            ax.text(v + max(per_capita)*0.01, i, f'{v:.1f}', va='center', fontsize=9, fontweight='bold')
        
        return generate_graph_base64(fig)
    
    def _generate_climate_correlation_graph(self, province, disease_name):
        """Analyze correlation between clinical data and climate factors"""
        
        clinical_climate = ClinicalData.objects.filter(
            health_zone__territory__province=province,
            disease_name=disease_name
        ).select_related('health_zone').values(
            'month', 
            'cases_count', 
            'health_zone__climatedata__temperature',
            'health_zone__climatedata__precipitation',
            'health_zone__climatedata__humidity'
        )
        
        if not clinical_climate.exists():
            return None
        
        df = pd.DataFrame(list(clinical_climate))
        
        if len(df) < 2:
            return None
        
        # Prepare data for correlation
        df['month'] = pd.to_datetime(df['month'])
        df_agg = df.groupby('month').agg({
            'cases_count': 'sum',
            'health_zone__climatedata__temperature': 'mean',
            'health_zone__climatedata__precipitation': 'mean',
            'health_zone__climatedata__humidity': 'mean'
        }).dropna()
        
        if len(df_agg) < 2:
            return None
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), facecolor='white')
        
        # Temperature vs Cases
        ax = axes[0]
        ax2 = ax.twinx()
        ax.plot(df_agg.index, df_agg['cases_count'], color='#ef4444', linewidth=2.5, marker='o', label='Cas')
        ax2.plot(df_agg.index, df_agg['health_zone__climatedata__temperature'], color='#f59e0b', linewidth=2, marker='s', label='Température')
        ax.set_ylabel('Cas', fontweight='bold', color='#ef4444')
        ax2.set_ylabel('Température (°C)', fontweight='bold', color='#f59e0b')
        ax.set_title(f'Corrélation Température - {disease_name}', fontweight='bold')
        ax.grid(True, alpha=0.2)
        
        # Precipitation vs Cases
        ax = axes[1]
        ax2 = ax.twinx()
        ax.plot(df_agg.index, df_agg['cases_count'], color='#ef4444', linewidth=2.5, marker='o', label='Cas')
        ax2.plot(df_agg.index, df_agg['health_zone__climatedata__precipitation'], color='#3b82f6', linewidth=2, marker='s', label='Précipitations')
        ax.set_ylabel('Cas', fontweight='bold', color='#ef4444')
        ax2.set_ylabel('Précipitations (mm)', fontweight='bold', color='#3b82f6')
        ax.set_title(f'Corrélation Précipitations - {disease_name}', fontweight='bold')
        ax.grid(True, alpha=0.2)
        
        # Humidity vs Cases
        ax = axes[2]
        ax2 = ax.twinx()
        ax.plot(df_agg.index, df_agg['cases_count'], color='#ef4444', linewidth=2.5, marker='o', label='Cas')
        if not df_agg['health_zone__climatedata__humidity'].isna().all():
            ax2.plot(df_agg.index, df_agg['health_zone__climatedata__humidity'], color='#8b5cf6', linewidth=2, marker='s', label='Humidité')
            ax2.set_ylabel('Humidité (%)', fontweight='bold', color='#8b5cf6')
        ax.set_ylabel('Cas', fontweight='bold', color='#ef4444')
        ax.set_xlabel('Mois/Année', fontweight='bold')
        ax.set_title(f'Corrélation Humidité - {disease_name}', fontweight='bold')
        ax.grid(True, alpha=0.2)
        
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return generate_graph_base64(fig)
    
    def _generate_zones_distribution_graph(self, province, disease_name):
        """Show disease distribution across health zones"""
        zones_data = HealthZone.objects.filter(
            territory__province=province,
            clinicaldata__disease_name=disease_name
        ).distinct().annotate(
            total_cases=Sum('clinicaldata__cases_count'),
            total_deaths=Sum('clinicaldata__deaths_count')
        ).order_by('-total_cases')[:20]
        
        if not zones_data.exists():
            return None
        
        zone_names = [z.name[:12] for z in zones_data]
        cases = [z.total_cases or 0 for z in zones_data]
        deaths = [z.total_deaths or 0 for z in zones_data]
        
        x = np.arange(len(zone_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')
        
        bars1 = ax.bar(x - width/2, cases, width, label='Cas', color='#ef4444', alpha=0.8)
        bars2 = ax.bar(x + width/2, deaths, width, label='Décès', color='#7f1d1d', alpha=0.8)
        
        ax.set_xlabel('Zone de santé', fontweight='bold')
        ax.set_ylabel('Nombre', fontweight='bold')
        ax.set_title(f'Distribution du {disease_name} par zone - {province.name}', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(zone_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        return generate_graph_base64(fig)
    
    def _generate_system_effectiveness_graph(self, province, disease_name):
        """Show system effectiveness - trends and predictions"""
        # Get actual data
        actual_data = ClinicalData.objects.filter(
            health_zone__territory__province=province,
            disease_name=disease_name
        ).values('month').annotate(
            cases=Sum('cases_count')
        ).order_by('month')
        
        # Get predictions
        predictions = Prediction.objects.filter(
            health_zone__territory__province=province,
            disease_name=disease_name
        ).values('prediction_date').annotate(
            predicted_cases=Avg('predicted_cases')
        ).order_by('prediction_date')
        
        if not actual_data.exists():
            return None
        
        df_actual = pd.DataFrame(list(actual_data))
        df_actual['month'] = pd.to_datetime(df_actual['month'])
        df_actual['type'] = 'Données réelles'
        
        fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')
        
        # Plot actual data
        ax.plot(df_actual['month'], df_actual['cases'], linewidth=3, marker='o', 
                color='#0ea5e9', label='Données réelles', markersize=6)
        ax.fill_between(df_actual['month'], df_actual['cases'], alpha=0.2, color='#0ea5e9')
        
        # Plot predictions if available
        if predictions.exists():
            df_pred = pd.DataFrame(list(predictions))
            df_pred['prediction_date'] = pd.to_datetime(df_pred['prediction_date'])
            ax.plot(df_pred['prediction_date'], df_pred['predicted_cases'], linewidth=2.5, marker='s', 
                    color='#f43f5e', label='Prédictions', markersize=5, linestyle='--', alpha=0.8)
        
        ax.set_title(f'Efficacité du système - {disease_name} en {province.name}\n(Données vs Prédictions)', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Mois/Année', fontweight='bold')
        ax.set_ylabel('Nombre de cas', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45, ha='right')
        
        # Add interpretation text
        if len(df_actual) > 0:
            trend = "↑ HAUSSE" if df_actual['cases'].iloc[-1] > df_actual['cases'].iloc[0] else "↓ BAISSE"
            ax.text(0.02, 0.95, f'Tendance: {trend}', transform=ax.transAxes, 
                   fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return generate_graph_base64(fig)
