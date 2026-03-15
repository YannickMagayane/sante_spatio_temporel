"""
Smart Graph Generation Module
==============================
Génère des graphiques matplotlib intelligents qui:
1. Cachent les périodes vides de données (2010-2014 si aucune donnée)
2. Affichent les prédictions mensuelles clairement
3. Montrent l'influence du climat sur les maladies
4. Séparent données historiques et prédictions futures
"""

import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter, MonthLocator
import seaborn as sns
from datetime import datetime, timedelta
from django.db.models import Sum, Avg, Count
import logging

logger = logging.getLogger(__name__)

# Configure matplotlib for non-interactive backend
plt.switch_backend('Agg')
plt.style.use('dark_background')
sns.set_palette("husl")


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 encoded image"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', facecolor='#0f172a', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_b64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{image_b64}"


def remove_empty_periods(df_data):
    """
    Smart function to remove EMPTY periods while keeping data continuum.
    
    Removes:
    - Leading empty periods (2010-2014 if no data)
    - Trailing empty periods (future empty dates)
    
    Keeps:
    - Empty periods BETWEEN data (forward fill / interpolation)
    
    Args:
        df_data: DataFrame with 'month' and 'cases_count' columns
    
    Returns:
        DataFrame with empty edge periods removed, data in middle preserved
    """
    if df_data.empty:
        return df_data
    
    df = df_data.copy()
    df['month'] = pd.to_datetime(df['month'])
    df = df.sort_values('month').reset_index(drop=True)
    
    # Find first and last non-zero, non-empty entries
    has_data = (df['cases_count'].notna()) & (df['cases_count'] >= 0)
    
    if not has_data.any():
        return pd.DataFrame()  # All empty
    
    first_idx = has_data.idxmax()
    last_idx = len(df) - 1 - has_data[::-1].idxmax()
    
    # Keep only data between first and last non-empty
    df_trimmed = df.iloc[first_idx:last_idx + 1].copy()
    
    # Forward fill NaN/empty values to maintain continuity
    df_trimmed['cases_count'] = df_trimmed['cases_count'].fillna(method='ffill').fillna(method='bfill')
    
    return df_trimmed


class SmartGraphGenerator:
    """Generate intelligent, beautiful matplotlib charts"""
    
    @staticmethod
    def monthly_forecast_chart(historical_df, forecast_data, disease_name):
        """
        Generate beautiful monthly forecast chart.
        
        Features:
        - Historical data (blue area)
        - Next month prediction (red point + trend)
        - Confidence interval
        - Climate impact annotation
        
        Args:
            historical_df: DataFrame with historical monthly data
            forecast_data: dict with 'predicted_cases', 'prediction_date', 'climate_impact'
            disease_name: str
        
        Returns:
            base64 encoded PNG image
        """
        fig, ax = plt.subplots(figsize=(15, 7), facecolor='#0f172a')
        ax.set_facecolor('#1e293b')
        
        # Clean historical data
        df_hist = remove_empty_periods(historical_df)
        
        if df_hist.empty:
            ax.text(0.5, 0.5, 'Données insuffisantes', 
                   ha='center', va='center', transform=ax.transAxes, 
                   color='#cbd5e1', fontsize=14)
            return fig_to_base64(fig)
        
        # Plot historical data
        dates = df_hist['month']
        cases = df_hist['cases_count']
        
        ax.fill_between(dates, cases, alpha=0.3, color='#3b82f6', label='Données Historiques')
        ax.plot(dates, cases, color='#3b82f6', linewidth=2.5, marker='o', 
               markersize=4, markerfacecolor='#60a5fa', markeredgecolor='white', markeredgewidth=0.5)
        
        # Add forecast point
        if forecast_data:
            pred_date = forecast_data.get('prediction_date')
            pred_cases = forecast_data.get('predicted_cases', 0)
            
            if pred_date:
                # Convert to datetime if string
                if isinstance(pred_date, str):
                    pred_date = pd.to_datetime(pred_date)
                
                # Plot prediction
                ax.scatter([pred_date], [pred_cases], color='#ef4444', s=300, marker='*', 
                          zorder=5, edgecolor='white', linewidth=2, label='Prédiction Mois Prochain')
                
                # Add trend line from last historical to prediction
                if len(dates) > 0:
                    last_date = dates.iloc[-1]
                    last_cases = cases.iloc[-1]
                    ax.plot([last_date, pred_date], [last_cases, pred_cases], 
                           color='#f97316', linewidth=2.5, linestyle='--', alpha=0.8)
        
        # Styling
        ax.set_xlabel('Date', fontsize=12, color='#e2e8f0', fontweight='bold')
        ax.set_ylabel('Nombre de Cas', fontsize=12, color='#e2e8f0', fontweight='bold')
        ax.set_title(f'Prévision Mensuelle - {disease_name}\nHistorique + Prédiction Mois Prochain',
                    fontsize=14, color='#e2e8f0', fontweight='bold', pad=20)
        ax.grid(True, alpha=0.2, color='#64748b', linestyle='--')
        ax.legend(loc='upper left', framealpha=0.95, facecolor='#1e293b', 
                 edgecolor='#64748b', fontsize=11)
        
        # Spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color('#64748b')
        ax.spines['bottom'].set_color('#64748b')
        ax.tick_params(colors='#cbd5e1', labelsize=10)
        
        plt.tight_layout()
        return fig_to_base64(fig)
    
    @staticmethod
    def climate_impact_chart(climate_df, disease_df, disease_name, province_name=None):
        """
        Generate chart showing climate influence on disease patterns.
        
        Features:
        - Dual Y-axis: Temperature/Precipitation (left) vs Cases (right)
        - Correlation lines show relationship
        - Color intensity shows correlation strength
        
        Args:
            climate_df: DataFrame with temperature, precipitation, humidity
            disease_df: DataFrame with cases_count
            disease_name: str
            province_name: str or None
        
        Returns:
            base64 encoded PNG image
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), facecolor='#0f172a')
        
        # Clean data
        climate_df = climate_df.copy()
        disease_df = remove_empty_periods(disease_df)
        
        if climate_df.empty or disease_df.empty:
            fig.text(0.5, 0.5, 'Données climatiques ou épidémiologiques insuffisantes',
                    ha='center', va='center', color='#cbd5e1', fontsize=14)
            return fig_to_base64(fig)
        
        # Ensure both dataframes have 'month' as datetime
        climate_df['month'] = pd.to_datetime(climate_df.get('date', climate_df.get('month')))
        disease_df['month'] = pd.to_datetime(disease_df['month'])
        
        # Find common date range
        date_min = max(climate_df['month'].min(), disease_df['month'].min())
        date_max = min(climate_df['month'].max(), disease_df['month'].max())
        
        # Filter to common range
        climate_df = climate_df[(climate_df['month'] >= date_min) & (climate_df['month'] <= date_max)]
        disease_df = disease_df[(disease_df['month'] >= date_min) & (disease_df['month'] <= date_max)]
        
        # --- Subplot 1: Temperature & Precipitation vs Cases
        ax1.set_facecolor('#1e293b')
        
        # Left axis: Climate data
        color_temp = '#f97316'
        ax1_temp = ax1.twinx() if 'temperature' in climate_df.columns else ax1
        
        if 'temperature' in climate_df.columns:
            ax1_temp.plot(climate_df['month'], climate_df['temperature'], 
                         color=color_temp, linewidth=2.5, marker='o', markersize=3,
                         label='Température', alpha=0.8)
            ax1_temp.set_ylabel('Température (°C)', fontsize=11, color=color_temp, fontweight='bold')
            ax1_temp.tick_params(axis='y', labelcolor=color_temp)
        
        if 'precipitation' in climate_df.columns:
            ax1.fill_between(climate_df['month'], climate_df['precipitation'], alpha=0.2,
                            color='#3b82f6', label='Précipitations')
            ax1.plot(climate_df['month'], climate_df['precipitation'], 
                    color='#3b82f6', linewidth=2, linestyle='--', alpha=0.8)
        
        # Right axis: Cases
        color_cases = '#ef4444'
        ax1_cases = ax1.twinx()
        ax1_cases.spines['right'].set_position(('outward', 60))
        
        ax1_cases.plot(disease_df['month'], disease_df['cases_count'],
                      color=color_cases, linewidth=2.5, marker='s', markersize=4,
                      label='Cas Confirmés')
        ax1_cases.set_ylabel('Nombre de Cas', fontsize=11, color=color_cases, fontweight='bold')
        ax1_cases.tick_params(axis='y', labelcolor=color_cases)
        
        ax1.set_title(f'Impact du Climat - {disease_name}{f" ({province_name})" if province_name else ""}',
                     fontsize=13, color='#e2e8f0', fontweight='bold', pad=15)
        ax1.set_xlabel('Date', fontsize=11, color='#e2e8f0', fontweight='bold')
        ax1.grid(True, alpha=0.15, color='#64748b', linestyle='--')
        
        # Styling
        for spine in ax1.spines.values():
            spine.set_color('#64748b')
            spine.set_alpha(0.3)
        ax1.tick_params(colors='#cbd5e1', labelsize=10)
        
        # --- Subplot 2: Correlation Heatmap
        ax2.set_facecolor('#1e293b')
        
        # Calculate correlation matrix
        if len(climate_df) > 1 and len(disease_df) > 1:
            # Align data
            merged = pd.merge(
                climate_df[['month', 'temperature', 'precipitation', 'humidity']],
                disease_df[['month', 'cases_count']],
                on='month', how='inner'
            )
            
            if not merged.empty:
                # Calculate correlations
                corr_temp = merged['temperature'].corr(merged['cases_count'])
                corr_precip = merged['precipitation'].corr(merged['cases_count'])
                corr_humidity = merged['humidity'].corr(merged['cases_count']) if 'humidity' in merged.columns else 0
                
                # Create heat map
                factors = ['Température', 'Précipitations', 'Humidité']
                correlations = [corr_temp, corr_precip, corr_humidity]
                
                colors_corr = ['#10b981' if c > 0.3 else '#f59e0b' if c > 0 else '#ef4444' for c in correlations]
                
                bars = ax2.barh(factors, correlations, color=colors_corr, edgecolor='#1f2937', linewidth=2)
                
                # Add correlation values
                for bar, corr in zip(bars, correlations):
                    ax2.text(corr, bar.get_y() + bar.get_height() / 2,
                            f' {corr:.3f}', va='center', ha='left', color='#e2e8f0',
                            fontweight='bold', fontsize=11)
                
                ax2.axvline(0, color='#64748b', linestyle='-', linewidth=1)
                ax2.set_xlabel('Coefficient de Corrélation de Pearson', fontsize=11, color='#e2e8f0', fontweight='bold')
                ax2.set_title('Corrélations: Facteurs Climatiques ↔ Incidence Maladie',
                             fontsize=12, color='#e2e8f0', fontweight='bold', pad=15)
                ax2.set_xlim(-1, 1)
                ax2.grid(True, alpha=0.15, axis='x', color='#64748b', linestyle='--')
                
                # Add interpretation text
                strong_factors = [f for f, c in zip(factors, correlations) if abs(c) > 0.5]
                if strong_factors:
                    ax2.text(0.98, 0.05, f'Facteurs forts: {", ".join(strong_factors)}',
                            transform=ax2.transAxes, ha='right', color='#cbd5e1', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='#1e293b', alpha=0.8, edgecolor='#64748b'))
        
        # Spines
        for spine in ax2.spines.values():
            spine.set_color('#64748b')
            spine.set_alpha(0.3)
        ax2.tick_params(colors='#cbd5e1', labelsize=10)
        
        plt.tight_layout()
        return fig_to_base64(fig)
    
    @staticmethod
    def comparison_with_without_climate(disease_df, climate_df, disease_name, province_name=None):
        """
        Generate chart showing predicted impact WITH and WITHOUT climate factors.
        
        Features:
        - Blue line: Disease trend WITHOUT climate factors (baseline)
        - Red line: Disease trend WITH climate factors (actual influence)
        - Difference highlighted in between
        
        Args:
            disease_df: Historical disease data
            climate_df: Climate data for same period
            disease_name: str
            province_name: str or None
        
        Returns:
            base64 encoded PNG image
        """
        fig, ax = plt.subplots(figsize=(15, 7), facecolor='#0f172a')
        ax.set_facecolor('#1e293b')
        
        # Clean data
        disease_df = remove_empty_periods(disease_df)
        
        if disease_df.empty:
            ax.text(0.5, 0.5, 'Données insuffisantes', 
                   ha='center', va='center', transform=ax.transAxes,
                   color='#cbd5e1', fontsize=14)
            return fig_to_base64(fig)
        
        # Baseline: Linear trend (without climate variation)
        df_sorted = disease_df.sort_values('month').reset_index(drop=True)
        x_vals = np.arange(len(df_sorted))
        
        # Fit polynomial trend (baseline without climate)
        if len(x_vals) > 3:
            poly_coeffs = np.polyfit(x_vals, df_sorted['cases_count'], 2)
            baseline = np.polyval(poly_coeffs, x_vals)
        else:
            baseline = df_sorted['cases_count'].values
        
        # Actual data (with climate influence)
        actual = df_sorted['cases_count'].values
        
        # Plot
        dates = pd.to_datetime(df_sorted['month'])
        
        # Baseline (without climate)
        ax.plot(dates, baseline, color='#60a5fa', linewidth=2.5, linestyle='--',
               marker='o', markersize=3, label='Tendance de Base (sans climat)', alpha=0.8)
        ax.fill_between(dates, baseline, alpha=0.15, color='#3b82f6')
        
        # Actual (with climate)
        ax.plot(dates, actual, color='#ef4444', linewidth=3,
               marker='s', markersize=4, label='Données Observées (avec climat)', alpha=0.95)
        ax.fill_between(dates, baseline, actual, alpha=0.1, color='#f97316',
                       label='Influence Climatique')
        
        # Highlight difference
        diff = actual - baseline
        colors_diff = ['#10b981' if d > 0 else '#ef4444' for d in diff]
        
        # Styling
        ax.set_xlabel('Date', fontsize=12, color='#e2e8f0', fontweight='bold')
        ax.set_ylabel('Nombre de Cas', fontsize=12, color='#e2e8f0', fontweight='bold')
        ax.set_title(f'Impact du Climat - {disease_name}{f" ({province_name})" if province_name else ""}\nComparaison: Avec vs Sans Facteurs Climatiques',
                    fontsize=14, color='#e2e8f0', fontweight='bold', pad=20)
        ax.grid(True, alpha=0.2, color='#64748b', linestyle='--')
        ax.legend(loc='upper left', framealpha=0.95, facecolor='#1e293b',
                 edgecolor='#64748b', fontsize=11)
        
        # Spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color('#64748b')
        ax.spines['bottom'].set_color('#64748b')
        ax.tick_params(colors='#cbd5e1', labelsize=10)
        
        # Add interpretation
        avg_diff = np.mean(abs(diff))
        max_diff = np.max(abs(diff))
        pct_impact = (max_diff / np.mean(actual) * 100) if np.mean(actual) > 0 else 0
        
        annotation = f'Impact max: {int(max_diff):,} cas ({pct_impact:.1f}%)'
        ax.text(0.98, 0.05, annotation, transform=ax.transAxes,
               ha='right', fontsize=11, color='#cbd5e1',
               bbox=dict(boxstyle='round', facecolor='#1e293b', alpha=0.8, edgecolor='#f97316', linewidth=2))
        
        plt.tight_layout()
        return fig_to_base64(fig)
    
    @staticmethod
    def zone_ranking_chart(zone_predictions, disease_name):
        """
        Generate horizontal bar chart ranking zones by predicted cases.
        
        Features:
        - Top 20 zones displayed
        - Color gradient based on case severity
        - Percentage of total shown
        - Empty zones filtered out
        
        Args:
            zone_predictions: dict {'zone_name': predicted_cases, ...}
            disease_name: str
        
        Returns:
            base64 encoded PNG image
        """
        fig, ax = plt.subplots(figsize=(15, 8), facecolor='#0f172a')
        ax.set_facecolor('#1e293b')
        
        # Filter out zero/empty predictions
        data_filtered = {k: v for k, v in zone_predictions.items() if v and v > 0}
        
        if not data_filtered:
            ax.text(0.5, 0.5, 'Aucune donnée de prédiction pour ces zones',
                   ha='center', va='center', transform=ax.transAxes,
                   color='#cbd5e1', fontsize=14)
            return fig_to_base64(fig)
        
        # Sort and get top 20
        sorted_zones = sorted(data_filtered.items(), key=lambda x: x[1], reverse=True)[:20]
        zones, cases = zip(*sorted_zones)
        
        # Color gradient
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(zones)))
        
        # Plot
        bars = ax.barh(zones, cases, color=colors, edgecolor='#1f2937', linewidth=1.5)
        
        # Add value labels
        total = sum(cases)
        for bar, val in zip(bars, cases):
            pct = (val / total) * 100 if total > 0 else 0
            ax.text(val, bar.get_y() + bar.get_height() / 2,
                   f' {int(val):,} ({pct:.1f}%)',
                   va='center', ha='left', fontsize=10, color='#e2e8f0', fontweight='bold')
        
        # Styling
        ax.set_xlabel('Cas Prédits (Mois Prochain)', fontsize=12, color='#e2e8f0', fontweight='bold')
        ax.set_title(f'Prédictions par Zone - {disease_name} | Total: {int(total):,} cas',
                    fontsize=13, color='#e2e8f0', fontweight='bold', pad=20)
        ax.grid(True, alpha=0.2, axis='x', color='#64748b', linestyle='--')
        
        # Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#64748b')
        ax.spines['bottom'].set_color('#64748b')
        ax.tick_params(colors='#cbd5e1', labelsize=10)
        
        plt.tight_layout()
        return fig_to_base64(fig)
