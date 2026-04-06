from datetime import date
from django.http import JsonResponse, HttpResponseRedirect
from django.views.generic import TemplateView, ListView
from django.urls import reverse
from django.contrib import messages
from .models import HealthZone, ClinicalData, Prediction, TaskRun
from django.db.models import Sum, Avg, Max, F
import os
import json
import logging
import signal
from pathlib import Path
import folium
from folium.plugins import HeatMap
from .intelligence import EpidemicContextualizer
from .task_runner import TaskRunner
from .disease_analyzer import DiseaseAnalyzer
from django.utils import timezone

logger = logging.getLogger(__name__)

class DashboardView(TemplateView):
    template_name = "analytics/dashboard.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Get selected zone if any
        selected_zone_id = self.request.GET.get('selected_zone')
        selected_zone = None

        if selected_zone_id:
            try:
                selected_zone = HealthZone.objects.get(id=selected_zone_id)
                context['selected_zone'] = selected_zone
                context['selected_zone_id'] = selected_zone.id
            except HealthZone.DoesNotExist:
                selected_zone = None
        
        # --- Base de Connaissances par Maladie (Santé Publique Avancée) ---
        knowledge_db = {
            'Paludisme': {
                'recs': [
                    "🦟 MILD : Accélérer la distribution de masse dans les zones à haute pluviométrie.",
                    "💊 CTA : Sécuriser la chaîne d'approvisionnement en artémisinine pour éviter les ruptures de stock.",
                    "🏥 PID : Envisager la pulvérisation intra-domiciliaire dans les zones de forte densité larvaire."
                ],
                'insight': "Moteur NeuralCore : Corrélation spatio-temporelle détectée via Bi-LSTM."
            },
            'Hépatite': {
                'recs': [
                    "💉 VACCIN : Renforcement de la dose de naissance pour l'Hépatite B dans les maternités provinciales.",
                    "🧪 SCREENING : Systématiser le dépistage chez les donneurs de sang et les populations à risque.",
                    "🩺 PROTOCOLE : Mise à jour des protocoles de prise en charge antivirale dans les hôpitaux de référence."
                ],
                'insight': "Observation IA : Cluster de transmission suspecté lié aux pratiques d'injection ou à l'eau (VHE)."
            },
            'Choléra': {
                'recs': [
                    "💧 AQUATAB : Distribution de kits d'urgence de purification d'eau aux ménages situés le long des axes fluviaux.",
                    "🏥 CTC : Pré-positionnement de Centres de Traitement du Choléra à moins de 5km des foyers actifs.",
                    "🧤 PCI : Renforcement strict des mesures de Prévention et Contrôle de l'Infection en milieu hospitalier."
                ],
                'insight': "Modèle CNN : Anomalie thermique détectée sur les rives, favorisant la prolifération du vibrion."
            },
            'Rougeole': {
                'recs': [
                    "💉 RIA : Organisation d'Activités de Vaccination Supplémentaire (AVS) dans les zones à faible couverture (Pev).",
                    "💊 VITAMINE A : Supplémentation immédiate des enfants de 6-59 mois pour réduire la mortalité.",
                    "📡 NOTIFICATION : Activation du système de surveillance communautaire pour tout cas de fièvre et éruption."
                ],
                'insight': "Pattern de Densité : Risque de transmission rapide dans les camps de déplacés et zones péri-urbaines."
            },
            'Malnutrition Aiguë': {
                'recs': [
                    "🥜 PLUMPY'NUT : Déploiement immédiat d'aliments thérapeutiques prêts à l'emploi (ATPE) vers les centres ruraux.",
                    "🏥 UNTA/UNTI : Renforcement des Unités Nutritionnelles Thérapeutiques pour les cas sévères avec complications.",
                    "💧 EAH : Couplage systématique de la prise en charge avec des interventions Eau, Assainissement et Hygiène."
                ],
                'insight': "Modèle ConvLSTM : Superposition avec les zones de récentes sécheresses ou conflits."
            }
        }

        # Filtres
        selected_disease = self.request.GET.get('disease', 'all')
        selected_province = self.request.GET.get('province', 'all')
        
        # Base Querysets
        zones_qs = HealthZone.objects.filter(geometry__contains={'level': '3'})
        preds_qs = Prediction.objects.select_related('health_zone')
        
        if selected_province != 'all':
            zones_qs = zones_qs.filter(territory__province__name=selected_province)
            preds_qs = preds_qs.filter(health_zone__territory__province__name=selected_province)
        
        if selected_disease != 'all':
            preds_qs = preds_qs.filter(disease_name=selected_disease)

        # 1. Statistics Dynamic - Reliabilité, Précision, Propagation
        all_preds = Prediction.objects.all()
        if all_preds.exists():
            reliabilities = [p.influence_factors.get('reliability', 0) for p in all_preds if p.influence_factors]
            spread_rates = [p.propagation_rate for p in all_preds if p.propagation_rate]
            context['avg_reliability'] = round(sum(reliabilities)/len(reliabilities), 1) if reliabilities else 0.0
            context['avg_prec_score'] = round(sum([p.influence_factors.get('r2', 0) for p in all_preds if p.influence_factors])*100/len(all_preds), 1) if all_preds.exists() else 0.0
            context['avg_spread'] = round(sum(spread_rates)/len(spread_rates), 1) if spread_rates else 0.0
            
            # temporal precision based on date range
            dates = all_preds.values_list('prediction_date', flat=True).distinct()
            if dates:
                min_date = min(dates)
                max_date = max(dates)
                delta = (max_date - min_date).days
                context['temp_precision'] = f"{delta if delta > 0 else 1} Jour(s)"
            else:
                context['temp_precision'] = "Analyse..."
        else:
            context['avg_reliability'] = 0.0
            context['avg_spread'] = 0.0
            context['temp_precision'] = "Analyse en cours..."

        # 2. Demographics Dynamic
        all_zones = HealthZone.objects.filter(geometry__contains={'level': '3'})
        if all_zones.exists():
            age_15_list = []
            rural_list = []
            for z in all_zones:
                geom = z.geometry or {}
                if 'age_under_15' in geom: age_15_list.append(geom['age_under_15'])
                if 'rural_percentage' in geom: rural_list.append(geom['rural_percentage'])
            
            context['dem_age_15'] = int(sum(age_15_list)/len(age_15_list)) if age_15_list else 47
            context['dem_rural'] = int(sum(rural_list)/len(rural_list)) if rural_list else 65
        else:
            context['dem_age_15'] = 47
            context['dem_rural'] = 65

        # 2.1 Get Clinical Data Range
        from django.db.models import Min, Max
        clinical_dates = ClinicalData.objects.aggregate(start=Min('month'), end=Max('month'))
        
        all_years = list(ClinicalData.objects.values_list('month__year', flat=True).distinct().order_by('month__year'))
        missing_years_global = []
        if all_years:
            min_y = min(all_years)
            max_y = max(all_years)
            missing_years_global = [y for y in range(min_y, max_y + 1) if y not in all_years]
            context['data_start_year'] = min_y
            context['data_end_year'] = max_y
            context['data_range'] = {
                'start': clinical_dates['start'].strftime('%b %Y') if clinical_dates['start'] else f"Jan {min_y}",
                'end': clinical_dates['end'].strftime('%b %Y') if clinical_dates['end'] else f"Jan {max_y}"
            }
        else:
            context['data_start_year'] = 2020
            context['data_end_year'] = 2026
            context['data_range'] = {'start': 'Jan 2020', 'end': 'Jan 2026'}

        # 3. Stats by Disease
        context['stats'] = Prediction.objects.values('disease_name').annotate(
            cases=Sum('predicted_cases')
        ).order_by('-cases')
        
        from .models import Province, Territory
        context['provinces'] = Province.objects.all().order_by('name')

        # 4. Map Generation
        # Center of DRC
        map_center = [-4.0383, 21.7587] 
        zoom = 5
        
        if selected_province != 'all':
            from django.db.models import Min, Max
            bounds = HealthZone.objects.filter(territory__province__name=selected_province).aggregate(
                min_lat=Min('latitude'), max_lat=Max('latitude'),
                min_lon=Min('longitude'), max_lon=Max('longitude')
            )
            if bounds['min_lat'] is not None:
                # Midpoint for center
                map_center = [(bounds['min_lat'] + bounds['max_lat']) / 2, 
                              (bounds['min_lon'] + bounds['max_lon']) / 2]
                zoom = 7 # Base zoom, will be adjusted by fit_bounds

        m = folium.Map(
            location=map_center, 
            zoom_start=zoom, 
            tiles="cartodbpositron", 
            attr="HealthDRC Intelligence",
            zoom_control=False,
            scrollWheelZoom=False
        )

        if selected_province != 'all' and bounds['min_lat'] is not None:
             m.fit_bounds([[bounds['min_lat'], bounds['min_lon']], [bounds['max_lat'], bounds['max_lon']]])
        
        # 4. Map Logic according to user request
        if selected_disease == 'all':
            # --- POPULATION DENSITY HEATMAP ---
            # Show where people are when no disease is selected
            pop_heat_data = []
            all_zones_pop = HealthZone.objects.filter(geometry__contains={'level': '3'})
            for z in all_zones_pop.iterator():
                if z.latitude and z.longitude and z.population > 0:
                    # Very subtle weight for background population
                    pop_heat_data.append([z.latitude, z.longitude, 0.2])
            
            if pop_heat_data:
                HeatMap(
                    pop_heat_data,
                    name="Densité Population",
                    radius=20,
                    blur=15,
                    min_opacity=0.1,
                    gradient={0.2: 'gray', 0.5: 'blue', 1.0: 'blue'}
                ).add_to(m)
        else:
            # --- DISEASE PROPAGATION HEATMAP WITH CLIMATE INFLUENCE ---
            # Normalized intensity basée sur cases + impact climatique
            disease_heat_data = []
            valid_cases = [p.predicted_cases for p in preds_qs if p.predicted_cases > 0]
            max_cases = max(valid_cases) if valid_cases else 1.0
            
            # Charger données climatiques pour impact
            climate_by_province = {}
            try:
                climate_provinces = verify_climate_data() if use_climate and climate_data else None
                if climate_provinces:
                    for prov_name, climate_df in climate_provinces.items():
                        if not climate_df.empty:
                            latest = climate_df.iloc[-1]
                            climate_by_province[prov_name] = {
                                'temp': latest.get('temperature', 25),
                                'precip': latest.get('precipitation', 100),
                                'humidity': latest.get('relative_humidity', 60)
                            }
            except:
                pass
            
            for p in preds_qs.iterator():
                lat, lon = p.health_zone.latitude, p.health_zone.longitude
                if lat and lon and p.predicted_cases > 0:
                    import math
                    
                    # Calcul de l'intensité basée sur cases
                    intensity = math.log1p(p.predicted_cases) / math.log1p(max_cases)
                    
                    # Ajustement climatique
                    climate_boost = 1.0
                    prov_name = None
                    if p.health_zone.territory and p.health_zone.territory.province:
                        prov_name = p.health_zone.territory.province.name
                    if prov_name and prov_name in climate_by_province:
                        clim = climate_by_province[prov_name]
                        # Boost si conditions favorables (T° 20-28°C + humidité >60% = vecteurs actifs)
                        if 20 <= clim['temp'] <= 28 and clim['humidity'] > 60:
                            climate_boost = 1.3  # +30% pour vecteurs actifs
                        elif 15 <= clim['temp'] <= 32:  # Gamme moins idéale mais acceptable
                            climate_boost = 1.1
                    
                    # Appliquer boost climatique
                    final_intensity = min(1.0, intensity * climate_boost)
                    
                    disease_heat_data.append([lat, lon, final_intensity])
            
            if disease_heat_data:
                HeatMap(
                    disease_heat_data,
                    name=f"Vecteur {selected_disease} (Climat intégré)",
                    radius=35,
                    blur=25,
                    min_opacity=0.5,
                    gradient={0.2: '#0ea5e9', 0.4: '#fbbf24', 0.7: '#ef4444', 1.0: '#7f1d1d'} # Thermal scale
                ).add_to(m)
                
                # Ajouter légende expliquant l'impact climatique
                legend_html = '''
                <div style="position: fixed; 
                     bottom: 50px; left: 50px; width: 280px; background-color: white; 
                     border:2px solid grey; z-index:9999; font-size:11px; padding:10px; border-radius:5px">
                    <p style="margin:0 0 8px 0; font-weight:bold; color:#1e293b;">🌡️ Impact Climatique</p>
                    <div style="line-height:1.6;">
                        <span style="color:#0ea5e9; font-weight:bold;">■</span> Risque faible (T°<15°C ou >32°C)<br>
                        <span style="color:#fbbf24; font-weight:bold;">■</span> Risque modéré (15-20°C ou 28-32°C)<br>
                        <span style="color:#ef4444; font-weight:bold;">■</span> <b>Risque ÉLEVÉ</b> (20-28°C + Humidité >60%)<br>
                        <span style="color:#7f1d1d; font-weight:bold;">■</span> Risque critique (Conditions optimales)
                    </div>
                    <p style="margin:8px 0 0 0; font-size:9px; color:#64748b; font-style:italic;">
                    Les vecteurs (moustiques, etc.) sont plus actifs à 20-28°C avec humidité élevée.
                    </p>
                </div>
                '''
                m.get_root().html.add_child(folium.Element(legend_html))


        # 4.1 ADD KINSHASA CAPITAL MARKER (Home Icon)
        # Coordinates for Kinshasa center
        kin_lat, kin_lon = -4.325, 15.3222
        folium.Marker(
            location=[kin_lat, kin_lon],
            icon=folium.Icon(color='blue', icon='home', prefix='fa'),
            popup="<b>Kinshasa (Capitale)</b><br>Capitale de la RDC."
        ).add_to(m)

        # 5. Province Boundaries & Markers
        if selected_province == 'all':
            adm1_path = "data/raw_hdx/geoBoundaries-COD-ADM1_simplified.geojson"
            if os.path.exists(adm1_path):
                folium.GeoJson(
                    adm1_path,
                    name="Limites Prov.",
                    style_function=lambda x: {'fillColor': '#00000000', 'color': '#fbbf24', 'weight': 1, 'opacity': 0.2}
                ).add_to(m)
        else:
            # When province is selected, show markers - ONLY with predicted_cases > 0
            display_preds = preds_qs.filter(predicted_cases__gt=0).order_by('-predicted_cases')[:150]
            for p in display_preds:
                lat, lon = p.health_zone.latitude, p.health_zone.longitude
                if lat and lon:
                    year = p.prediction_date.year
                    
                    # Filtrer selon la maladie sélectionnée
                    disease_for_hist = selected_disease if selected_disease != 'all' else p.disease_name
                    
                    # Chercher les données historiques UNIQUEMENT pour la maladie sélectionnée
                    hist = ClinicalData.objects.filter(health_zone=p.health_zone, disease_name=disease_for_hist).order_by('month')
                    
                    hist_dates = [h.month for h in hist]
                    
                    if hist_dates:
                        z_start = hist_dates[0].year
                        z_end = hist_dates[-1].year
                        z_years = set(d.year for d in hist_dates)
                        z_missing = [y for y in range(z_start, z_end + 1) if y not in z_years]
                        z_missing_str = f"Manque: {', '.join(map(str, z_missing))}" if z_missing else "Données complètes"
                        z_info = f"<p style='font-size: 8px; color: #94a3b8; font-family: monospace; line-height: 1.2; margin: 0 0 8px 0;'>Données: {z_start} - {z_end}<br>{z_missing_str}<br>Prédiction <b>{year}</b></p>"
                    else:
                        z_info = f"<p style='font-size: 8px; color: #94a3b8; font-family: monospace; line-height: 1.2; margin: 0 0 8px 0;'>Aucune donnée historique pour {disease_for_hist}<br>Prédiction <b>{year}</b></p>"
                    
                    hist = hist.order_by('-month')[:2]
                    hist_rows = "".join([f"<tr><td style='padding:4px; border-bottom:1px solid #eee;'>{h.month.strftime('%b %y')}</td><td style='padding:4px; border-bottom:1px solid #eee;'>{h.disease_name}</td><td style='padding:4px; border-bottom:1px solid #eee;'>{int(h.cases_count)}</td></tr>" for h in hist])
                    
                    # Récupérer le taux de propagation (peut être négatif = baisse, positif = hausse)
                    propagation_rate = p.propagation_rate if p.propagation_rate is not None else 0
                    # ✅ Garder les vraies valeurs (positives = croissance, négatives = décroissance)

                    # Calculer le pourcentage de récurrence (persistance) de la maladie dans cette zone
                    recurrence_data = DiseaseAnalyzer.get_recurrence_percentage(p.health_zone, disease_for_hist)

                    # -- ESTHÉTIQUE PREMIUM : Cercles Pulsants & Couleurs Vibrantes --
                    # Rouge: #f43f5e (Rose-Red), Jaune: #fbbf24 (Amber), Bleu: #38bdf8 (Sky)
                    color_hex = '#f43f5e' if p.predicted_cases > 150 else '#fbbf24' if p.predicted_cases > 50 else '#38bdf8'
                    
                    # Popup HTML Amélioré avec Suggestion en cas de vide
                    if not hist_rows:
                        hist_rows = f"""
                        <tr>
                            <td colspan='2' style='padding:12px; text-align:center; color:#94a3b8; border:1px dashed #e2e8f0; border-radius:8px;'>
                                <span style='font-size:14px;'>🧠</span><br>
                                <i style='font-size:9px;'>Synthèse Prédictive Active (IA).<br>Basé sur le profil {p.disease_name}.</i>
                            </td>
                        </tr>
                        """

                    popup_html = f"""
                    <div style='font-family: "Inter", sans-serif; width: 240px; padding: 0; border-radius: 16px; overflow: hidden; box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1);'>
                        <div style='background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: white; padding: 15px;'>
                            <div style='display: flex; justify-content: space-between; align-items: start;'>
                                <div>
                                    <h4 style='margin: 0; font-size: 14px; font-weight: 900; letter-spacing: -0.025em;'>{p.health_zone.name.upper()}</h4>
                                    <p style='margin: 2px 0 0 0; font-size: 9px; opacity: 0.7;'>Province: {selected_province if selected_province != 'all' else 'RDC'}</p>
                                </div>
                                <span style='background: {color_hex}44; color: {color_hex}; padding: 2px 6px; border-radius: 6px; font-size: 9px; font-weight: 900;'>{p.influence_factors.get('model_used', 'IA')}</span>
                            </div>
                            <div style='margin-top: 10px; font-size: 22px; font-weight: 900; font-family: "JetBrains Mono", monospace;'>
                                {int(p.predicted_cases)} <span style='font-size: 10px; font-weight: 400; opacity: 0.7;'>cas {p.disease_name}</span>
                            </div>
                        </div>
                        <div style='padding: 15px; background: white;'>
                            {z_info}
                            <p style='font-size: 10px; font-weight: 800; color: #64748b; margin: 0 0 8px 0; text-transform: uppercase; letter-spacing: 0.05em;'>📊 Analyse Historique {'⚠️ (Denses Limited)' if p.influence_factors.get('data_sparse') else ''}</p>
                            <table style='width: 100%; border-collapse: collapse;'>{hist_rows}</table>

                            <!-- 🔥 MESSAGE DE RÉCURRENCE (PERSISTANCE) - MIS EN AVANT -->
                            <div style='margin: 12px 0; padding: 12px; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-left: 4px solid #f59e0b; border-radius: 8px; font-size: 10px; color: #92400e; line-height: 1.5;'>
                                <div style='font-weight: 900; margin-bottom: 4px;'>🔄 Récurrence Historique</div>
                                <div><strong>Taux de présence : {recurrence_data['percentage']}%</strong></div>
                                <div style='font-size: 9px; margin-top: 4px;'>
                                    Sur la période {recurrence_data['first_date'].year if recurrence_data['first_date'] else '?'}-{recurrence_data['last_date'].year if recurrence_data['last_date'] else '?'}, la maladie était présente dans <b>{recurrence_data['active_months']} mois</b> sur <b>{recurrence_data['total_months_in_period']} mois</b>.<br>
                                    <i>Données disponibles : {recurrence_data['data_months']} mois ({recurrence_data['completeness']}% de complétude)</i>
                                </div>
                            </div>

                            <div style='margin-top: 15px; padding-top: 15px; border-top: 1px solid #f1f5f9;'>
                                <div style='display: flex; gap: 8px; align-items: center; margin-bottom: 8px;'>
                                    <div style='width: 8px; height: 8px; border-radius: 50%; background: {color_hex};'></div>
                                    <span style='font-size: 10px; font-weight: 700; color: #1e293b;'>Risque: {p.influence_factors.get('attack_angle', 'Modéré')}</span>
                                </div>
                                <div style='display: flex; gap: 8px; align-items: center; margin-bottom: 8px;'>
                                    <span style='font-size: 10px; font-weight: 700; color: #1e293b;'>📈 Propagation: {round(propagation_rate, 1)}%/semaine</span>
                                </div>
                                <div style='background: #f8fafc; padding: 10px; border-radius: 10px; font-size: 9px; color: #475569; line-height: 1.4;'>
                                    🎯 <b>Cible IA :</b> {p.disease_name}. <br>
                                    💡 <b>Action :</b> {knowledge_db.get(p.disease_name, {}).get('recs', ['Intervention standard'])[0]}
                                </div>
                                <div class="flex gap-2 items-center">
                                    <div class="w-6 h-6 rounded-lg bg-brand-500/20 flex items-center justify-center text-[10px]">🧠</div>
                                    <div class="w-6 h-6 rounded-lg bg-emerald-500/20 flex items-center justify-center text-[10px]">🛰️</div>
                                    <template x-if="selectedPred?.data_sparse">
                                        <span class="text-[8px] px-2 py-0.5 bg-amber-500/20 text-amber-400 rounded-lg font-black uppercase">⚠️ Sparse Data</span>
                                    </template>
                                </div>
                            </div>
                        </div>
                    </div>
                    """
                    
                    # Determine if this is the selected zone
                    is_selected = selected_zone and p.health_zone.id == selected_zone.id
                    marker_radius = 14 if is_selected else (10 if p.predicted_cases > 150 else 7)
                    marker_weight = 4 if is_selected else 3
                    marker_color = 'yellow' if is_selected else color_hex

                    circle = folium.CircleMarker(
                        location=[lat, lon],
                        radius=marker_radius,
                        color=marker_color,
                        fill=True,
                        fill_color=color_hex,
                        fill_opacity=0.7 if not is_selected else 0.9,
                        weight=marker_weight,
                        popup=folium.Popup(popup_html, max_width=300),
                        tooltip=f"ALERTE {p.disease_name.upper()}: {int(p.predicted_cases)} cas à {p.health_zone.name}",
                        # Custom attribute for zone ID (will be in layer.options.zoneId)
                        **{'zoneId': p.health_zone.id}
                    )
                    circle.add_to(m)

        # 6. PROPAGATION PATH - Yellow line showing spread from selected zone
        if selected_zone and selected_disease != 'all':
            # Build propagation path connecting zones with similar disease
            selected_zone_preds = Prediction.objects.filter(
                health_zone=selected_zone,
                disease_name=selected_disease
            ).order_by('-predicted_cases')

            if selected_zone_preds.exists():
                # Get the main prediction for selected zone
                main_pred = selected_zone_preds.first()

                # Find connected zones (adjacent territories or similar patterns)
                propagation_zones = Prediction.objects.filter(
                    disease_name=selected_disease,
                    health_zone__latitude__isnull=False,
                    health_zone__longitude__isnull=False
                ).exclude(health_zone=selected_zone)

                # Get ALL zones with predictions for this disease
                # Shows propagation across all provinces (not limited to selected province)
                propagation_zones = propagation_zones.order_by('-predicted_cases')

                # Build coordinates path: start from selected zone, connect to propagation zones
                propagation_coords = []
                if selected_zone.latitude and selected_zone.longitude:
                    propagation_coords.append([selected_zone.latitude, selected_zone.longitude])

                    for pz in propagation_zones:
                        if pz.health_zone.latitude and pz.health_zone.longitude:
                            propagation_coords.append([pz.health_zone.latitude, pz.health_zone.longitude])

                    # Draw the yellow propagation line
                    if len(propagation_coords) > 1:
                        folium.PolyLine(
                            propagation_coords,
                            color='yellow',
                            weight=4,
                            opacity=0.8,
                            dash_array='8, 4',
                            tooltip=f'Propagation path: {selected_zone.name} → Adjacent zones ({selected_disease})',
                            popup=f'<b>Propagation Path</b><br>From: {selected_zone.name}<br>Disease: {selected_disease}<br>Predicted cases: {int(main_pred.predicted_cases) if main_pred else "N/A"}'
                        ).add_to(m)

        # Legend removed from Folium to be placed outside the map (Sidebar) as requested

        # 4.2 VECTEURS DE PROPAGATION (Routes & Axes de Déplacement)
        if selected_disease != 'all' and selected_province != 'all':
            # Light version with high-traffic roads only
            roads_path = "data/raw_hdx/roads_simplified.geojson"  # Use simplified version if available
            
            if not os.path.exists(roads_path):
                # Fallback to main roads file
                roads_path = "data/raw_hdx/democratic-republic-of-congo.geojson"
            
            if os.path.exists(roads_path):
                try:
                    folium.GeoJson(
                        roads_path,
                        name="🔴 Vecteurs de Propagation (Routes & Déplacements)",
                        style_function=lambda x: {
                            'color': '#ef4444',  # Red for propagation vectors
                            'weight': 1.5,
                            'opacity': 0.25,
                            'dashArray': '3, 3'  # Dashed lines to show risk pathways
                        },
                        zoom_on_click=False
                    ).add_to(m)
                except Exception as e:
                    # Silently fail if GeoJSON is too large
                    pass
            


        folium.LayerControl().add_to(m)
        context['map_html'] = m._repr_html_()
        context['selected_disease'] = selected_disease
        context['selected_province'] = selected_province
        
        # 5. Global RDC Statistics Hierarchy - Dynamique depuis la BD
        from .models import Province, Territory
        context['prov_count'] = Province.objects.count()
        context['territory_count'] = Territory.objects.count()
        context['zone_count'] = HealthZone.objects.count()
        
        # Statistique de menace (Filtrée par Province si sélectionnée)
        stats_base = Prediction.objects.all()
        if selected_province != 'all':
            stats_base = stats_base.filter(health_zone__territory__province__name=selected_province)
        
        context['stats'] = stats_base.values('disease_name').annotate(
            cases=Sum('predicted_cases')
        ).order_by('-cases')

        # Propagation moyenne par maladie
        spread_by_disease = {}
        for disease_rec in context['stats']:
            disease_name = disease_rec['disease_name']
            disease_preds = stats_base.filter(disease_name=disease_name)
            rates = [p.propagation_rate for p in disease_preds if p.propagation_rate]
            avg = round(sum(rates)/len(rates), 1) if rates else 0.0
            spread_by_disease[disease_name] = avg
        context['spread_by_disease'] = spread_by_disease

        # Prediction Year
        first_pred = Prediction.objects.order_by('-prediction_date').first()
        context['prediction_year'] = first_pred.prediction_date.year if first_pred else 2026

        # Province-specific recurrence for the selected disease (if disease selected)
        province_recurrence = None
        if selected_disease != 'all' and selected_province != 'all':
            summary = DiseaseAnalyzer.get_disease_recurrence_summary(selected_disease, selected_province)
            province_recurrence = {
                'percentage': summary['avg_recurrence'],
                'zones_count': summary['zones_count']
            }
        context['province_recurrence'] = province_recurrence

        # 1. Spatio-Temporal Data Pattern Analysis
        if context['zone_count'] > 0:
            # Utiliser propagation_rate directement (champ du modèle)
            spread_rates = [p.propagation_rate for p in preds_qs if p.propagation_rate]
            context['avg_spread'] = round(sum(spread_rates)/len(spread_rates), 1) if spread_rates else 0.0
        else:
            context['avg_spread'] = 0.0
        
        high_risk_zones = preds_qs.filter(predicted_cases__gt=100).count()
        total_p = preds_qs.count()
        cluster_density = (high_risk_zones / total_p * 100) if total_p > 0 else 0
        
        # 6. RECOMMANDATIONS SANTÉ PUBLIQUE (IA INTELLIGENTE BASÉE SUR DEEP LEARNING)
        recommendations = []
        cnn_insights = []
        system_anomalies = []
        target_disease = selected_disease if selected_disease != 'all' else (context['stats'][0]['disease_name'] if context['stats'] else "Indéfini")
        
        # --- Détection des Anomalies Critiques Géo-contextuelles ---
        
        if selected_province != 'all':
            hot_zone = preds_qs.order_by('-predicted_cases').first()
            if hot_zone and hot_zone.predicted_cases > 50:
                system_anomalies.append(f"🚩 ÉPICENTRE : La Zone de Santé de {hot_zone.health_zone.name} concentre le risque le plus élevé.")
                
                # Utiliser le moteur d'intelligence avec les données du modèle deep learning
                smart_recs = EpidemicContextualizer.generate_intelligent_recommendations(
                    hot_zone.health_zone,
                    target_disease,
                    hot_zone.predicted_cases,
                    prediction_obj=hot_zone  # Passer la Prediction avec ses influence_factors
                )
                recommendations.extend(smart_recs)
                
                # Insights du modèle
                insight = EpidemicContextualizer.get_smart_insights(
                    hot_zone.health_zone,
                    target_disease,
                    prediction_obj=hot_zone
                )
                cnn_insights.append(insight)
            
            # Alertes contextuelles par bloc
            if "Kinshasa" in selected_province:
                system_anomalies.append("🧬 Hyper-densité urbaine : Risque de transmission accélérée en milieu fermé.")
            elif any(p in selected_province for p in ["Kivu", "Ituri"]):
                system_anomalies.append("⛺ Zone de conflits/déplacements : Instabilité compromettant l'hygiène et la surveillance.")
            elif any(p in selected_province for p in ["Katanga", "Lualaba"]):
                system_anomalies.append("⛏️ Zone minière : Promiscuité accrue sur les sites d'extraction.")
            elif any(p in selected_province for p in ["Equateur", "Tshopo", "Mongala"]):
                system_anomalies.append("🌳 Bassin forestier : Réservoir de zoonoses et maladies à vecteur forestier.")
        
        vitesse_propagation = context.get('avg_spread', 0)
        if vitesse_propagation > 15:
            system_anomalies.append(f"⚠️ Cinétique rapide : Vitesse de {vitesse_propagation}%/semaine (Alerte maximale).")
            
        if cluster_density > 30:
            system_anomalies.append(f"🔥 Densité de clusters : {round(cluster_density)}% des zones en alerte.")

        # Si pas encore de recommandations intelligentes, générer par défaut
        if not recommendations:
            threat_type = EpidemicContextualizer.detect_threat_type(target_disease)
            if threat_type == 'epidemic':
                recommendations.append(f"🦠 ÉPIDÉMIE : {target_disease} est une menace contagieuse. Surveillance étroite requise.")
                recommendations.append("📢 Activation des systèmes de surveillance communautaire et de riposte rapide.")
            elif threat_type == 'nutritional_crisis':
                recommendations.append(f"🌽 CRISE NUTRITIONNELLE : {target_disease} est une menace structurelle (non contagieuse).")
                recommendations.append("🥜 Mobiliser secteur humanitaire + agriculture résiliente + filets de sécurité alimentaire.")
            
            cnn_insights.append(f"🧠 Analyse IA : Surveillance standard activée pour {target_disease}.")

        context['recommendations'] = list(dict.fromkeys(recommendations))
        context['cnn_insights'] = list(dict.fromkeys(cnn_insights))
        context['system_anomalies'] = list(dict.fromkeys(system_anomalies))
        context['target_disease'] = target_disease

        # --- PREPARATION DES DONNÉES POUR CHART.JS (À partir du dernier entraînement) ---
        # Les données viennent directement de la table Prediction (remplie par le script d'entraînement)
        
        # 1. GRAPHIQUE 1 : Pression Virale RÉCENTE — TOP 10 ZONES LES PLUS AFFECTÉES
        if selected_disease != 'all':
            # Focus Maladie: Top 10 ZDS avec prédictions du modèle
            zds_qs = preds_qs.filter(disease_name=selected_disease).order_by('-predicted_cases')[:10]
            labels = [f"{p.health_zone.name}" for p in zds_qs]
            cases = [float(p.predicted_cases) for p in zds_qs]
            # Ajouter le modèle utilisé au label
            models = [f"({(p.influence_factors or {}).get('model_used', 'IA')})" for p in zds_qs]
            labels = [f"{l} {m}" for l, m in zip(labels, models)]
            context['chart_disease_labels'] = json.dumps(labels)
            context['chart_disease_values'] = json.dumps(cases)
            context['chart1_title'] = f"🎯 Top 10 Zones à Risque - {selected_disease}"
            context['chart1_type'] = "bar"
        else:
            # Global: Répartition par maladie avec prédictions
            disease_labels = [s['disease_name'] for s in context['stats']]
            disease_cases = [float(s['cases']) for s in context['stats']]
            context['chart_disease_labels'] = json.dumps(disease_labels)
            context['chart_disease_values'] = json.dumps(disease_cases)
            context['chart1_title'] = "📊 Répartition des Maladies"
            context['chart1_type'] = "doughnut"

        # 2. GRAPHIQUE 2 : Métriques de Performance du Dernier Entraînement
        # ⭐ EXTRACTION DIRECTE : R², Fiabilité, MAE des modèles
        from collections import defaultdict
        model_metrics = defaultdict(lambda: {'r2': [], 'reliability': [], 'mae': [], 'tier': []})
        
        training_qs = Prediction.objects.all().select_related('health_zone')
        if selected_province != 'all':
            training_qs = training_qs.filter(health_zone__territory__province__name=selected_province)
        if selected_disease != 'all':
            training_qs = training_qs.filter(disease_name=selected_disease)
        
        # Extraire les scores du dernier entraînement
        for p in training_qs.iterator():
            inf = p.influence_factors or {}
            model_name = inf.get('model_used', 'Unknown')
            r2 = float(inf.get('r2', 0))
            reliability = float(inf.get('reliability', 50))
            mae = float(inf.get('mae', 0))
            tier = int(inf.get('tier', 4))
            
            model_metrics[model_name]['r2'].append(r2)
            model_metrics[model_name]['reliability'].append(reliability)
            model_metrics[model_name]['mae'].append(mae)
            model_metrics[model_name]['tier'].append(tier)
        
        # Calculer les moyennes par modèle
        m_labels = []
        m_reliability = []
        m_r2 = []
        
        model_order = ['ConvLSTM2D', 'CNN-LSTM', 'Bi-LSTM', 'Persistence']
        for model_name in model_order:
            if model_name in model_metrics and model_metrics[model_name]['r2']:
                m_labels.append(model_name)
                avg_reliability = round(sum(model_metrics[model_name]['reliability']) / len(model_metrics[model_name]['reliability']), 1)
                avg_r2 = round(sum(model_metrics[model_name]['r2']) / len(model_metrics[model_name]['r2']), 3)
                m_reliability.append(float(avg_reliability))
                m_r2.append(float(avg_r2))
        
        # Fallback si aucun modèle
        if not m_labels:
            m_labels = ['Entraînement en cours...']
            m_reliability = [0]
            m_r2 = [0]
        
        context['chart_model_labels'] = json.dumps(m_labels)
        context['chart_model_values'] = json.dumps(m_reliability)
        context['chart_model_r2'] = json.dumps(m_r2)
        context['chart_model_title'] = "🎯 Scores du Dernier Entraînement (R² & Fiabilité %)"

        # 3. GRAPHIQUE 3 : Évolution Temporelle — HISTORIQUE + PRÉDICTIONS RÉCENTES
        from datetime import datetime
        min_date = datetime(2014, 1, 1)
        
        temporal_qs = ClinicalData.objects.filter(month__gte=min_date)
        if selected_province != 'all':
            temporal_qs = temporal_qs.filter(health_zone__territory__province__name=selected_province)
        if selected_disease != 'all':
            temporal_qs = temporal_qs.filter(disease_name=selected_disease)
        
        # Données HISTORIQUES (jusqu'à maintenant)
        temporal_data = temporal_qs.values('month').annotate(total=Sum('cases_count')).order_by('month')
        
        t_labels = [d['month'].strftime('%b %y') for d in temporal_data]
        t_vals = [float(d['total']) for d in temporal_data]
        
        # Données PRÉDICTIONS RÉCENTES (depuis dernier modèle)
        future_qs = Prediction.objects.all()
        if selected_province != 'all':
            future_qs = future_qs.filter(health_zone__territory__province__name=selected_province)
        if selected_disease != 'all':
            future_qs = future_qs.filter(disease_name=selected_disease)
        
        # Grouper les prédictions par mois
        future_by_month = future_qs.values('prediction_date').annotate(
            total=Sum('predicted_cases')
        ).order_by('prediction_date')
        
        # Ajouter les prédictions aux graphiques
        pred_labels = []
        pred_vals = []
        for fp in future_by_month:
            pred_date = fp['prediction_date']
            label = pred_date.strftime('%b %y (IA)')
            pred_labels.append(label)
            pred_vals.append(float(fp['total']))
        
        # Combiner historique + prédictions
        combined_labels = t_labels + pred_labels
        combined_vals = t_vals + pred_vals
        
        context['chart_temp_labels'] = json.dumps(combined_labels)
        context['chart_temp_values'] = json.dumps(combined_vals)
        context['chart_temp_title'] = "📈 Évolution Temporelle : Historique (2014+) + Prédictions IA Récentes"
        
        # 4. GRAPHIQUE 4 : Distribution des Tiers (Architecture des Modèles Entraînés)
        # ⭐ NOUVEAU : Montre quel tier a été utilisé pour chaque prédiction
        tier_dist = {}
        all_preds_tier = Prediction.objects.all()
        if selected_province != 'all':
            all_preds_tier = all_preds_tier.filter(health_zone__territory__province__name=selected_province)
        if selected_disease != 'all':
            all_preds_tier = all_preds_tier.filter(disease_name=selected_disease)
        
        for p in all_preds_tier.iterator():
            tier = (p.influence_factors or {}).get('tier', 4)
            tier_name = {
                1: 'Tier 1 (ConvLSTM)',
                2: 'Tier 2 (CNN-LSTM)',
                3: 'Tier 3 (Bi-LSTM)',
                4: 'Tier 4 (Fallback)'
            }.get(tier, 'Indéfini')
            
            if tier_name not in tier_dist:
                tier_dist[tier_name] = 0
            tier_dist[tier_name] += 1
        
        tier_labels = list(tier_dist.keys()) if tier_dist else ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']
        tier_counts = list(tier_dist.values()) if tier_dist else [0, 0, 0, 0]
        
        context['chart_tier_labels'] = json.dumps(tier_labels)
        context['chart_tier_values'] = json.dumps(tier_counts)
        context['chart_tier_title'] = "🏗️ Architecture Modèles Utilisés (Dernière Prédiction)"
        context['chart_temp_title'] = "📈 Évolution Temporelle : Historique (2014+) + Prédictions IA Récentes"
        
        # 5. GRAPHIQUE 5 : Corrélation Climat-Épidémie ⭐ NOUVEAU
        # Récupérer VRAIES données climatiques depuis ClimateData (ERA5-Land)
        climate_impact_data = {
            'temperature_correlation': 0.0,
            'precipitation_correlation': 0.0,
            'humidity_correlation': 0.0,
            'current_temperature': None,
            'current_precipitation': None,
            'current_humidity': None,
            'climate_explanation': "📊 Analyse climatique en cours..."
        }
        
        try:
            from .models import ClimateData
            from scipy.stats import pearsonr
            import numpy as np
            
            # Base QuerySet pour ClimateData
            climate_qs = ClimateData.objects.all()
            if selected_province != 'all':
                climate_qs = climate_qs.filter(health_zone__territory__province__name=selected_province)
            
            # Récupérer les dernières données climatiques réelles de la BD
            latest_climate = climate_qs.order_by('-month').first()
            if latest_climate:
                # Température déjà en Celsius (pas de conversion)
                temp_celsius = round(latest_climate.temperature, 1)
                climate_impact_data['current_temperature'] = temp_celsius
                # Précipitations déjà en mm
                climate_impact_data['current_precipitation'] = round(latest_climate.precipitation, 1)
                if latest_climate.humidity is not None:
                    climate_impact_data['current_humidity'] = round(latest_climate.humidity, 1)
            
            # Construire les listes pour corrélations Pearson
            # Récupérer tous les couples (température, cas) et (précipitation, cas)
            temp_cases = []
            precip_cases = []
            humidity_cases = []
            
            # Boucle sur les zones de santé avec à la fois ClimateData et Predictions
            zones_with_both = HealthZone.objects.filter(
                climatedata__isnull=False,
                prediction__isnull=False
            ).distinct()
            
            if selected_province != 'all':
                zones_with_both = zones_with_both.filter(territory__province__name=selected_province)
            
            for zone in zones_with_both.iterator():
                # Récupérer tous les mois climatiques pour cette zone
                climate_data = ClimateData.objects.filter(health_zone=zone).order_by('-month')
                
                for clim in climate_data[:36]:  # Derniers 36 mois pour avoir assez de points
                    # ⭐ IMPORTANT: Utiliser ClinicalData (données RÉELLES) et non Predictions (qui sont futures)
                    # ClinicalData couvre l'historique (2010-2025), Predictions couvrent le futur (2026)
                    
                    # Récupérer les DONNÉES RÉELLES pour ce mois/zone (tous les diseases)
                    clinical = ClinicalData.objects.filter(
                        health_zone=zone,
                        month__year=clim.month.year,
                        month__month=clim.month.month
                    )
                    
                    if selected_disease != 'all':
                        clinical = clinical.filter(disease_name=selected_disease)
                    
                    # Summar les cas réels de ce mois
                    total_cases = clinical.aggregate(Sum('cases_count'))['cases_count__sum'] or 0
                    
                    if total_cases > 0:
                        temp_cases.append((clim.temperature, float(total_cases)))
                        precip_cases.append((clim.precipitation, float(total_cases)))
                        if clim.humidity is not None:
                            humidity_cases.append((clim.humidity, float(total_cases)))
            
            # Calculer corrélations Pearson si on a assez de points (min 10)
            if len(temp_cases) >= 10:
                try:
                    temp_vals = np.array([x[0] for x in temp_cases])
                    cases_vals = np.array([x[1] for x in temp_cases])
                    corr_t, _ = pearsonr(temp_vals, cases_vals)
                    climate_impact_data['temperature_correlation'] = round(corr_t, 2)  # Garder le signe
                except:
                    pass

            if len(precip_cases) >= 10:
                try:
                    precip_vals = np.array([x[0] for x in precip_cases])
                    cases_vals = np.array([x[1] for x in precip_cases])
                    corr_p, _ = pearsonr(precip_vals, cases_vals)
                    climate_impact_data['precipitation_correlation'] = round(corr_p, 2)  # Garder le signe
                except:
                    pass

            if len(humidity_cases) >= 10:
                try:
                    humidity_vals = np.array([x[0] for x in humidity_cases])
                    cases_vals = np.array([x[1] for x in humidity_cases])
                    corr_h, _ = pearsonr(humidity_vals, cases_vals)
                    climate_impact_data['humidity_correlation'] = round(corr_h, 2)  # Garder le signe
                except:
                    pass
            
            # Générer explication scientifique basée sur les corrélations RÉELLES (avec signe)
            r_temp = climate_impact_data['temperature_correlation']
            r_precip = climate_impact_data['precipitation_correlation']
            r_humidity = climate_impact_data['humidity_correlation']

            current_temp_str = f"{climate_impact_data['current_temperature']}°C" if climate_impact_data['current_temperature'] is not None else "N/A"
            current_rain_str = f"{climate_impact_data['current_precipitation']}mm" if climate_impact_data['current_precipitation'] is not None else "N/A"

            # Utiliser la magnitude (valeur absolue) pour évaluer la force
            abs_r_temp = abs(r_temp)
            abs_r_precip = abs(r_precip)
            abs_r_humidity = abs(r_humidity)

            if abs_r_temp > 0.4:
                direction = "augmente" if r_temp > 0 else "diminue"
                climate_impact_data['climate_explanation'] = f"🌡️ FORTE corrélation température-épidémie (r={r_temp}). Température actuelle: {current_temp_str}. Quand T° {direction}, cas {direction}nt."
            elif abs_r_precip > 0.4:
                direction = "augmente" if r_precip > 0 else "diminue"
                climate_impact_data['climate_explanation'] = f"🌧️ FORTE corrélation pluie-épidémie (r={r_precip}). Pluies actuelles: {current_rain_str}. Eau stagnante {direction} → vecteurs {direction}nt."
            elif abs_r_humidity > 0.3:
                direction = "augmente" if r_humidity > 0 else "diminue"
                climate_impact_data['climate_explanation'] = f"💨 Humidité significative (r={r_humidity}). Conditions favorables aux vecteurs {direction}nt. Surveiller."
            elif max(abs_r_temp, abs_r_precip, abs_r_humidity) > 0.2:
                # Au moins une corrélation modérée
                climate_impact_data['climate_explanation'] = f"📊 Corrélations modérées: Temp={r_temp}, Pluies={r_precip}, Hum={r_humidity}. Facteurs climatiques partiellement significatifs."
            else:
                climate_impact_data['climate_explanation'] = "⛅ Facteurs climatiques peu corrélés. Autres facteurs (démographie, accès soins) plus déterminants."
        
        except Exception as e:
            # Default: pas assez de données climatiques disponibles
            climate_impact_data['climate_explanation'] = "📊 Données climatiques insuffisantes. Analyse en cours..."
        
        context['climate_impact'] = climate_impact_data
        
        # ⭐ LÉGENDE CLIMATIQUE INTELLIGENTE
        # Génère une légende dynamique basée sur les conditions ACTUELLES
        try:
            current_temp = climate_impact_data.get('current_temperature')
            current_humidity = climate_impact_data.get('current_humidity')
            
            climate_legend = []
            
            # Cas 1: Conditions OPTIMALES pour transmission (T° 20-28°C + Humidité >60%)
            if current_temp and 20 <= current_temp <= 28 and current_humidity and current_humidity > 60:
                climate_legend = [
                    {
                        'icon': '■',
                        'color': '#7f1d1d',
                        'description': f'🔴 RISQUE CRITIQUE (T°={current_temp}°C, H={current_humidity}%) — Conditions OPTIMALES pour vecteurs'
                    },
                    {
                        'icon': '■',
                        'color': '#ef4444',
                        'description': f'🟡 ALERTE ÉLEVÉE (T° 20-28°C, Humidité >60%) — Transmission maximale détectée'
                    },
                    {
                        'icon': '■',
                        'color': '#fbbf24',
                        'description': 'Risque modéré (15-20°C ou 28-32°C, ou Humidité 50-60%)'
                    },
                    {
                        'icon': '■',
                        'color': '#0ea5e9',
                        'description': 'Risque faible (T°<15°C ou >32°C, Humidité<50%)'
                    }
                ]
            
            # Cas 2: T° dans plage modérée
            elif current_temp and 15 <= current_temp < 20 or 28 < current_temp <= 32:
                climate_legend = [
                    {
                        'icon': '■',
                        'color': '#fbbf24',
                        'description': f'🟡 RISQUE MODÉRÉ (T°={current_temp}°C) — Conditions partiellement favorables'
                    },
                    {
                        'icon': '■',
                        'color': '#0ea5e9',
                        'description': 'Risque faible (T°<15°C ou >32°C)'
                    },
                    {
                        'icon': '■',
                        'color': '#ef4444',
                        'description': '🔴 RISQUE CRITIQUE (T° 20-28°C + Humidité >60%)'
                    }
                ]
            
            # Cas 3: T° très basse ou très haute
            elif current_temp and (current_temp < 15 or current_temp > 32):
                climate_legend = [
                    {
                        'icon': '■',
                        'color': '#0ea5e9',
                        'description': f'🟢 RISQUE FAIBLE (T°={current_temp}°C) — Conditions défavorables pour vecteurs'
                    },
                    {
                        'icon': '■',
                        'color': '#fbbf24',
                        'description': 'Risque modéré (15-20°C ou 28-32°C, ou Humidité 50-60%)'
                    },
                    {
                        'icon': '■',
                        'color': '#ef4444',
                        'description': '🔴 RISQUE CRITIQUE (T° 20-28°C + Humidité >60%)'
                    }
                ]
            
            # Fallback: légende standard
            if not climate_legend:
                climate_legend = [
                    {
                        'icon': '■',
                        'color': '#ef4444',
                        'description': '🔴 RISQUE CRITIQUE (T° 20-28°C + Humidité >60%)'
                    },
                    {
                        'icon': '■',
                        'color': '#fbbf24',
                        'description': 'Risque modéré (15-20°C ou 28-32°C, Humidité 50-60%)'
                    },
                    {
                        'icon': '■',
                        'color': '#0ea5e9',
                        'description': 'Risque faible (T°<15°C ou >32°C, Humidité<50%)'
                    }
                ]
            
            context['climate_legend'] = climate_legend
            
        except Exception as e:
            # Fallback: légende par défaut
            context['climate_legend'] = [
                {
                    'icon': '■',
                    'color': '#ef4444',
                    'description': '🔴 RISQUE CRITIQUE — T° 20-28°C + Humidité >60% (optimal pour vecteurs)'
                },
                {
                    'icon': '■',
                    'color': '#fbbf24',
                    'description': '🟡 RISQUE MODÉRÉ — T° 15-20°C ou 28-32°C, Humidité 50-60%'
                },
                {
                    'icon': '■',
                    'color': '#0ea5e9',
                    'description': '🟢 RISQUE FAIBLE — T°<15°C ou >32°C, Humidité<50%'
                }
            ]
        
        return context

def api_zone_detail(request, zone_id):
    """API endpoint to get detailed data for a specific health zone"""
    try:
        zone = HealthZone.objects.get(id=zone_id)

        # Get predictions for this zone
        preds = Prediction.objects.filter(health_zone=zone).select_related('health_zone')

        # Get adjacent zones (simplified: zones in same territory)
        adjacent_zones = HealthZone.objects.filter(
            territory=zone.territory
        ).exclude(id=zone.id)[:10]

        # Build response
        data = {
            'zone_id': zone.id,
            'zone_name': zone.name,
            'territory': zone.territory.name if zone.territory else 'N/A',
            'province': zone.territory.province.name if zone.territory and zone.territory.province else 'N/A',
            'latitude': zone.latitude,
            'longitude': zone.longitude,
            'population': zone.population,
            'predictions': [],
            'adjacent_zones': [],
            'recurrences': []  # Add recurrence data for each disease
        }

        # Calculate recurrence for the zone's main diseases
        disease_names = set(pred.disease_name for pred in preds)
        for disease in disease_names:
            recurrence_data = DiseaseAnalyzer.get_recurrence_percentage(zone, disease)
            data['recurrences'].append({
                'disease': disease,
                'percentage': recurrence_data['percentage'],
                'active_months': recurrence_data['active_months'],
                'total_months': recurrence_data['total_months'],
                'message': recurrence_data['message']
            })

        for pred in preds:
            data['predictions'].append({
                'disease': pred.disease_name,
                'predicted_cases': pred.predicted_cases,
                'propagation_rate': pred.propagation_rate,
                'reliability': pred.influence_factors.get('reliability', 0) if pred.influence_factors else 0,
                'r2': pred.influence_factors.get('r2', 0) if pred.influence_factors else 0,
                'model_used': pred.influence_factors.get('model_used', 'Unknown') if pred.influence_factors else 'Unknown',
                'prediction_date': pred.prediction_date.strftime('%Y-%m-%d') if pred.prediction_date else None
            })

        for adj in adjacent_zones:
            data['adjacent_zones'].append({
                'id': adj.id,
                'name': adj.name,
                'latitude': adj.latitude,
                'longitude': adj.longitude,
                'population': adj.population
            })

        return JsonResponse(data)
    except HealthZone.DoesNotExist:
        return JsonResponse({'error': 'Zone not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def api_zone_propagation_path(request, zone_id):
    """API endpoint to get propagation path for a zone (for yellow line)"""
    try:
        zone = HealthZone.objects.get(id=zone_id)

        # Get predictions to identify propagation direction
        preds = Prediction.objects.filter(health_zone=zone).order_by('-predicted_cases')[:5]

        # Build propagation path: connect to zones with similar disease patterns
        propagation_points = []

        # Start with the selected zone
        if zone.latitude and zone.longitude:
            propagation_points.append([zone.latitude, zone.longitude])

            # Find zones with similar predictions (potential spread destinations)
            if preds.exists():
                main_disease = preds.first().disease_name
                nearby_zones = HealthZone.objects.filter(
                    territory__province=zone.territory.province if zone.territory else None
                ).exclude(id=zone.id).filter(
                    latitude__isnull=False,
                    longitude__isnull=False
                )[:8]

                for adj in nearby_zones:
                    # Check if this zone has predictions for same disease
                    adj_pred = Prediction.objects.filter(
                        health_zone=adj,
                        disease_name=main_disease
                    ).first()

                    if adj_pred:
                        propagation_points.append([adj.latitude, adj.longitude])

        return JsonResponse({
            'zone_name': zone.name,
            'points': propagation_points,
            'disease': main_disease if preds.exists() else None
        })
    except HealthZone.DoesNotExist:
        return JsonResponse({'error': 'Zone not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def api_stats(request):
    from .disease_analyzer import DiseaseAnalyzer

    stats = Prediction.objects.values('disease_name').annotate(
        cases=Sum('predicted_cases'),
        avg_spread=Avg('propagation_rate')
    ).order_by('-cases')
    data = []
    for stat in stats:
        disease_name = stat['disease_name']
        # Calculate recurrence summary for this disease globally
        recurrence_summary = DiseaseAnalyzer.get_disease_recurrence_summary(disease_name)
        data.append({
            'disease_name': disease_name,
            'cases': stat['cases'],
            'avg_spread': round(stat['avg_spread'] or 0, 1),
            'recurrence_percentage': recurrence_summary['avg_recurrence'],
            'recurrence_zones': recurrence_summary['zones_count']
        })
    return JsonResponse(data, safe=False)

def api_predictions(request):
    predictions = Prediction.objects.select_related('health_zone').all()
    data = []

    for p in predictions:
        # Get historical context for this specific zone and disease
        history_qs = ClinicalData.objects.filter(
            health_zone=p.health_zone,
            disease_name=p.disease_name
        ).order_by('-month')

        # Last 2 years (exact years present)
        years_data = history_qs.values('month__year').annotate(total=Sum('cases_count')).order_by('-month__year')[:2]
        # Last 2 months
        months_data = history_qs.values('month').annotate(total=Sum('cases_count')).order_by('-month')[:2]

        data.append({
            "id": p.id,
            "zone_name": p.health_zone.name,
            "zone_id": p.health_zone.id,
            "disease": p.disease_name,
            "cases": p.predicted_cases,
            "lat": p.health_zone.latitude,
            "lon": p.health_zone.longitude,
            "reliability": (p.influence_factors or {}).get('reliability', 85.5),
            "attack_angle": (p.influence_factors or {}).get('attack_angle', "Observation active"),
            "model": (p.influence_factors or {}).get('model_used', "LSTM"),
            "r2": (p.influence_factors or {}).get('r2', 0),
            "mae": (p.influence_factors or {}).get('mae', 0),
            "data_sparse": (p.influence_factors or {}).get('data_sparse', False),
            "history": {
                "years": list(years_data),
                "months": [{"month": m['month'].strftime('%b %Y'), "total": m['total']} for m in months_data]
            }
        })
    return JsonResponse(data, safe=False)

def api_alerts(request):
    return JsonResponse([], safe=False)


# ===== NOUVELLES VIEWS POUR ENTRAÎNEMENT & INGESTION =====

class TaskRunListView(ListView):
    """Affiche l'historique des tâches."""
    model = TaskRun
    template_name = 'analytics/task_list.html'
    context_object_name = 'task_runs'
    paginate_by = 10
    
    def get_queryset(self):
        return TaskRun.objects.all().order_by('-created_at')

def launch_training_view(request):
    """Lance un entraînement COMPLET des modèles."""
    if request.method == 'POST':
        # Force le réentraînement en mettant à jour le flag dans les métadonnées
        metadata_file = Path("/app/models/training_metadata.json")
        
        try:
            # Charger les métadonnées existantes
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            # Mettre le flag force_retrain à True pour forcer un réentraînement complet
            metadata['force_retrain'] = True
            
            # Sauvegarder les métadonnées
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("✓ Flag force_retrain activé - l'entraînement sera complet")
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des métadonnées: {e}")
        
        task_run = TaskRunner.run_task_async('train')
        messages.success(request, f"⏳ RE-ENTRAÎNEMENT complet lancé en arrière-plan (ID: {task_run.id}). Cela peut prendre plusieurs minutes. Consultez l'historique pour voir la progression et les erreurs éventuelles.")
    
    return HttpResponseRedirect(request.META.get('HTTP_REFERER', reverse('analytics:dashboard')))

def launch_ingest_view(request):
    """Lance l'ingestion des données HDX."""
    if request.method == 'POST':
        task_run = TaskRunner.run_task_async('ingest')
        messages.success(request, f"⏳ Ingestion lancée en arrière-plan (ID: {task_run.id}). Consultez l'historique pour voir la progression et les erreurs éventuelles.")
    
    return HttpResponseRedirect(request.META.get('HTTP_REFERER', reverse('analytics:dashboard')))

def launch_ingest_climate_view(request):
    """Lance l'ingestion des données climatiques ERA5-Land."""
    if request.method == 'POST':
        task_run = TaskRunner.run_task_async('ingest_climate')
        messages.success(request, f"⏳ Ingestion climatique lancée en arrière-plan (ID: {task_run.id}). Consultez l'historique pour voir la progression et les erreurs éventuelles.")
    
    return HttpResponseRedirect(request.META.get('HTTP_REFERER', reverse('analytics:dashboard')))

def task_status_api(request, task_id):
    """API JSON pour obtenir le statut d'une tâche (AJAX)."""
    try:
        task_run = TaskRun.objects.get(id=task_id)
        data = {
            'id': task_run.id,
            'task_type': task_run.task_type,
            'status': task_run.status,
            'progress': task_run.progress,
            'process_id': task_run.process_id,  #Pour arrêt
            'started_at': task_run.started_at.isoformat() if task_run.started_at else None,
            'completed_at': task_run.completed_at.isoformat() if task_run.completed_at else None,
            'output_log': task_run.output_log[-500:] if task_run.output_log else '',  # Derniers 500 chars
            'error_log': task_run.error_log[-500:] if task_run.error_log else '',
        }
        return JsonResponse(data)
    except TaskRun.DoesNotExist:
        return JsonResponse({'error': 'Task not found'}, status=404)


def training_performance_api(request):
    """
    API pour récupérer les métriques de performance des modèles d'entraînement
    de façon dynamique (à partir de la base de données)
    """
    try:
        from collections import defaultdict
        import json

        # Récupérer toutes les prédictions avec leurs influence_factors
        preds = Prediction.objects.all().select_related('health_zone')
        province = request.GET.get('province', 'all')
        disease = request.GET.get('disease', 'all')

        # Appliquer les filtres si nécessaire
        if province != 'all':
            preds = preds.filter(health_zone__territory__province__name=province)
        if disease != 'all':
            preds = preds.filter(disease_name=disease)

        # Structure pour accumuler les métriques par modèle
        model_metrics = defaultdict(lambda: {
            'r2': [], 'reliability': [], 'mae': [], 'tier': [], 'diseases': set(), 'zones': set()
        })

        # Parcourir toutes les prédictions
        for p in preds.iterator():
            inf = p.influence_factors or {}
            model_name = inf.get('model_used', 'Unknown')
            r2 = float(inf.get('r2', 0))
            reliability = float(inf.get('reliability', 50))
            mae = float(inf.get('mae', 0))
            tier = int(inf.get('tier', 4))

            model_metrics[model_name]['r2'].append(r2)
            model_metrics[model_name]['reliability'].append(reliability)
            model_metrics[model_name]['mae'].append(mae)
            model_metrics[model_name]['tier'].append(tier)
            model_metrics[model_name]['diseases'].add(p.disease_name)
            model_metrics[model_name]['zones'].add(p.health_zone.name)

        # Préparer les données pour le graphique
        models_data = []
        # Exclure le Persistence (tier 4) pour éviter le fallback
        model_order = ['ConvLSTM2D', 'CNN-LSTM', 'Bi-LSTM']  # Ne pas inclure 'Persistence'

        for model_name in model_order:
            if model_name in model_metrics and model_metrics[model_name]['r2']:
                metrics = model_metrics[model_name]
                avg_r2 = round(sum(metrics['r2']) / len(metrics['r2']) * 100, 1)  # en %
                avg_reliability = round(sum(metrics['reliability']) / len(metrics['reliability']), 1)
                avg_mae = round(sum(metrics['mae']) / len(metrics['mae']), 1)

                models_data.append({
                    'name': model_name,
                    'r2': avg_r2,
                    'reliability': avg_reliability,
                    'mae': avg_mae,
                    'tier': int(metrics['tier'][0]) if metrics['tier'] else 4,
                    'disease_count': len(metrics['diseases']),
                    'zone_count': len(metrics['zones'])
                })

        # Récupérer les statistiques globales
        total_predictions = preds.count()
        avg_spread = 0
        if total_predictions > 0:
            spread_rates = [p.propagation_rate for p in preds if p.propagation_rate]
            avg_spread = round(sum(spread_rates) / len(spread_rates), 1) if spread_rates else 0.0

        # Récupérer les prédictions par mois (pour graphique temporel)
        from django.db.models import Sum
        temporal_data = preds.values('prediction_date__year', 'prediction_date__month').annotate(
            total_cases=Sum('predicted_cases')
        ).order_by('prediction_date__year', 'prediction_date__month')

        temporal_labels = []
        temporal_values = []
        for item in temporal_data:
            label = f"{item['prediction_date__year']}-{item['prediction_date__month']:02d}"
            temporal_labels.append(label)
            temporal_values.append(float(item['total_cases']))

        return JsonResponse({
            'success': True,
            'data': {
                'models': models_data,
                'total_predictions': total_predictions,
                'avg_spread': avg_spread,
                'temporal': {
                    'labels': temporal_labels,
                    'values': temporal_values
                }
            }
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


def zone_recurrence_api(request):
    """
    API pour récupérer les pourcentages de récurrence des maladies par zone
    (pour affichage dynamique sur la carte)
    """
    try:
        from .disease_analyzer import DiseaseAnalyzer
        from django.db.models import Count, Q

        province = request.GET.get('province', 'all')
        disease = request.GET.get('disease', 'all')

        # Requête de base pour les zones
        zones_qs = HealthZone.objects.all()
        if province != 'all':
            zones_qs = zones_qs.filter(territory__province__name=province)

        zones_data = []
        # Limiter aux zones qui ont des données historiques
        zones_with_data = zones_qs.filter(
            clinicaldata__disease_name=disease if disease != 'all' else None
        ).distinct()[:100]  # Limite pour performance

        for zone in zones_with_data:
            # Si une maladie est spécifiée, calculer la récurrence pour cette maladie
            # Sinon, calculer pour toutes les maladies de la zone
            if disease != 'all':
                recurrence = DiseaseAnalyzer.get_recurrence_percentage(zone, disease)
                zones_data.append({
                    'zone_id': zone.id,
                    'zone_name': zone.name,
                    'province': zone.province.name if zone.province else 'N/A',
                    'disease': disease,
                    'recurrence': recurrence['percentage'],
                    'active_months': recurrence['active_months'],
                    'total_months': recurrence['total_months']
                })
            else:
                # Trouver la maladie principale (celle avec le plus de cas)
                main_disease = ClinicalData.objects.filter(
                    health_zone=zone
                ).values('disease_name').annotate(
                    total=Count('id')
                ).order_by('-total').first()
                if main_disease:
                    recurrence = DiseaseAnalyzer.get_recurrence_percentage(zone, main_disease['disease_name'])
                    zones_data.append({
                        'zone_id': zone.id,
                        'zone_name': zone.name,
                        'province': zone.province.name if zone.province else 'N/A',
                        'disease': main_disease['disease_name'],
                        'recurrence': recurrence['percentage'],
                        'active_months': recurrence['active_months'],
                        'total_months': recurrence['total_months']
                    })

        return JsonResponse({
            'success': True,
            'zones': zones_data
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


def pressure_virale_api(request):
    """
    API pour le graphique Pression Virale.
    Retourne le top 10 zones par cas prédits, en privilégiant les modèles robustes (tiers 1-3).
    """
    try:
        disease_filter = request.GET.get('disease', 'all')
        province_filter = request.GET.get('province', 'all')

        # Query de base
        preds = Prediction.objects.all().select_related('health_zone')

        # Appliquer les filtres
        if disease_filter != 'all':
            preds = preds.filter(disease_name=disease_filter)
        if province_filter != 'all':
            preds = preds.filter(health_zone__territory__province__name=province_filter)

        # Récupérer les prédictions triées par cas (décroissant)
        all_preds = list(preds.order_by('-predicted_cases')[:200])

        # Séparer les prédictions robustes (tier 1-3) des fallbacks (tier 4)
        robust_preds = []
        fallback_preds = []

        for p in all_preds:
            tier = (p.influence_factors or {}).get('tier', 4)
            if tier < 4:
                robust_preds.append(p)
            else:
                fallback_preds.append(p)

        # Utiliser les robustes en priorité, sinon compléter avec fallbacks
        # Mais on marque les fallbacks clairement
        final_preds = robust_preds[:10]
        if len(final_preds) < 10:
            # Ajouter des fallbacks pour compléter si nécessaire
            needed = 10 - len(final_preds)
            final_preds.extend(fallback_preds[:needed])

        # Prendre le top 10 final
        top_10 = final_preds[:10]

        data = []
        for p in top_10:
            tier = (p.influence_factors or {}).get('tier', 4)
            model_name = (p.influence_factors or {}).get('model_used', 'Unknown')
            data.append({
                'zone_name': p.health_zone.name,
                'province': p.health_zone.territory.province.name if p.health_zone.territory and p.health_zone.territory.province else 'N/A',
                'cases': float(p.predicted_cases),
                'disease': p.disease_name,
                'model': model_name,
                'tier': tier,
                'reliability': (p.influence_factors or {}).get('reliability', 0),
                'r2': (p.influence_factors or {}).get('r2', 0),
                'is_fallback': tier == 4  # Flag pour le frontend
            })

        return JsonResponse({
            'success': True,
            'data': data,
            'robust_count': len(robust_preds),
            'fallback_count': len(fallback_preds),
            'message': f"Robust: {len(robust_preds)}, Fallback: {len(fallback_preds)}"
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
            'data': []
        }, status=500)


def stop_task_view(request, task_id):
    """Arrête une tâche en cours d'exécution."""
    if request.method == 'POST':
        try:
            task_run = TaskRun.objects.get(id=task_id)

            # Vérifier que la tâche est en cours
            if task_run.status != 'running' or not task_run.process_id:
                messages.warning(request, f"⚠️ La tâche #{task_id} n'est pas en cours d'exécution.")
                return HttpResponseRedirect(request.META.get('HTTP_REFERER', reverse('analytics:task_list')))

            # Envoyer SIGTERM au processus
            try:
                os.kill(task_run.process_id, signal.SIGTERM)
                logger.info(f"✅ Signal d'arrêt envoyé au processus {task_run.process_id} (tâche #{task_id})")
            except ProcessLookupError:
                # Le processus est déjà terminé
                logger.warning(f"Processus {task_run.process_id} déjà terminé")
            except Exception as e:
                logger.error(f"Erreur lors de l'arrêt du processus {task_run.process_id}: {e}")
                messages.error(request, f"❌ Erreur lors de l'arrêt: {e}")
                return HttpResponseRedirect(request.META.get('HTTP_REFERER', reverse('analytics:task_list')))

            # Mettre à jour la tâche
            task_run.status = 'stopped'
            task_run.completed_at = timezone.now()
            task_run.process_id = None
            task_run.save()

            messages.success(request, f"✅ Tâche #{task_id} arrêtée avec succès.")
        except TaskRun.DoesNotExist:
            messages.error(request, f"❌ Tâche #{task_id} introuvable.")
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt de la tâche {task_id}: {e}")
            messages.error(request, f"❌ Erreur: {str(e)}")

    return HttpResponseRedirect(request.META.get('HTTP_REFERER', reverse('analytics:task_list')))
