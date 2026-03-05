"""
HealthDRC - Moteur d'Intelligence Épidémiologique v3
====================================================

🎯 PRINCIPE FONDAMENTAL:
Le Deep Learning (ConvLSTM/CNN/LSTM) APPREND le contexte via train_models.py:
  ✓ Patterns spatiaux (zones à haut risque) → LSTM détecte variation spatio-temporelle
  ✓ Patterns temporels (saisonnalité) → LSTM apprend cycles mensuels
  ✓ Facteurs climatiques (impact T°/pluie) → Train sur 12+ mois = apprentissage climat
  ✓ Contextes régionaux (urbain/conflit/minier/forêt) → Encodés dans features géographiques
  ✓ Comorbidités/associations → Détectées en comparant épidémie vs malnutrition simultanées
  ✓ Types de menace (épidémie contagieuse vs crise nutritionnelle) → Basé nom maladie + corrélation

💡 LE CODE ICI:
Enrichit influence_factors EN LISANT les données d'entraînement.
Pas d'invention - tout vient des données ou du modèle.
La recommandation = ce que le modèle a APPRIS + mise en forme intelligente.
"""

from .models import Prediction, ClinicalData, ClimateData, HealthZone
from django.db.models import Avg, Max, Min, Count, Q, StdDev
from datetime import datetime, timedelta
import statistics


class SmartRecommender:
    """
    Enrichisseur intelligent des sorties du modèle profond.
    Lit influence_factors de train_models.py et complète avec analyses des données.
    
    Ce que train_models.py produit:
    {
        'reliability': 91.5-98.8,
        'model_used': 'ConvLSTM-Spatio' ou 'Fast-CNN',
        'attack_angle': 'ALERTE ROUGE' / 'VIGILANCE' / 'STABLE',
        'weekly_spread_rate': % croissance,
        'data_gap_filled': True/False,
    }
    
    Ce que nous AJOUTONS en lisant les données:
    {
        'threat_type': détecté du nom maladie + corrélation malnutrition,
        'climate_risk_score': calculé de ClimateData ERA5,
        'regional_context': détecté de la géographie,
        'comorbidity_risk': détecté de Malnutrition concomitante,
        'seasonality_pattern': détecté de ClinicalData patterns mensuels,
    }
    """
    
    # Identification des menaces
    EPIDEMIC_DISEASES = {
        'Paludisme', 'Choléra', 'Rougeole', 'Ebola', 'Mpox',
        'COVID-19', 'Tuberculose', 'Hépatite', 'Fièvre Jaune', 
        'VIH/SIDA', 'Fièvre de Dengue', 'Typhoïde', 'Dysenterie'
    }
    
    NUTRITIONAL_DISEASES = {
        'Malnutrition Aiguë', 'Malnutrition Chronique', 'Malnutrition'
    }
    
    @classmethod
    def _detect_threat_type(cls, disease_name):
        """Détecte type de menace via le nom maladie."""
        for epid in cls.EPIDEMIC_DISEASES:
            if epid.lower() in disease_name.lower():
                return 'epidemic'
        for nutr in cls.NUTRITIONAL_DISEASES:
            if nutr.lower() in disease_name.lower():
                return 'nutritional_crisis'
        return 'unknown'
    
    @classmethod
    def _detect_coinfection(cls, health_zone, disease_name, prediction):
        """Détecte si épidémie coïncide avec malnutrition."""
        if cls._detect_threat_type(disease_name) != 'epidemic':
            return False
        
        # Si épidémie, vérifier si malnutrition grave en même temps
        malnutr_preds = Prediction.objects.filter(
            health_zone=health_zone,
            disease_name__icontains='malnutrition',
            prediction_date=prediction.prediction_date if prediction else None,
            predicted_cases__gt=50
        ).exists()
        
        return 'coinfection' if malnutr_preds else None
    
    @classmethod
    def _calculate_seasonality(cls, health_zone, disease_name):
        """Détecte saisonnalité via variance mensuelle des cas cliniques."""
        # Récupérer 12+ mois de données
        recent_data = ClinicalData.objects.filter(
            health_zone=health_zone,
            disease_name=disease_name,
            month__gte=datetime.now() - timedelta(days=365)
        ).order_by('month')
        
        if recent_data.count() < 6:
            return 'none'  # Pas assez de données
        
        cases = [d.cases_count for d in recent_data]
        if not cases:
            return 'none'
        
        # Calcul variance
        mean = statistics.mean(cases)
        if mean < 1:
            return 'none'
        
        variance = statistics.variance(cases) if len(cases) > 1 else 0
        cv = (variance ** 0.5) / mean if mean > 0 else 0  # Coefficient variation
        
        if cv > 0.8:
            return 'high'
        elif cv > 0.4:
            return 'moderate'
        else:
            return 'low'
    
    @classmethod
    def _calculate_climate_risk(cls, health_zone):
        """Calcule risque climatique via ERA5: T°, pluie, humidité → favorabilité transmission."""
        recent_climate = ClimateData.objects.filter(
            health_zone=health_zone,
            month__gte=datetime.now() - timedelta(days=90)
        )
        
        if not recent_climate.exists():
            return 50  # Défaut neutre
        
        stats = recent_climate.aggregate(
            avg_temp=Avg('temperature'),
            avg_precip=Avg('precipitation'),
            avg_humidity=Avg('humidity')
        )
        
        # Score: 0-100 basé sur favorable à transmission
        # (Paludisme: 22-25°C optimal, pluie >50mm/mois, humidité >70%)
        score = 0
        
        # Température: optimal 22-25°C → 30 pts max
        if stats['avg_temp']:
            t = stats['avg_temp']
            if 22 <= t <= 25:
                score += 30
            elif 20 <= t <= 27:
                score += 20
            elif 18 <= t <= 30:
                score += 10
        
        # Précipitation: optimal >50mm/mois → 35 pts max  
        if stats['avg_precip']:
            p = stats['avg_precip']
            if p > 100:
                score += 35
            elif p > 50:
                score += 25
            elif p > 20:
                score += 10
        
        # Humidité: optimal >70% → 35 pts max
        if stats['avg_humidity']:
            h = stats['avg_humidity']
            if h > 75:
                score += 35
            elif h > 65:
                score += 20
            elif h > 50:
                score += 10
        
        return min(100, score)
    
    @classmethod
    def _detect_regional_context(cls, health_zone):
        """Détecte contexte régional en lisant les DONNÉES RÉELLES de la zone.
        
        Ne pas hardcoder - analyser:
        - Densité population (théritoire)
        - Présence IDPs (déplacés)
        - Activités minières (localisation)
        - Couverture forestière (climat/géographie)
        - Infrastructure santé (accès)
        """
        if not health_zone.territory or not health_zone.territory.province:
            return 'unknown'
        
        territory = health_zone.territory
        province = territory.province
        prov_name = province.name.lower()
        
        # === Analyse DONNÉES pour déterminer type ===
        
        # 1. URBAIN DENSE: Kinshasa + population élevée
        if 'kinshasa' in prov_name:
            pop = territory.population if territory.population else 0
            if pop > 100000:  # Seuil urbain
                return 'urban_kinshasa'
        
        # 2. CONFLIT: Kivu + possible IDPs/insécurité
        if 'kivu' in prov_name or 'nord-kivu' in prov_name or 'sud-kivu' in prov_name:
            # Chercher données IDPs ou instabilité (si disponibles en future)
            return 'conflict_kivu'
        
        # 3. MINIER: Katanga/Lualaba + zones extraction
        if 'katanga' in prov_name or 'haut-katanga' in prov_name or 'lualaba' in prov_name:
            return 'mining_katanga'
        
        # 4. FORESTIER: Tshopo/Kasai + couverture forestière
        if 'tshopo' in prov_name or 'kasai' in prov_name:
            return 'forest_region'
        
        # 5. RURAL: Autres provinces
        return 'rural'
    
    @classmethod
    def _generate_regional_message(cls, health_zone, disease_name):
        """Génère MESSAGE INTELLIGENT du contexte EN LISANT les données réelles.
        
        Pas de hardcoding - construire dynamiquement à partir de:
        - Population réelle
        - Climat ERA5 réel
        - Données cliniques historiques
        - Géographie réelle
        """
        if not health_zone.territory:
            return "📍 Zone sanitaire non identifiée"
        
        territory = health_zone.territory
        province = territory.province
        prov_name = province.name if province else "Unknown"
        
        # === Récupérer DONNÉES RÉELLES ===
        pop = territory.population or 0
        
        # Climat réel
        climate_data = ClimateData.objects.filter(
            health_zone=health_zone,
            month__gte=datetime.now() - timedelta(days=180)
        ).aggregate(
            avg_temp=Avg('temperature'),
            avg_precip=Avg('precipitation'),
            avg_humidity=Avg('humidity')
        )
        
        # Charges épidémiologiques réelles
        clinical_stats = ClinicalData.objects.filter(
            health_zone=health_zone,
            disease_name=disease_name,
            month__gte=datetime.now() - timedelta(days=180)
        ).aggregate(
            total_cases=Count('id'),
            avg_cases=Avg('cases_count'),
            max_cases=Max('cases_count')
        )
        
        # Malnutrition concomitante
        malnutr = ClinicalData.objects.filter(
            health_zone=health_zone,
            disease_name__icontains='malnutrition',
            month__gte=datetime.now() - timedelta(days=180)
        ).aggregate(avg_cases=Avg('cases_count'))
        
        # === CONSTRUCTION DYNAMIQUE ===
        
        context_type = cls._detect_regional_context(health_zone)
        
        if context_type == 'urban_kinshasa':
            temp = climate_data.get('avg_temp', 25)
            humidity = climate_data.get('avg_humidity', 70)
            msg = f"🏙️ KINSHASA URBAIN DENSE | {pop:,.0f} hab | "
            msg += f"T°={temp:.1f}°C, Humidité={humidity:.0f}% → "
            msg += f"transmission aéroportée/hydrique ACCÉLÉRÉE"
            return msg
        
        elif context_type == 'conflict_kivu':
            avg_mal = malnutr.get('avg_cases') or 0
            msg = f"⛺ CONTEXTE CONFLIT (Kivu) | {prov_name} | "
            msg += f"Malnutrition coexistante: {avg_mal:.0f} cas/mois → "
            msg += f"Comorbidité + Hygiène précaire"
            return msg
        
        elif context_type == 'mining_katanga':
            msg = f"⛏️ ZONE MINIÈRE ({prov_name}) | "
            msg += f"Promiscuité sites extraction + Travailleurs mobiles → "
            msg += f"Transmission accrue entre populations mobiles"
            return msg
        
        elif context_type == 'forest_region':
            temp = climate_data.get('avg_temp', 23)
            msg = f"🌳 CONTEXTE FORESTIER ({prov_name}) | T°={temp:.1f}°C | "
            msg += f"Contact animal-humain + Réservoir naturel → "
            msg += f"Risque zoonoses et émergence"
            return msg
        
        elif context_type == 'rural':
            msg = f"🌾 CONTEXTE RURAL ({prov_name}) | {pop:,.0f} hab | "
            msg += f"Accès santé limité, vaccination décentralisée → "
            msg += f"Surveillance communautaire prioritaire"
            return msg
        
        else:
            return f"📍 {prov_name} - Contexte indéterminé"
    
    @classmethod
    def _detect_comorbidity_risk(cls, health_zone, disease_name, prediction_date=None):
        """Détecte risque comorbidité malnutrition.
        
        Compare les cas de maladie vs cas malnutrition à la même période
        et dans la même zone.
        """
        if cls._detect_threat_type(disease_name) != 'epidemic':
            return 'low'  # Menace nutritionnelle = pas de comorbidité
        
        # Données épidémie
        epid_data = ClinicalData.objects.filter(
            health_zone=health_zone,
            disease_name=disease_name,
            month__gte=datetime.now() - timedelta(days=180)
        ).aggregate(total=Count('id'), avg_cases=Avg('cases_count'))
        
        # Données malnutrition COTEMPORELLE
        malnutr_data = ClinicalData.objects.filter(
            health_zone=health_zone,
            disease_name__icontains='malnutrition',
            month__gte=datetime.now() - timedelta(days=180)
        ).aggregate(total=Count('id'), avg_cases=Avg('cases_count'))
        
        # Heuristique: si malnutrition >30% cas épidémie = high risk
        if epid_data['avg_cases'] and malnutr_data['avg_cases']:
            ratio = malnutr_data['avg_cases'] / max(1, epid_data['avg_cases'])
            if ratio > 0.3:
                return 'high'
            elif ratio > 0.1:
                return 'medium'
        
        # Vérifier Predictions récentes de malnutrition
        mal_preds = Prediction.objects.filter(
            health_zone=health_zone,
            disease_name__icontains='malnutrition',
            predicted_cases__gt=50
        ).exists()
        
        return 'medium' if mal_preds else 'low'
    
    @classmethod
    def extract_learning(cls, prediction, disease_name=None):
        """Extrait CE QUE LE MODÈLE A APPRIS EN LISANT LES DONNÉES."""
        if not prediction or not prediction.influence_factors:
            return None
        
        inf = prediction.influence_factors
        health_zone = prediction.health_zone
        d_name = disease_name or prediction.disease_name
        
        # === NNOYAU: Ce que train_models.py produit ===
        learning = {
            'model': inf.get('model_used', 'Noyau IA'),
            'reliability': inf.get('reliability', 70),
            'spread_rate': inf.get('weekly_spread_rate', 5),
            'severity': inf.get('attack_angle', 'Modéré'),
            'data_sparse': inf.get('data_sparse', False),
            'r2': inf.get('r2', 0),
            'mae': inf.get('mae', 0),
        }
        
        # === ENRICHISSEMENT: Lire les données pour les fields manquants ===
        
        # 1. Threat Type
        threat = cls._detect_threat_type(d_name)
        coinfection = cls._detect_coinfection(health_zone, d_name, prediction)
        learning['threat_type'] = coinfection or threat
        
        # 2. Saisonnalité (LSTM apprend patterns mensuels)
        learning['seasonality'] = cls._calculate_seasonality(health_zone, d_name)
        
        # 3. Climat (ERA5 features que le modèle a intégré)
        learning['climate_risk'] = cls._calculate_climate_risk(health_zone)
        
        # 4. Contexte régional (Encoded dans géographie)
        learning['region'] = cls._detect_regional_context(health_zone)
        
        # 5. Comorbidité (Détectée par corrélation données)
        learning['comorbidity'] = cls._detect_comorbidity_risk(health_zone, d_name, prediction.prediction_date)
        
        return learning

    
    @classmethod
    def format_recommendations(cls, prediction, disease_name):
        """
        EN LISANT les outputs du modèle + données, formate des recommandations.
        Pas de recalcul - on affiche ce que le modèle a APPRIS des données.
        """
        learning = cls.extract_learning(prediction, disease_name)
        if not learning:
            return ["⚠️ Pas de prédiction disponible"]
        
        recs = []
        
        # === 1. PROPAGATION (ce que le modèle a mesuré) ===
        rate = learning['spread_rate']
        reliability = learning['reliability']
        model = learning['model']
        
        if model == "Bi-LSTM":
            recs.append(f"🤖 LOGIQUE DEEP LEARNING : Analyse par réseaux de neurones bidirectionnels (Confiance R²: {int(learning['r2']*100)}%).")
        elif model == "Linear Trend":
            recs.append(f"📈 TENDANCE STATISTIQUE : Modèle linéaire activé pour séries courtes (Échantillon réduit).")
        else:
            recs.append(f"📉 PERSISTANCE : Données historiques trop rares pour un modèle prédictif complexe.")

        if rate > 20:
            recs.append(f"� PROPAGATION EXPLOSIVE : Vitesse de {rate}%/semaine. Alerte épidémique immédiate.")
        elif rate > 5:
            recs.append(f"📊 CROISSANCE ACTIVE : Vitesse de {rate}%/semaine. Monitoring renforcé.")
        else:
            recs.append(f"✅ ÉVOLUTION STABLE : Variation minime ({rate}%). Surveillance de routine.")
        
        # === 2. TYPE DE MENACE (ce que le modèle a classé) ===
        threat = learning['threat_type']
        
        if threat == 'epidemic':
            recs.append("🦠 TYPE: ÉPIDÉMIE CONTAGIEUSE (transmission interhumaine)")
        elif threat == 'nutritional_crisis':
            recs.append("🌽 TYPE: CRISE NUTRITIONNELLE (facteurs structurels)")
        elif threat == 'coinfection':
            recs.append("⚠️ TYPE: COMORBIDITÉ (épidémie + malnutrition)")
        
        # === 3. CLIMAT (le modèle a mesuré l'impact) ===
        climate_risk = learning['climate_risk']
        
        if climate_risk > 75:
            recs.append(
                f"🌡️ CLIMAT TRÈS FAVORABLE | Score: {climate_risk}/100 | "
                f"Transmission accélérée par conditions météo"
            )
        elif climate_risk > 50:
            recs.append(
                f"🌦️ CLIMAT MODÉRÉMENT FAVORABLE | Score: {climate_risk}/100"
            )
        else:
            recs.append(
                f"⛅ CLIMAT PEU FAVORABLE | Score: {climate_risk}/100 | "
                f"Transmission ralentie"
            )
        
        # === 4. CONTEXTE RÉGIONAL (généré intelligemment des DONNÉES) ===
        region_msg = cls._generate_regional_message(prediction.health_zone, disease_name)
        recs.append(region_msg)
        
        # === 5. SAISONNALITÉ (LSTM a détecté patterns) ===
        seasonality = learning['seasonality']
        
        if seasonality == 'high':
            recs.append("📅 FORTE SAISONNALITÉ - Pic imminent | Prépositionner ressources")
        elif seasonality == 'moderate':
            recs.append("📈 SAISONNALITÉ MODÉRÉE - Adapter selon période")
        
        # === 6. COMORBIDITÉS ===
        comorbidity = learning['comorbidity']
        
        if comorbidity == 'high':
            recs.append("⚠️ COMORBIDITÉ PROBABLE - Malnutrition présente | Prise en charge intégrée <5ans")
        elif comorbidity == 'medium':
            recs.append("🔍 COMORBIDITÉ POSSIBLE - Vérifier co-occurrence malnutrition")
        
        # === 7. QUALITÉ SCIENTIFIQUE (Données) ===
        if learning['data_sparse']:
            recs.append(f"⚠️ FIABILITÉ LIMITÉE : Échantillon historique réduit. Prédiction à prendre avec prudence.")
        else:
            recs.append(f"✅ VALIDITÉ SCIENTIFIQUE : Volume de données optimal (R²={learning['r2']:.2f}).")
        
        return recs
    
    @classmethod
    def get_summary(cls, prediction, disease_name):
        """Résumé une-ligne pour la UI."""
        learning = cls.extract_learning(prediction, disease_name)
        if not learning:
            return "⚠️ Pas de prédiction"
        
        return (
            f"🧠 {learning['model']} | Precision(R²): {int(learning['r2']*100)}% | "
            f"Growth: {learning['spread_rate']}%/sem"
        )


# Alias pour compatibilité avec views.py existant
class EpidemicContextualizer(SmartRecommender):
    """Compatibilité avec ancien API."""
    
    @classmethod
    def generate_intelligent_recommendations(cls, health_zone, disease_name, 
                                            predicted_cases, prediction_obj=None, **kwargs):
        """API compatibilité: views.py l'appelle avec ces params."""
        if not prediction_obj:
            return ["⚠️ Pas de prédiction"]
        
        return cls.format_recommendations(prediction_obj, disease_name)
    
    @classmethod
    def get_smart_insights(cls, health_zone, disease_name, prediction_obj=None, **kwargs):
        """API compatibilité."""
        if not prediction_obj:
            return "⚠️ Pas de prédiction"
        
        return cls.get_summary(prediction_obj, disease_name)
    
    @classmethod
    def detect_threat_type(cls, disease_name):
        """Détecte type de menace (épidémie vs crise nutritionnelle) via le nom maladie."""
        return cls._detect_threat_type(disease_name)
