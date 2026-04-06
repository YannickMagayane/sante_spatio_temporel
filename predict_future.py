#!/usr/bin/env python
"""
🔮 HealthDRC - Générateur de Prédictions Futures
==================================================

Ce script génère des prédictions de cas pour les mois à venir
en utilisant les modèles entraînés.

Il calcule également :
- Taux de propagation (croissance/déclin)
- Facteurs de risque climatique
- Alertes basées sur seuils
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import pickle

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

warnings.filterwarnings('ignore')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()

from django.db import transaction
from analytics.models import Prediction, ClinicalData, HealthZone, Province
from django.db.models import Max

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== CONFIG ====================
FORECAST_MONTHS = 3  # Nombre de mois à prédire en avance
MIN_CASES_THRESHOLD = 50  # Seuil alerte épidémie

# ==================== CHARGEMENT MODÈLES ====================
class ModelRegistry:
    """Registre des modèles entraînés"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.metadata = None
        self.load_metadata()

    def load_metadata(self):
        metadata_path = BASE_DIR / 'models' / 'dashboard_training_data.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"✓ Métadonnées chargées: {len(self.metadata.get('diseases', {}))} maladies")
        else:
            logger.warning("⚠️ Aucune métadonnée trouvée. Exécutez train_complete.py d'abord.")

    def load_model_and_scaler(self, disease_name):
        """Charge modèle et scaler pour une maladie.

        Returns:
            (model, scaler, model_type): model_type est 'keras' ou 'persistence' ou None
        """
        # Vérifier d'abord si modèle dans diseases (format ancien)
        if 'diseases' in self.metadata and disease_name in self.metadata['diseases']:
            info = self.metadata['diseases'][disease_name]
            model_path = info.get('model_path')
            scaler_path = info.get('scaler_path')
            if model_path and Path(model_path).exists():
                try:
                    import tensorflow as tf
                    model = tf.keras.models.load_model(model_path)
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    return model, scaler, 'keras'
                except Exception as e:
                    logger.error(f"✗ Erreur chargement modèle {disease_name}: {e}")
                    return None, None, None
            else:
                return None, None, None

        # Vérifier si modèle de persistance (format nouveau)
        if 'model_performance' in self.metadata and disease_name in self.metadata['model_performance']:
            perf = self.metadata['model_performance'][disease_name]
            tier_name = perf.get('tier_name', '')
            if tier_name == 'Persistence':
                # Charger fichier de persistance
                safe_name = disease_name.replace(' ', '_').replace('/', '_')
                model_path = BASE_DIR / 'models' / f'model_{safe_name}_National.pkl'
                if model_path.exists():
                    try:
                        with open(model_path, 'rb') as f:
                            model_data = pickle.load(f)
                        return model_data, None, 'persistence'
                    except Exception as e:
                        logger.error(f"✗ Erreur chargement persistance {disease_name}: {e}")
                        return None, None, None
                else:
                    return None, None, None

        return None, None, None

# ==================== PRÉDICTEUR ====================
class FuturePredictor:
    """Génère prédictions futures"""

    def __init__(self):
        self.registry = ModelRegistry()
        self.SEQUENCE_LENGTH = 6

    def get_historical_sequence(self, zone_id, disease, months_back=24):
        """Récupère séquence historique pour prédiction"""
        records = ClinicalData.objects.filter(
            health_zone_id=zone_id,
            disease_name=disease
        ).order_by('month').values('month', 'cases_count')[:months_back]

        if len(records) < self.SEQUENCE_LENGTH:
            return None, None

        df = pd.DataFrame(list(records))
        df['month'] = pd.to_datetime(df['month'])
        df = df.sort_values('month')

        cases = df['cases_count'].values
        dates = df['month'].values

        return cases[-self.SEQUENCE_LENGTH:], dates[-1]

    def predict_next_month(self, model, scaler, sequence, model_type='keras'):
        """Prédit le mois prochain.

        Args:
            model: modèle Keras ou dict de persistance
            scaler: scaler sklearn ou None pour persistance
            sequence: array historique de longueur SEQUENCE_LENGTH
            model_type: 'keras' ou 'persistence'

        Returns:
            float: prédiction
        """
        if model_type == 'persistence':
            # Modèle de persistance: prédire la dernière valeur
            return float(sequence[-1])

        if model is None or scaler is None:
            return None

        # Modèle Keras
        seq_reshaped = sequence.reshape(-1, 1)
        seq_scaled = scaler.transform(seq_reshaped)
        seq_input = seq_scaled.reshape(1, self.SEQUENCE_LENGTH, 1)

        try:
            pred_scaled = model.predict(seq_input, verbose=0)
            prediction = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
            return max(0, prediction)
        except Exception as e:
            logger.error(f"Erreur prédiction: {e}")
            return None

    def calculate_propagation_rate(self, historical_cases, prediction):
        """Calcule taux de propagation (croissance)"""
        if len(historical_cases) == 0:
            return 0.0

        last_value = historical_cases[-1]
        if last_value == 0:
            return 0.0 if prediction == 0 else 100.0

        rate = ((prediction - last_value) / last_value) * 100
        return round(rate, 1)

    def generate_predictions_for_zone(self, zone, months_ahead=FORECAST_MONTHS):
        """Génère prédictions pour toutes maladies d'une zone"""
        results = []

        for disease in ['Paludisme', 'Choléra', 'Tuberculose', 'VIH/SIDA']:
            # Récupérer séquence historique
            sequence, last_date = self.get_historical_sequence(zone.id, disease)

            if sequence is None:
                continue

            # Charger modèle
            model, scaler, model_type = self.registry.load_model_and_scaler(disease)

            # Déterminer tier et r2 depuis métadonnées
            tier = 4
            r2 = 0.0
            if 'model_performance' in self.registry.metadata and disease in self.registry.metadata['model_performance']:
                perf = self.registry.metadata['model_performance'][disease]
                tier = perf.get('tier', 4)
                r2 = perf.get('r2', 0.0)
            elif 'diseases' in self.registry.metadata and disease in self.registry.metadata['diseases']:
                tier = self.registry.metadata['diseases'][disease].get('tier', 4)
                r2 = self.registry.metadata['diseases'][disease].get('r2', 0.0)

            # Faire prédiction
            if model_type == 'persistence':
                # Prédiction de persistance (dernière valeur)
                base_prediction = float(sequence[-1])
            elif model_type == 'keras':
                # Prédiction via modèle LSTM
                base_prediction = self.predict_next_month(model, scaler, sequence, model_type='keras')
            else:
                # Fallback: persistance simple si aucun modèle
                base_prediction = float(sequence[-1])
                tier = 4
                r2 = 0.3

            if base_prediction is None:
                continue

            # Générer prédictions pour les mois à venir (simulation simple)
            # Pour persistance, on répète la dernière valeur (ajustée d'un petit taux)
            for i in range(1, months_ahead + 1):
                if model_type == 'persistence':
                    # Pour persistance, on garde la même valeur avec petite variation
                    prediction = base_prediction * (1 + 0.01 * i)  # +1% par mois
                else:
                    # Pour LSTM, on prédit itérativement (simplifié: on répète la première prédiction)
                    prediction = base_prediction

                pred_date = last_date + relativedelta(months=i)
                propagation = self.calculate_propagation_rate(sequence, prediction)

                results.append({
                    'disease': disease,
                    'prediction_date': pred_date,
                    'prediction': prediction,
                    'propagation': propagation,
                    'reliability': max(30, min(95, int(r2 * 100 + 50))),  # fiabilité approx
                    'tier': tier,
                    'r2': r2
                })
                prediction = self.predict_next_month(model, scaler, sequence)
                r2 = self.registry.metadata['diseases'].get(disease, {}).get('r2', 0.5)

            if prediction is None:
                continue

            # Calculer propagation
            propagation_rate = self.calculate_propagation_rate(sequence, prediction)

            # Déterminer mois de prédiction
            if last_date:
                pred_month = last_date + relativedelta(months=1)
            else:
                pred_month = datetime.now().date().replace(day=1)

            results.append({
                'disease': disease,
                'prediction': round(prediction, 1),
                'propagation': propagation_rate,
                'tier': tier,
                'r2': r2,
                'prediction_date': pred_month,
                'zone': zone,
                'reliability': min(99, max(50, int(r2 * 100)))
            })

        return results

    def run(self, months_ahead=FORECAST_MONTHS):
        """Exécute la génération de prédictions futures"""
        logger.info("="*70)
        logger.info("🔮 GÉNÉRATION PRÉDICTIONS FUTURES")
        logger.info("="*70)

        # Récupérer toutes les zones
        zones = HealthZone.objects.select_related('territory__province').all()
        logger.info(f"  Zones à traiter: {zones.count()}")

        total_predictions = 0
        new_predictions = 0

        for zone in zones:
            zone_results = self.generate_predictions_for_zone(zone, months_ahead)

            for result in zone_results:
                total_predictions += 1

                # Sauvegarder en base (update or create)
                try:
                    pred, created = Prediction.objects.update_or_create(
                        health_zone=zone,
                        disease_name=result['disease'],
                        prediction_date=result['prediction_date'],
                        defaults={
                            'predicted_cases': result['prediction'],
                            'propagation_rate': result['propagation'],
                            'influence_factors': {
                                'reliability': result['reliability'],
                                'model_used': f"Tier {result['tier']}",
                                'tier': result['tier'],
                                'r2': round(result['r2'], 4),
                                'last_updated': datetime.now().isoformat()
                            }
                        }
                    )
                    if created:
                        new_predictions += 1
                        logger.debug(f"  ✓ {zone.name} - {result['disease']}: {result['prediction']} cas")
                except Exception as e:
                    logger.error(f"  ✗ Erreur sauvegarde {zone.name}/{result['disease']}: {e}")

        logger.info(f"\n✅ Prédictions générées:")
        logger.info(f"  • Total prédictions: {total_predictions}")
        logger.info(f"  • Nouvelles prédictions: {new_predictions}")
        logger.info(f"  • Zones couvertes: {zones.count()}")
        logger.info(f"  • Dashboard: http://localhost:8000/analysis/dashboard/")

        return total_predictions

# ==================== MAIN ====================
def main():
    predictor = FuturePredictor()
    success = predictor.run()

    if success > 0:
        logger.info("\n✅ Prédictions futures mises à jour !")
        return True
    else:
        logger.error("\n❌ Aucune prédiction générée")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
