#!/usr/bin/env python
"""
🏥 HealthDRC - Fichier d'Entraînement Complet v2.0
======================================================

Pipeline intégré qui fait :
1. Récupération données maladies depuis Django ORM
2. Récupération données provinces et zones de santé
3. Chargement données climatiques (fichiers .nc)
4. Nettoyage + traitement des trous de données
5. Entraînement multi-tier adaptatif
6. Génération prédictions pour toutes zones
7. Sauvegarde en base + préparation dashboard

Auteur : Yannick Magayane
Date : Avril 2026
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import gc

# 🔧 Setup paths
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()

from django.db import transaction
from analytics.models import HealthZone, ClinicalData, Prediction, Province, ClimateData
from django.db.models import Count, Avg, Max, Min
from analytics.disease_analyzer import DiseaseAnalyzer

# ML Libraries
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Bidirectional,
                                     Conv1D, MaxPooling1D, Flatten,
                                     TimeDistributed, Input, Reshape)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / 'training.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
class Config:
    """Configuration centralisée"""

    # Diseases to train
    DISEASES = ['Paludisme', 'Choléra', 'Tuberculose', 'VIH/SIDA']

    # Tier thresholds (minimum months of data)
    TIER1_MIN_MONTHS = 16    # ConvLSTM2D (spatio-temporal)
    TIER2_MIN_MONTHS = 8     # CNN-LSTM (patterns + climate)
    TIER3_MIN_MONTHS = 4     # Bi-LSTM (temporal only)

    # Training parameters
    SEQUENCE_LENGTH = 6      # Months in each sequence
    EPOCHS_TIER1 = 80
    EPOCHS_TIER2 = 100
    EPOCHS_TIER3 = 60
    BATCH_SIZE = 16
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 15

    # Data paths
    DATA_DIR = BASE_DIR / 'data'
    CLIMATE_DIR = DATA_DIR / 'climat'
    MODELS_DIR = BASE_DIR / 'models'
    PROCESSED_DIR = DATA_DIR / 'processed'
    DASHBOARD_DATA_FILE = MODELS_DIR / 'dashboard_training_data.json'

    # Create directories
    MODELS_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)

    # Climate variables to extract
    CLIMATE_VARS = ['temperature', 'precipitation', 'relative_humidity']

# ==================== DATA LOADERS ====================
class DataLoader:
    """Charge toutes les données nécessaires"""

    @staticmethod
    def load_provinces_and_zones():
        """Charge provinces et zones de santé avec géométries"""
        logger.info("📋 Chargement provinces et zones de santé...")

        provinces = list(Province.objects.all().values(
            'id', 'code', 'name'
        ))
        logger.info(f"  ✓ {len(provinces)} provinces chargées")

        zones = list(HealthZone.objects.select_related('territory__province').all().values(
            'id', 'code', 'name', 'territory_id', 'territory__name',
            'territory__province_id', 'territory__province__name',
            'latitude', 'longitude', 'population'
        ))
        logger.info(f"  ✓ {len(zones)} zones de santé chargées")

        return provinces, zones

    @staticmethod
    def load_clinical_data(diseases=None):
        """Charge données cliniques depuis Django ORM"""
        if diseases is None:
            diseases = Config.DISEASES

        logger.info("🏥 Chargement données cliniques...")

        records = ClinicalData.objects.filter(
            disease_name__in=diseases
        ).select_related('health_zone').values(
            'id', 'health_zone_id', 'disease_name', 'cases_count',
            'deaths_count', 'month'
        )

        df = pd.DataFrame(list(records))
        logger.info(f"  ✓ {len(df)} enregistrements cliniques")

        if len(df) > 0:
            df['month'] = pd.to_datetime(df['month'])
            df = df.rename(columns={
                'disease_name': 'disease',
                'cases_count': 'cases',
                'deaths_count': 'deaths',
                'territory__province__name': 'province'
            })

            # Statistiques par maladie
            for disease in diseases:
                count = len(df[df['disease'] == disease])
                logger.info(f"    • {disease}: {count} records")

        return df

    @staticmethod
    def load_climate_data_for_zone(zone_name, province_name, start_date, end_date):
        """
        Charge données climatiques NetCDF pour une province
        Fichiers : data/climat/{province}_{year}.nc
        """
        climate_data = []

        # Generate years to check
        years = list(range(start_date.year, end_date.year + 1))

        for year in years:
            # Build filename (lowercase province name)
            prov_filename = province_name.lower().replace(' ', '-').replace('é', 'e').replace('è', 'e')
            nc_file = Config.CLIMATE_DIR / f"{prov_filename}_{year}.nc"

            if nc_file.exists():
                try:
                    ds = xr.open_dataset(nc_file)

                    # Extract variables for the zone (assuming province-wide data)
                    for var_name in Config.CLIMATE_VARS:
                        if var_name in ds:
                            # Get time series
                            time_series = ds[var_name].to_series()
                            if isinstance(time_series, pd.Series):
                                for date, value in time_series.items():
                                    climate_data.append({
                                        'date': pd.Timestamp(date),
                                        'province': province_name,
                                        'zone': zone_name,
                                        'variable': var_name,
                                        'value': float(value)
                                    })

                    ds.close()
                    logger.debug(f"    ✓ {nc_file.name}")
                except Exception as e:
                    logger.warning(f"    ✗ Erreur lecture {nc_file.name}: {e}")

        if climate_data:
            climate_df = pd.DataFrame(climate_data)
            # Pivot to wide format
            climate_pivot = climate_df.pivot_table(
                index=['date', 'province', 'zone'],
                columns='variable',
                values='value'
            ).reset_index()
            return climate_pivot

        return None

# ==================== DATA PROCESSOR ====================
class DataProcessor:
    """Nettoyage et prétraitement des données"""

    @staticmethod
    def handle_missing_values(df, strategy='interpolate'):
        """Gère les valeurs manquantes"""
        logger.info("🧹 Nettoyage des données manquantes...")

        original_len = len(df)

        # For time series, forward fill then backward fill
        if 'date' in df.columns:
            df = df.sort_values('date')

        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df[col].isna().any():
                if strategy == 'interpolate':
                    df[col] = df[col].interpolate(method='linear')
                elif strategy == 'ffill':
                    df[col] = df[col].ffill()

                # If still NaN, fill with mean
                if df[col].isna().any():
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val)
                    logger.info(f"  • Colonne '{col}': {df[col].isna().sum()} NaN restants (remplacés par mean={mean_val:.2f})")

        logger.info(f"  ✓ Données nettoyées: {len(df)}/{original_len} lignes")
        return df

    @staticmethod
    def merge_climate_with_epi(epi_df, climate_df):
        """Fusionne données climatiques et épidémiologiques"""
        if climate_df is None:
            logger.warning("⚠️ Pas de données climatiques disponibles")
            return epi_df

        logger.info("🌦️ Fusion données climatiques...")

        # Ensure date columns are datetime
        epi_df['month'] = pd.to_datetime(epi_df['month'])
        climate_df['date'] = pd.to_datetime(climate_df['date'])

        # Merge by zone and month
        merged = pd.merge(
            epi_df,
            climate_df,
            left_on=['health_zone_id', 'month'],
            right_on=['zone', 'date'],
            how='left',
            suffixes=('_epi', '_climate')
        )

        # Check merge results
        missing_climate = merged[['temperature', 'precipitation', 'relative_humidity']].isna().any(axis=1).sum()
        total = len(merged)
        coverage = 100 * (1 - missing_climate / total)
        logger.info(f"  • Couverture climatique: {coverage:.1f}% ({total-missing_climate}/{total})")

        return merged

    @staticmethod
    def create_sequences(data, sequence_length=6):
        """Crée séquences pour LSTM"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(data[i+sequence_length])
        return np.array(X), np.array(y)

# ==================== MODEL ARCHITECTURES ====================
class ModelBuilder:
    """Construit les modèles multi-tier"""

    @staticmethod
    def build_tier1_model(input_shape):
        """
        Tier 1: ConvLSTM2D pour propagation spatio-temporelle
        Nécessite: données multi-zones simultanées (non implémenté ici car complexe)
        """
        logger.warning("⚠️ Tier 1 désactivé (requires multi-zone matrix)")
        return None

    @staticmethod
    def build_tier2_model(input_shape, climate_features=3):
        """
        Tier 2: CNN-LSTM avec features climatiques
        Input: [historical_cases + climate_features] × sequence_length
        """
        inputs = Input(shape=input_shape)

        # Process time series with Conv1D for local patterns
        x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = Dropout(0.2)(x)
        x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
        x = Dropout(0.2)(x)

        # LSTM for long-term dependencies
        x = LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.3)(x)
        x = LSTM(64, kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.3)(x)

        # Dense layers
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    @staticmethod
    def build_tier3_model(input_shape):
        """
        Tier 3: Bi-LSTM pour séries temporelles univariées
        """
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True,
                              kernel_regularizer=l2(1e-4)),
                         input_shape=input_shape),
            Dropout(0.3),
            Bidirectional(LSTM(64, kernel_regularizer=l2(1e-4))),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

# ==================== TRAINER ====================
class Trainer:
    """Gère l'entraînement des modèles"""

    def __init__(self):
        self.metadata = {
            'training_date': datetime.now().isoformat(),
            'diseases': {},
            'models_trained': 0,
            'tier_distribution': {'Tier 1': 0, 'Tier 2': 0, 'Tier 3': 0, 'Tier 4': 0}
        }

    def select_tier(self, n_months, has_climate, n_zones=1):
        """Sélectionne le tier approprié selon disponibilité des données"""
        if n_months >= Config.TIER1_MIN_MONTHS and n_zones >= 2:
            return 1, "ConvLSTM2D"
        elif n_months >= Config.TIER2_MIN_MONTHS and has_climate:
            return 2, "CNN-LSTM"
        elif n_months >= Config.TIER3_MIN_MONTHS:
            return 3, "Bi-LSTM"
        else:
            return 4, "Persistence"

    def train_disease_model(self, disease_name, disease_data, climate_data=None):
        """Entraîne un modèle pour une maladie donnée"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🧠 Entraînement: {disease_name}")
        logger.info(f"{'='*60}")

        # Prepare data
        disease_ts = disease_data.sort_values('month')
        months_count = len(disease_ts)
        has_climate = climate_data is not None and len(climate_data) > 0

        # Select tier
        tier, tier_name = self.select_tier(
            months_count,
            has_climate,
            n_zones=1  # Single zone training
        )

        logger.info(f"  Tier {tier}: {tier_name} ( données: {months_count} mois, climate: {'oui' if has_climate else 'non'} )")

        # Prepare features
        cases = disease_ts['cases'].values.reshape(-1, 1)

        if tier == 2 and has_climate:
            # Combine cases with climate features
            climate_features = disease_ts[['temperature', 'precipitation', 'relative_humidity']].values
            features = np.hstack([cases, climate_features])
        else:
            features = cases

        # Scale features
        feature_scaler = RobustScaler()
        features_scaled = feature_scaler.fit_transform(features)

        # Create sequences
        X, y = DataProcessor.create_sequences(features_scaled, Config.SEQUENCE_LENGTH)

        if len(X) < 2:
            logger.warning(f"  ⚠️ Pas assez de séquences ({len(X)} < 2)")
            return None

        # Train/Test split
        split_idx = int(len(X) * (1 - Config.VALIDATION_SPLIT))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(f"  • Séquences: train={len(X_train)}, test={len(X_test)}")
        logger.info(f"  • Shape X: {X_train.shape}")

        # Train model (skip Tier 1 - too complex, Tier 4 - no training)
        if tier == 4:
            logger.info("  • Modèle Persistence (aucun entraînement requis)")
            model = None
            mae = 0
            rmse = 0
            r2 = 1
            epochs = 0
        else:
            # Build model
            if tier == 1:
 model = ModelBuilder.build_tier1_model((Config.SEQUENCE_LENGTH, features.shape[1]))
            else:
                model = ModelBuilder.build_tier3_model((Config.SEQUENCE_LENGTH, features.shape[1]))

            if model is None:
                logger.warning(f"  ⚠️ Construction modèle Tier {tier} échouée")
                return None

            # Training parameters
            epochs = Config.EPOCHS_TIER1 if tier == 1 else (
                Config.EPOCHS_TIER2 if tier == 2 else Config.EPOCHS_TIER3
            )

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=Config.EARLY_STOPPING_PATIENCE,
                              restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
            ]

            # Train
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=Config.BATCH_SIZE,
                callbacks=callbacks,
                verbose=0
            )

            # Evaluate
            y_pred = model.predict(X_test, verbose=0)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            epochs = len(history.history['loss'])

            logger.info(f"  • Époques: {epochs}")
            logger.info(f"  • MAE:  {mae:.4f}")
            logger.info(f"  • RMSE: {rmse:.4f}")
            logger.info(f"  • R²:   {r2:.4f}")

            # Save model
            safe_name = disease_name.replace('/', '_').replace(' ', '_')
            model_path = Config.MODELS_DIR / f"model_{safe_name}_tier{tier}.keras"
            model.save(model_path)
            logger.info(f"  ✅ Modèle sauvé: {model_path.name}")

        # Save scaler
        safe_name = disease_name.replace('/', '_').replace(' ', '_')
        scaler_path = Config.MODELS_DIR / f"scaler_{safe_name}.pkl"
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(feature_scaler, f)

        # Save zone-specific scaler for predictions
        if climate_data is not None:
            climate_scaler_path = Config.MODELS_DIR / f"climate_scaler_{safe_name}.pkl"
            climate_scaler = RobustScaler()
            climate_scaler.fit(climate_features if tier == 2 else np.array([]).reshape(-1, 1))
            with open(climate_scaler_path, 'wb') as f:
                pickle.dump(climate_scaler, f)

        # Update metadata
        self.metadata['diseases'][disease_name] = {
            'tier': tier,
            'tier_name': tier_name,
            'months_of_data': months_count,
            'has_climate': has_climate,
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'epochs': epochs,
            'n_sequences': len(X),
            'model_path': str(model_path) if model else None,
            'scaler_path': str(scaler_path)
        }
        self.metadata['tier_distribution'][f"Tier {tier}"] += 1
        self.metadata['models_trained'] += 1

        # Cleanup
        gc.collect()

        return {
            'tier': tier,
            'tier_name': tier_name,
            'r2': r2,
            'mae': mae,
            'model': model,
            'scaler': feature_scaler
        }

    def save_metadata(self):
        """Sauvegarde métadonnées d'entraînement"""
        self.metadata['last_updated'] = datetime.now().isoformat()

        with open(Config.DASHBOARD_DATA_FILE, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

        logger.info(f"\n✅ Métadonnées sauvegardées: {Config.DASHBOARD_DATA_FILE.name}")

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("📊 RÉSUMÉ D'ENTRAÎNEMENT")
        logger.info(f"{'='*60}")
        for disease, info in self.metadata['diseases'].items():
            logger.info(f"\n  {disease}:")
            logger.info(f"    Tier: {info['tier']} ({info['tier_name']})")
            logger.info(f"    Données: {info['months_of_data']} mois")
            logger.info(f"    R²: {info['r2']:.4f}")
            logger.info(f"    MAE: {info['mae']:.4f}")

# ==================== PREDICTOR ====================
class Predictor:
    """Génère prédictions pour toutes les zones"""

    def __init__(self, trainer):
        self.trainer = trainer
        self.predictions = []

    def predict_for_disease_zone(self, disease, zone_id, zone_name, province_name,
                                  historical_data, climate_data=None, model_info=None):
        """Génère prédiction pour une zone et maladie"""

        if model_info is None:
            # No model trained - use persistence
            last_cases = historical_data['cases'].iloc[-1] if len(historical_data) > 0 else 0
            prediction_value = last_cases * 1.1  # +10% conservative
            tier = 4
            confidence = 0.3
            return prediction_value, tier, confidence

        # Load model and scaler
        import tensorflow as tf
        import pickle

        try:
            model = tf.keras.models.load_model(model_info['model_path'])
            with open(model_info['scaler_path'], 'rb') as f:
                scaler = pickle.load(f)
        except Exception as e:
            logger.error(f"  ✗ Erreur chargement modèle: {e}")
            return None, None, None

        # Prepare last sequence
        cases_ts = historical_data.sort_values('month')['cases'].values

        if len(cases_ts) < Config.SEQUENCE_LENGTH:
            logger.warning(f"  ⚠️ Pas assez de données pour prédiction ({len(cases_ts)} < {Config.SEQUENCE_LENGTH})")
            return None, None, None

        last_sequence = cases_ts[-Config.SEQUENCE_LENGTH:].reshape(1, -1, 1)
        last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1)).reshape(1, -1, 1)

        # Predict
        try:
            pred_scaled = model.predict(last_sequence_scaled, verbose=0)
            prediction = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]

            # Ensure non-negative
            prediction = max(0, prediction)

            # Confidence based on R²
            r2 = model_info.get('r2', 0)
            confidence = max(0, min(100, r2 * 100))
            tier = model_info['tier']

            return prediction, tier, confidence
        except Exception as e:
            logger.error(f"  ✗ Erreur prédiction: {e}")
            return None, None, None

    def generate_all_predictions(self):
        """Génère prédictions pour toutes les zones et maladies"""
        logger.info("\n🔮 Génération prédictions pour toutes zones...")

        # Get all zones
        from analytics.models import HealthZone
        zones = HealthZone.objects.select_related('territory__province').all()

        predictions_count = 0

        for zone in zones:
            zone_id = zone.id
            zone_name = zone.name
            province_name = zone.territory.province.name if zone.territory else None

            if not province_name:
                continue

            for disease in Config.DISEASES:
                # Get historical data for this zone and disease
                historical = ClinicalData.objects.filter(
                    health_zone_id=zone_id,
                    disease_name=disease
                ).order_by('month').values('month', 'cases_count')

                hist_df = pd.DataFrame(list(historical))
                if len(hist_df) == 0:
                    continue

                hist_df = hist_df.rename(columns={'cases_count': 'cases'})
                hist_df['month'] = pd.to_datetime(hist_df['month'])

                # Check if we have a trained model for this disease
                model_info = self.trainer.metadata['diseases'].get(disease)

                # Predict next month
                next_month = hist_df['month'].max() + relativedelta(months=1)
                prediction_value, tier, confidence = self.predict_for_disease_zone(
                    disease, zone_id, zone_name, province_name,
                    hist_df, model_info=model_info
                )

                if prediction_value is not None:
                    # Calculer le taux de propagation basé sur les tendances historiques
                    # Récupérer l'objet zone complet
                    try:
                        zone_obj = HealthZone.objects.get(id=zone_id)
                        propagation_rate = DiseaseAnalyzer.calculate_propagation_rate(
                            health_zone=zone_obj,
                            disease_name=disease,
                            months_window=6  # utiliser les 6 derniers mois pour tendance récente
                        )
                    except Exception as e:
                        logger.warning(f"  ⚠️ Erreur calcul propagation {zone_name}/{disease}: {e}")
                        propagation_rate = 0.0

                    # Save prediction in database
                    try:
                        Prediction.objects.update_or_create(
                            health_zone_id=zone_id,
                            disease_name=disease,
                            prediction_date=next_month,
                            defaults={
                                'predicted_cases': round(prediction_value, 1),
                                'propagation_rate': propagation_rate,
                                'influence_factors': {
                                    'reliability': round(confidence, 1),
                                    'model_used': model_info['tier_name'] if model_info else 'Persistence',
                                    'tier': tier,
                                    'data_months': len(hist_df),
                                    'last_updated': datetime.now().isoformat()
                                }
                            }
                        )
                        predictions_count += 1
                    except Exception as e:
                        logger.error(f"  ✗ Erreur sauvegarde prédiction {disease}/{zone_name}: {e}")

        logger.info(f"  ✅ {predictions_count} prédictions générées et sauvegardées")
        return predictions_count

# ==================== MAIN ====================
def main():
    """Pipeline principal d'entraînement"""

    logger.info("="*70)
    logger.info("🏥 HEALTHDRC - FICHIER D'ENTRAÎNEMENT COMPLET")
    logger.info("="*70)

    try:
        # ==================== ÉTAPE 1: Chargement données ====================
        logger.info("\n" + "="*70)
        logger.info("📥 ÉTAPE 1: CHARGEMENT DONNÉES")
        logger.info("="*70)

        provinces, zones = DataLoader.load_provinces_and_zones()
        epi_df = DataLoader.load_clinical_data()

        if len(epi_df) == 0:
            logger.error("❌ Aucune donnée clinique disponible. Arrêt.")
            return False

        # ==================== ÉTAPE 2: Nettoyage ====================
        logger.info("\n" + "="*70)
        logger.info("🧹 ÉTAPE 2: NETTOYAGE DONNÉES")
        logger.info("="*70)

        epi_df = DataProcessor.handle_missing_values(epi_df, strategy='interpolate')

        # Save processed data
        epi_df.to_csv(Config.PROCESSED_DIR / 'epi_data_processed.csv', index=False)
        logger.info(f"  ✅ Données épidémiologiques sauvées: {Config.PROCESSED_DIR / 'epi_data_processed.csv'}")

        # ==================== ÉTAPE 3: Chargement climat ====================
        logger.info("\n" + "="*70)
        logger.info("🌦️ ÉTAPE 3: CHARGEMENT DONNÉES CLIMATIQUES")
        logger.info("="*70)

        # For each province, load climate data
        climate_data_all = []
        provinces_with_climate = set()

        for province in provinces:
            prov_name = province['name']
            try:
                # Get date range from epi data for this province
                prov_epi = epi_df[epi_df.get('province') == prov_name]
                if len(prov_epi) > 0:
                    start_date = prov_epi['month'].min()
                    end_date = prov_epi['month'].max()

                    climate_prov = DataLoader.load_climate_data_for_zone(
                        zone_name=prov_name,
                        province_name=prov_name,
                        start_date=start_date,
                        end_date=end_date
                    )

                    if climate_prov is not None:
                        climate_data_all.append(climate_prov)
                        provinces_with_climate.add(prov_name)
                        logger.info(f"  ✓ {prov_name}: données climatiques chargées")
            except Exception as e:
                logger.warning(f"  ✗ {province['name']}: {e}")

        climate_df = pd.concat(climate_data_all, ignore_index=True) if climate_data_all else None
        if climate_df is not None:
            climate_df.to_csv(Config.PROCESSED_DIR / 'climate_data_processed.csv', index=False)
            logger.info(f"  ✅ Données climatiques sauvées")

        # ==================== ÉTAPE 4: Fusion données ====================
        logger.info("\n" + "="*70)
        logger.info("🔗 ÉTAPE 4: FUSION DONNÉES ÉPIDÉMIO + CLIMAT")
        logger.info("="*70)

        if climate_df is not None:
            merged_df = DataProcessor.merge_climate_with_epi(epi_df, climate_df)
            merged_df.to_csv(Config.PROCESSED_DIR / 'merged_data.csv', index=False)
            logger.info(f"  ✅ Données fusionnées sauvées")
        else:
            merged_df = epi_df
            logger.warning("  ⚠️ Pas de données climatiques, utilisation épidémiologie seule")

        # ==================== ÉTAPE 5: Entraînement par zone ====================
        logger.info("\n" + "="*70)
        logger.info("🧠 ÉTAPE 5: ENTRAÎNEMENT MODÈLES")
        logger.info("="*70)

        trainer = Trainer()

        # Group data by zone
        zones_trained = 0
        for zone in zones[:10]:  # LIMITER pour test (enlever [:10] pour toutes zones)
            zone_id = zone.id
            zone_name = zone.name
            province_name = zone.territory.province.name if zone.territory else None

            logger.info(f"\n📍 Zone: {zone_name} ({province_name})")

            for disease in Config.DISEASES:
                # Get historical data for this zone
                zone_disease_data = merged_df[
                    (merged_df.get('health_zone_id') == zone_id) &
                    (merged_df.get('disease') == disease)
                ].copy()

                if len(zone_disease_data) < Config.SEQUENCE_LENGTH + 2:
                    logger.debug(f"    ⚠️ {disease}: pas assez de données ({len(zone_disease_data)} mois)")
                    continue

                # Get climate data for this zone/province
                zone_climate = None
                if climate_df is not None and province_name in provinces_with_climate:
                    zone_climate = climate_df[
                        (climate_df.get('zone') == zone_name) |
                        (climate_df.get('province') == province_name)
                    ]

                # Train model
                result = trainer.train_disease_model(
                    disease_name=f"{disease}_{zone_name}",
                    disease_data=zone_disease_data,
                    climate_data=zone_climate
                )

                if result:
                    zones_trained += 1

        logger.info(f"\n✅ {zones_trained} modèles entraînés")

        # ==================== ÉTAPE 6: Sauvegarde métadonnées ====================
        trainer.save_metadata()

        # ==================== ÉTAPE 7: Génération prédictions ====================
        logger.info("\n" + "="*70)
        logger.info("🔮 ÉTAPE 7: GÉNÉRATION PRÉDICTIONS")
        logger.info("="*70)

        predictor = Predictor(trainer)
        preds_count = predictor.generate_all_predictions()

        # ==================== TERMINE ====================
        logger.info("\n" + "="*70)
        logger.info("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
        logger.info("="*70)
        logger.info(f"  • Modèles entraînés: {trainer.metadata['models_trained']}")
        logger.info(f"  • Prédictions générées: {preds_count}")
        logger.info(f"  • Dashboard prêt: http://localhost:8000/analysis/dashboard/")
        logger.info("="*70)

        return True

    except Exception as e:
        logger.error(f"\n❌ ERREUR CRITIQUE: {e}", exc_info=True)
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
