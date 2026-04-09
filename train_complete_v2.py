#!/usr/bin/env python
"""
🏥 HealthDRC - Fichier d'Entraînement Complet v2.0 (CORRIGÉ)
============================================================

Version corrigée qui fonctionne avec la structure de données existante.

AMÉLIORATIONS par rapport à v1:
- Gestion correcte des clés étrangères Django
- Entraînement par maladie (non par zone) pour suffisance données
- Intégration climatique par province
- Calcul propagation rate réaliste
- Sauvegarde modèles par maladie seulement
- Support better pour petites données (Tier 4)
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
import pickle

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()

from django.db import transaction
from analytics.models import (
    HealthZone, ClinicalData, Prediction, Province, Territory, ClimateData
)
from django.db.models import Count, Avg, Max, Min, Q
from analytics.disease_analyzer import DiseaseAnalyzer

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Bidirectional,
                                     Conv1D, LSTM as KerasLSTM)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / 'training_complete.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ==================== CONFIG ====================
class Config:
    DISEASES = ['Paludisme', 'Choléra', 'Tuberculose', 'VIH/SIDA']
    SEQUENCE_LENGTH = 6
    EPOCHS_TIER2 = 100  # CNN-LSTM
    EPOCHS_TIER3 = 60   # Bi-LSTM
    BATCH_SIZE = 16
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 15

    TIER2_MIN_MONTHS = 8   # Need at least 8 months for CNN-LSTM
    TIER3_MIN_MONTHS = 4   # Need at least 4 months for Bi-LSTM

    DATA_DIR = BASE_DIR / 'data'
    CLIMATE_DIR = DATA_DIR / 'climat'
    MODELS_DIR = BASE_DIR / 'models'
    PROCESSED_DIR = DATA_DIR / 'processed'

    MODELS_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)

    DASHBOARD_DATA_FILE = MODELS_DIR / 'dashboard_training_data.json'

# ==================== DATA LOADER ====================
class ClimateDataLoader:
    """Charge et aggregate données climatiques par province"""

    def __init__(self):
        self.cache = {}

    def get_province_climate_timeseries(self, province_name, start_date, end_date):
        """
        Récupère données climatiques pour une province
        depuis data/climat/{province}_{year}.nc
        """
        cache_key = f"{province_name}_{start_date.year}_{end_date.year}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        climate_records = []

        # Generate years to fetch
        years = range(start_date.year, end_date.year + 1)

        for year in years:
            # Build filename: normalize province name
            prov_normalized = province_name.lower() \
                .replace(' ', '-') \
                .replace('é', 'e') \
                .replace('è', 'e') \
                .replace('ê', 'e') \
                .replace('à', 'a') \
                .replace('â', 'a')

            nc_file = Config.CLIMATE_DIR / f"{prov_normalized}_{year}.nc"

            if nc_file.exists():
                try:
                    ds = xr.open_dataset(nc_file)

                    # Extract variables
                    for var in ['temperature', 'precipitation', 'relative_humidity']:
                        if var in ds:
                            # Convert time series to pandas
                            series = ds[var].to_series()
                            for date, value in series.items():
                                climate_records.append({
                                    'date': pd.Timestamp(date),
                                    'province': province_name,
                                    'variable': var,
                                    'value': float(value)
                                })

                    ds.close()
                    logger.debug(f"    ✓ {nc_file.name}")
                except Exception as e:
                    logger.warning(f"    ✗ {nc_file.name}: {e}")

        if not climate_records:
            logger.warning(f"⚠️ Aucune donnée climatique pour {province_name}")
            self.cache[cache_key] = None
            return None

        # Pivot to wide format
        df = pd.DataFrame(climate_records)
        pivot = df.pivot_table(
            index=['date', 'province'],
            columns='variable',
            values='value'
        ).reset_index()

        self.cache[cache_key] = pivot
        return pivot

# ==================== DATA GATHERING ====================
def gather_all_epi_data():
    """Rassemble toutes les données épidémiologiques avec géographie"""
    logger.info("🏥 Rassemblement données épidémiologiques...")

    # Query with joins to get zone and province
    records = ClinicalData.objects.select_related(
        'health_zone__territory__province'
    ).values(
        'health_zone_id',
        'health_zone__name',
        'health_zone__territory__province__name',
        'disease_name',
        'cases_count',
        'deaths_count',
        'month'
    )

    df = pd.DataFrame(list(records))

    if len(df) == 0:
        logger.error("❌ Aucune donnée clinique en base")
        return pd.DataFrame()

    # Rename columns
    df = df.rename(columns={
        'health_zone__name': 'zone_name',
        'health_zone__territory__province__name': 'province',
        'disease_name': 'disease',
        'cases_count': 'cases',
        'deaths_count': 'deaths',
        'health_zone_id': 'zone_id'
    })

    df['month'] = pd.to_datetime(df['month'])

    logger.info(f"  ✓ {len(df)} enregistrements collectés")
    logger.info(f"  • Zones: {df['zone_name'].nunique()}")
    logger.info(f"  • Provinces: {df['province'].nunique()}")
    logger.info(f"  • Maladies: {df['disease'].unique().tolist()}")
    logger.info(f"  • Période: {df['month'].min()} à {df['month'].max()}")

    return df

def add_climate_features(epi_df, climate_loader):
    """Ajoute variables climatiques aux données épidémiologiques"""
    logger.info("🌦️  Ajout données climatiques...")

    if 'temperature' in epi_df.columns:
        logger.info("  • Données climatiques déjà présentes")
        return epi_df

    climate_cols = ['temperature', 'precipitation', 'relative_humidity']
    for col in climate_cols:
        epi_df[col] = np.nan

    provinces_processed = set()
    total_merged = 0

    for province in epi_df['province'].unique():
        if province in provinces_processed:
            continue

        # Get date range for this province
        prov_data = epi_df[epi_df['province'] == province]
        start_date = pd.to_datetime(prov_data['month'].min())
        end_date = pd.to_datetime(prov_data['month'].max())

        # Load climate data
        climate_df = climate_loader.get_province_climate_timeseries(
            province, start_date, end_date
        )

        if climate_df is not None:
            # Merge on month/province
            prov_mask = epi_df['province'] == province
            for idx, row in epi_df[prov_mask].iterrows():
                month = row['month']
                clim_row = climate_df[
                    (climate_df['date'].dt.year == month.year) &
                    (climate_df['date'].dt.month == month.month)
                ]
                if len(clim_row) > 0:
                    for col in climate_cols:
                        if col in clim_row.columns:
                            epi_df.at[idx, col] = clim_row.iloc[0][col]
            total_merged += len(prov_data)
            logger.info(f"  ✓ {province}: {len(prov_data)} enregistrements enrichis")
        else:
            logger.warning(f"  ✗ {province}: pas de données climatiques")

        provinces_processed.add(province)

    # Check coverage
    coverage = epi_df[climate_cols].notna().sum().sum() / (len(epi_df) * len(climate_cols)) * 100
    logger.info(f"\n  • Couverture climatique globale: {coverage:.1f}%")

    return epi_df

def clean_and_fill_gaps(df):
    """Nettoie et comble les valeurs manquantes"""
    logger.info("🧹 Nettoyage et comblage trous...")

    original_len = len(df)

    # Sort by zone, disease, month
    df = df.sort_values(['zone_name', 'disease', 'month']).reset_index(drop=True)

    # Forward fill within each zone+disease group
    for (zone, disease), group in df.groupby(['zone_name', 'disease']):
        idx = group.index
        df.loc[idx, 'cases'] = df.loc[idx, 'cases'].ffill().bfill()
        for col in ['temperature', 'precipitation', 'relative_humidity']:
            if col in df.columns:
                df.loc[idx, col] = df.loc[idx, col].interpolate(method='linear').ffill().bfill()

    # If any remaining NaN, fill with group mean
    df['cases'] = df['cases'].fillna(df.groupby(['zone_name', 'disease'])['cases'].transform('mean'))
    df['cases'] = df['cases'].fillna(0)

    logger.info(f"  ✓ {len(df)}/{original_len} lignes après nettoyage")
    logger.info(f"  • NaN cases restants: {df['cases'].isna().sum()}")

    return df

# ==================== MODEL BUILDERS ====================
def build_tier2_model(input_shape, climate_features=3):
    """Tier 2: CNN-LSTM avec features climatiques"""
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        Dropout(0.2),
        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        Dropout(0.2),
        KerasLSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4)),
        Dropout(0.3),
        KerasLSTM(64, kernel_regularizer=l2(1e-4)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_tier3_model(input_shape):
    """Tier 3: Bi-LSTM"""
    model = Sequential([
        Bidirectional(KerasLSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4)),
                     input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(KerasLSTM(64, kernel_regularizer=l2(1e-4))),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ==================== TRAINING ====================
def prepare_sequences(data, sequence_length=6):
    """Crée séquences temporelles"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

def train_disease_models(epi_df, climate_loader):
    """Entraîne modèles par maladie (agrégé au niveau national)"""
    logger.info("\n" + "="*70)
    logger.info("🧠 ENTRAÎNEMENT MODÈLES (par maladie)")
    logger.info("="*70)

    metadata = {
        'training_date': datetime.now().isoformat(),
        'diseases': {},
        'models_trained': 0,
        'tier_distribution': {'Tier 2': 0, 'Tier 3': 0, 'Tier 4': 0}
    }

    for disease in Config.DISEASES:
        logger.info(f"\n{'='*60}")
        logger.info(f"Maladie: {disease}")
        logger.info(f"{'='*60}")

        # Agréger au niveau national (toutes zones confondues)
        disease_data = epi_df[epi_df['disease'] == disease].copy()
        disease_data = disease_data.groupby('month').agg({
            'cases': 'sum',
            'temperature': 'mean',
            'precipitation': 'mean',
            'relative_humidity': 'mean'
        }).reset_index().sort_values('month')

        n_months = len(disease_data)
        logger.info(f"  • Données: {n_months} mois agrégés")

        if n_months < Config.TIER3_MIN_MONTHS + 2:
            logger.warning(f"  ⚠️ Pas assez de données (<{Config.TIER3_MIN_MONTHS+2} mois)")
            continue

        # Check climate availability
        has_climate = disease_data[['temperature', 'precipitation', 'relative_humidity']].notna().all().all()
        logger.info(f"  • Données climatiques: {'Oui' if has_climate else 'Non'}")

        # Select tier
        if n_months >= Config.TIER2_MIN_MONTHS and has_climate:
            tier = 2
            tier_name = "CNN-LSTM"
        elif n_months >= Config.TIER3_MIN_MONTHS:
            tier = 3
            tier_name = "Bi-LSTM"
        else:
            tier = 4
            tier_name = "Persistence"
            logger.info(f"  • Tier {tier}: {tier_name}")
            continue  # No training needed for persistence

        logger.info(f"  • Tier {tier}: {tier_name}")

        # Prepare features
        features_list = ['cases']
        if tier == 2 and has_climate:
            features_list.extend(['temperature', 'precipitation', 'relative_humidity'])

        features = disease_data[features_list].values

        # Handle any remaining NaN
        features = np.nan_to_num(features, nan=np.nanmean(features, axis=0))

        # Scale
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features)

        # Create sequences
        X, y = prepare_sequences(features_scaled, Config.SEQUENCE_LENGTH)

        if len(X) < 3:
            logger.warning(f"  ⚠️ Pas assez de séquences ({len(X)})")
            continue

        # Split
        split = int(len(X) * (1 - Config.VALIDATION_SPLIT))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        logger.info(f"  • Séquences: train={len(X_train)}, test={len(X_test)}")
        logger.info(f"  • Input shape: {X_train.shape}")

        # Build & train model
        if tier == 2:
            model = build_tier2_model((Config.SEQUENCE_LENGTH, len(features_list)))
        else:
            model = build_tier3_model((Config.SEQUENCE_LENGTH, len(features_list)))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=Config.EARLY_STOPPING_PATIENCE,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=Config.EPOCHS_TIER2 if tier == 2 else Config.EPOCHS_TIER3,
            batch_size=Config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=0
        )

        # Evaluate
        y_pred = model.predict(X_test, verbose=0)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        logger.info(f"  • Époques: {len(history.history['loss'])}")
        logger.info(f"  • MAE: {mae:.4f}")
        logger.info(f"  • RMSE: {rmse:.4f}")
        logger.info(f"  • R²: {r2:.4f}")

        # Save model and scaler
        safe_name = disease.replace('/', '_').replace(' ', '_')
        model_path = Config.MODELS_DIR / f"model_{safe_name}_tier{tier}.keras"
        scaler_path = Config.MODELS_DIR / f"scaler_{safe_name}.pkl"

        model.save(model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        logger.info(f"  ✅ Modèle sauvé: {model_path.name}")

        # Update metadata
        metadata['diseases'][disease] = {
            'tier': tier,
            'tier_name': tier_name,
            'months_of_data': n_months,
            'has_climate': has_climate,
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'model_path': str(model_path),
            'scaler_path': str(scaler_path)
        }
        metadata['tier_distribution'][f"Tier {tier}"] += 1
        metadata['models_trained'] += 1

        gc.collect()

    # Save metadata
    metadata['last_updated'] = datetime.now().isoformat()
    with open(Config.DASHBOARD_DATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"\n✅ {metadata['models_trained']} modèles entraînés")
    logger.info(f"📊 Métadonnées: {Config.DASHBOARD_DATA_FILE.name}")

    return metadata

# ==================== PREDICTION GENERATION ====================
def generate_predictions(metadata):
    """Génère prédictions pour toutes les zones"""
    logger.info("\n" + "="*70)
    logger.info("🔮 GÉNÉRATION PRÉDICTIONS PAR ZONE")
    logger.info("="*70)

    zones = HealthZone.objects.select_related('territory__province').all()
    logger.info(f"  Zones à prédire: {zones.count()}")

    model_registry = {}
    for disease, info in metadata['diseases'].items():
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(info['model_path'])
            with open(info['scaler_path'], 'rb') as f:
                scaler = pickle.load(f)
            model_registry[disease] = {
                'model': model,
                'scaler': scaler,
                'tier': info['tier'],
                'features': ['cases', 'temperature', 'precipitation', 'relative_humidity'] if info['tier'] == 2 else ['cases']
            }
            logger.info(f"  ✓ {disease}: modèle chargé (Tier {info['tier']})")
        except Exception as e:
            logger.error(f"  ✗ {disease}: {e}")

    total_preds = 0

    for zone in zones:
        province = zone.territory.province.name if zone.territory else None

        for disease, model_info in model_registry.items():
            # Get historical data for this zone
            historical = ClinicalData.objects.filter(
                health_zone=zone,
                disease_name=disease
            ).order_by('month')

            if historical.count() < Config.SEQUENCE_LENGTH:
                continue

            # Build feature vector
            hist_df = pd.DataFrame(list(historical.values('month', 'cases_count')))
            hist_df = hist_df.rename(columns={'cases_count': 'cases'})
            hist_df['month'] = pd.to_datetime(hist_df['month'])
            hist_df = hist_df.sort_values('month')

            # Get climate data if needed
            if model_info['tier'] == 2 and province:
                # Fetch latest climate data
                latest_month = hist_df['month'].max()
                climate = ClimateData.objects.filter(
                    health_zone=zone,
                    month__year=latest_month.year,
                    month__month=latest_month.month
                ).first()

                if climate:
                    hist_df['temperature'] = climate.temperature
                    hist_df['precipitation'] = climate.precipitation
                    hist_df['relative_humidity'] = climate.humidity or 60
                else:
                    # Use province averages
                    hist_df['temperature'] = 25
                    hist_df['precipitation'] = 100
                    hist_df['relative_humidity'] = 60

            # Prepare sequence
            features = model_info['features']
            for feat in features:
                if feat not in hist_df.columns:
                    hist_df[feat] = 0  # Default

            seq_data = hist_df[features].values[-Config.SEQUENCE_LENGTH:]

            # Scale
            try:
                seq_scaled = model_info['scaler'].transform(seq_data)
                seq_input = seq_scaled.reshape(1, Config.SEQUENCE_LENGTH, len(features))

                # Predict
                pred_scaled = model_info['model'].predict(seq_input, verbose=0)
                prediction = model_info['scaler'].inverse_transform(
                    np.hstack([pred_scaled, np.zeros((1, len(features)-1))])[:, 0].reshape(-1, 1)
                )[0, 0]

                prediction = max(0, prediction)

                # Calculer le taux de propagation basé sur les tendances historiques
                propagation_rate = DiseaseAnalyzer.calculate_propagation_rate(
                    health_zone=zone,
                    disease_name=disease,
                    months_window=6  # utiliser les 6 derniers mois pour tendance récente
                )

                # Save prediction
                pred_date = hist_df['month'].max() + relativedelta(months=1)

                Prediction.objects.update_or_create(
                    health_zone=zone,
                    disease_name=disease,
                    prediction_date=pred_date,
                    defaults={
                        'predicted_cases': round(prediction, 1),
                        'propagation_rate': propagation_rate,
                        'influence_factors': {
                            'reliability': max(50, min(99, int(metadata['diseases'][disease]['r2'] * 100))),
                            'model_used': f"Tier {model_info['tier']}",
                            'tier': model_info['tier'],
                            'data_months': len(hist_df)
                        }
                    }
                )
                total_preds += 1
            except Exception as e:
                logger.warning(f"  ⚠️ {zone.name}/{disease}: {e}")

    logger.info(f"  ✅ {total_preds} prédictions générées")

# ==================== MAIN ====================
def main():
    logger.info("="*70)
    logger.info("🏥 HEALTHDRC - ENTRAÎNEMENT COMPLET v2.0")
    logger.info("="*70)

    try:
        # 1. Load data
        epi_df = gather_all_epi_data()
        if len(epi_df) == 0:
            return False

        # 2. Add climate
        climate_loader = ClimateDataLoader()
        epi_df = add_climate_features(epi_df, climate_loader)

        # 3. Clean data
        epi_df = clean_and_fill_gaps(epi_df)

        # Save processed
        epi_df.to_csv(Config.PROCESSED_DIR / 'epi_clean.csv', index=False)
        logger.info(f"✅ Données nettoyées sauvées: {Config.PROCESSED_DIR / 'epi_clean.csv'}")

        # 4. Train models
        metadata = train_disease_models(epi_df, climate_loader)

        # 5. Generate predictions
        generate_predictions(metadata)

        # 6. Print summary
        logger.info("\n" + "="*70)
        logger.info("✅ PIPELINE TERMINÉ")
        logger.info("="*70)
        logger.info(f"  • Modèles: {metadata['models_trained']}")
        for disease, info in metadata['diseases'].items():
            logger.info(f"    - {disease}: Tier {info['tier']} (R²={info['r2']:.3f})")
        logger.info(f"  • Dashboard: http://localhost:8000/analysis/dashboard/")
        logger.info("="*70)

        return True

    except Exception as e:
        logger.error(f"\n❌ ERREUR: {e}", exc_info=True)
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
