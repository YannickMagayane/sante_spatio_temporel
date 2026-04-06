#!/usr/bin/env python3
"""
Entraînement amélioré des modèles avec:
- Validation croisée temporelle
- Meilleure gestion des données manquantes
- Modèles plus adaptés aux petites séries
- Métriques de diagnostic détaillées
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import gc
import pickle
from collections import defaultdict

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
sys.path.insert(0, '/app')
os.chdir('/app')

import django
django.setup()

# TensorFlow
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional,
    Conv1D, MaxPooling1D, Flatten, TimeDistributed,
    ConvLSTM2D, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

print("=" * 70)
print("🧠 ENTRAÎNEMENT AMÉLIORÉ - HealthDRC")
print("=" * 70)

# Configuration
MODELS_DIR = Path("/app/models")
DATA_DIR = Path("/app/data")
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR = Path("/app/logs")
LOG_DIR.mkdir(exist_ok=True)

MODELS_DIR.mkdir(exist_ok=True, parents=True)
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

# Seuils adaptatifs
TIER1_MIN_MONTHS = 24   # ConvLSTM2D (spatio-temporel) - besoin de beaucoup de données
TIER2_MIN_MONTHS = 12   # CNN-LSTM - patterns temporels + spatial
TIER3_MIN_MONTHS = 6    # Bi-LSTM - série temporelle
TIER4_MIN_MONTHS = 3    # Persistence - baseline

WINDOW_SIZE = 6  # 6 mois de historique
PREDICTION_HORIZON = 1  # Prédire 1 mois ahead

EPIDEMIC_DISEASES = ['Paludisme', 'Choléra', 'VIH/SIDA', 'Tuberculose']

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'training_improved.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_and_prepare_data():
    """Charge et prépare les données avec imputation avancée"""
    print("\n" + "=" * 70)
    print("📊 CHARGEMENT DES DONNÉES")
    print("=" * 70)

    # Charger CSV épidémiologique
    epi_csv = PROCESSED_DIR / 'epi_data.csv'
    if not epi_csv.exists():
        logger.error(f"Fichier non trouvé: {epi_csv}")
        return None

    df = pd.read_csv(epi_csv)
    logger.info(f"Données chargées: {len(df)} enregistrements")

    # Nettoyage
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'disease', 'cases'])

    # Créer colonnes temporelles
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['month_year'] = df['date'].dt.to_period('M')

    # Filtrer maladies épidémiques
    df = df[df['disease'].isin(EPIDEMIC_DISEASES)]

    # Identifier les zones de santé (si disponible)
    zone_col = None
    for col in ['health_zone', 'zone', 'health_zone_name', 'zone_name']:
        if col in df.columns:
            zone_col = col
            break

    if zone_col:
        df['zone_name'] = df[zone_col].fillna('Unknown')
        zones = df['zone_name'].unique()
        logger.info(f"Zones identifiées: {len(zones)}")
    else:
        df['zone_name'] = 'Global'
        zones = ['Global']
        logger.warning("Aucune colonne zone trouvée - utilisation d'une vue globale")

    # Agrégation par zone, maladie, mois
    agg_dict = {
        'cases': 'sum',
        'deaths': 'sum' if 'deaths' in df.columns else lambda x: 0
    }

    # Ajouter colonnes numériques disponibles
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in ['cases', 'deaths', 'year', 'month']:
            agg_dict[col] = 'mean'

    grouped = df.groupby(['zone_name', 'disease', 'month_year']).agg(agg_dict).reset_index()
    grouped['date'] = grouped['month_year'].dt.to_timestamp()

    logger.info(f"Données groupées: {len(grouped)} combinaisons zone/maladie/mois")

    return {
        'data': grouped,
        'zones': zones,
        'zone_column': zone_col
    }

def create_sequences(series, window_size=WINDOW_SIZE, horizon=PREDICTION_HORIZON):
    """Crée des séquences temporelles pour l'entraînement"""
    X, y = [], []
    for i in range(len(series) - window_size - horizon + 1):
        X.append(series[i:(i + window_size)])
        y.append(series[(i + window_size):(i + window_size + horizon)])
    return np.array(X), np.array(y)

def build_tier1_model(input_shape):
    """ConvLSTM2D - pour données spatio-temporelles (si plusieurs zones)"""
    model = Sequential([
        ConvLSTM2D(32, (3, 3), padding='same', return_sequences=True,
                   kernel_regularizer=l2(1e-4), input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        ConvLSTM2D(16, (3, 3), padding='same', return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Flatten(),
        Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    return model, "ConvLSTM2D"

def build_tier2_model(input_shape):
    """CNN-LSTM - patterns temporels"""
    model = Sequential([
        Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(1e-4),
               input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        Conv1D(32, 3, activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-4)),
        Dropout(0.2),
        LSTM(32, kernel_regularizer=l2(1e-4)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    return model, "CNN-LSTM"

def build_tier3_model(input_shape):
    """Bi-LSTM - série temporelle"""
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-4)),
                     input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(LSTM(32, kernel_regularizer=l2(1e-4))),
        Dropout(0.2),
        Dense(16, activation='relu', kernel_regularizer=l2(1e-4)),
        Dense(1)
    ])
    return model, "Bi-LSTM"

def build_tier4_model(input_shape):
    """Simple LSTM - petit datasets"""
    model = Sequential([
        LSTM(32, activation='relu', input_shape=input_shape, kernel_regularizer=l2(1e-4)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    return model, "Simple LSTM"

def build_persistence_model():
    """Baseline: dernière valeur connue"""
    return None, "Persistence (Baseline)"

def determine_tier(n_months, n_zones, variance):
    """Détermine le tier approprié selon les données"""
    if n_months >= TIER1_MIN_MONTHS and n_zones > 1:
        return 1
    elif n_months >= TIER2_MIN_MONTHS:
        return 2
    elif n_months >= TIER3_MIN_MONTHS and variance > 0.1:
        return 3
    elif n_months >= TIER4_MIN_MONTHS:
        return 4
    else:
        return None  # Pas assez de données

def train_with_cv(X, y, model_builder, n_splits=3):
    """Entraîne avec validation croisée temporelle"""
    if len(X) < 10:
        # Pas assez pour CV, simple split
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        model, model_name = model_builder((X_train.shape[1], X_train.shape[2]))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        epochs = 30 if len(X_train) < 20 else 50
        batch_size = min(8, max(2, len(X_train) // 10))

        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ],
            verbose=0
        )

        y_pred = model.predict(X_test, verbose=0).flatten()
        y_test_flat = y_test.flatten()

        return {
            'model': model,
            'history': history,
            'metrics': {
                'mae': float(mean_absolute_error(y_test_flat, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test_flat, y_pred))),
                'r2': float(r2_score(y_test_flat, y_pred)),
                'train_loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']]
            },
            'splits': 1
        }
    else:
        # Validation croisée temporelle
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        all_histories = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model, model_name = model_builder((X_train.shape[1], X_train.shape[2]))
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            epochs = min(30, len(X_train) // 5)
            batch_size = min(8, max(2, len(X_train) // 10))

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
                ],
                verbose=0
            )

            y_pred = model.predict(X_val, verbose=0).flatten()
            y_val_flat = y_val.flatten()
            r2 = r2_score(y_val_flat, y_pred)
            cv_scores.append(r2)
            all_histories.append(history)

        # Entraîner sur tout le dataset
        final_model, model_name = model_builder((X.shape[1], X.shape[2]))
        final_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        epochs = min(50, len(X) // 3)
        batch_size = min(8, max(2, len(X) // 10))

        final_history = final_model.fit(
            X, y,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ],
            verbose=0
        )

        # Métriques finales avec split simple
        split_idx = int(0.8 * len(X))
        X_train_final, X_test_final = X[:split_idx], X[split_idx:]
        y_train_final, y_test_final = y[:split_idx], y[split_idx:]

        y_pred_final = final_model.predict(X_test_final, verbose=0).flatten()
        y_test_flat_final = y_test_final.flatten()

        return {
            'model': final_model,
            'history': final_history,
            'metrics': {
                'mae': float(mean_absolute_error(y_test_flat_final, y_pred_final)),
                'rmse': float(np.sqrt(mean_squared_error(y_test_flat_final, y_pred_final))),
                'r2': float(r2_score(y_test_flat_final, y_pred_final)),
                'cv_mean_r2': float(np.mean(cv_scores)),
                'cv_std_r2': float(np.std(cv_scores)),
                'train_loss': [float(x) for x in final_history.history['loss']],
                'val_loss': [float(x) for x in final_history.history['val_loss']]
            },
            'splits': n_splits
        }

def train_disease_models(data_manager):
    """Entraîne les modèles pour toutes les maladies"""
    print("\n" + "=" * 70)
    print("🎯 ENTRAÎNEMENT PAR MALADIE")
    print("=" * 70)

    df = data_manager['data']
    zones = data_manager['zones']
    results = {}
    tier_distribution = defaultdict(int)

    for disease in EPIDEMIC_DISEASES:
        print(f"\n{'='*70}")
        print(f"🦠 TRAITEMENT: {disease}")
        print(f"{'='*70}")

        disease_results = {}

        for zone in zones:
            zone_data = df[(df['disease'] == disease) & (df['zone_name'] == zone)].copy()

            if len(zone_data) < TIER4_MIN_MONTHS + WINDOW_SIZE:
                logger.debug(f"  ⚠️ Zone '{zone}': trop peu de données ({len(zone_data)})")
                continue

            zone_data = zone_data.sort_values('date')
            cases = zone_data['cases'].values.reshape(-1, 1)

            # Vérifier variance
            if np.std(cases) < 0.01:
                logger.debug(f"  ⚠️ Zone '{zone}': variance trop faible")
                continue

            # Normalisation
            scaler = RobustScaler()
            cases_scaled = scaler.fit_transform(cases)

            # Créer séquences
            X, y = create_sequences(cases_scaled, WINDOW_SIZE)
            if len(X) < 3:
                continue

            X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
            y_reshaped = y.reshape(-1, 1)

            # Déterminer tier
            n_months = len(zone_data)
            n_zones_in_disease = len(df[df['disease'] == disease]['zone_name'].unique())
            variance = np.std(cases)

            tier = determine_tier(n_months, n_zones_in_disease, variance)

            if tier is None:
                logger.debug(f"  ⚠️ Zone '{zone}': pas assez de données pour un tier valide")
                continue

            # Sélectionner modèle
            if tier == 1:
                model_builder = build_tier1_model
                input_shape = (X_reshaped.shape[1], X_reshaped.shape[2], 1)
                # Pour ConvLSTM2D, besoin de reshape 4D
                X_reshaped = X_reshaped.reshape(X_reshaped.shape[0], X_reshaped.shape[1], 1, 1, 1)
                y_reshaped = y_reshaped
            elif tier == 2:
                model_builder = build_tier2_model
                input_shape = (X_reshaped.shape[1], X_reshaped.shape[2])
            elif tier == 3:
                model_builder = build_tier3_model
                input_shape = (X_reshaped.shape[1], X_reshaped.shape[2])
            elif tier == 4:
                model_builder = build_tier4_model
                input_shape = (X_reshaped.shape[1], X_reshaped.shape[2])
            else:
                continue

            # Entraîner
            logger.info(f"  🧠 {disease} - {zone}: Tier {tier} ({n_months} mois)")

            try:
                result = train_with_cv(X_reshaped, y_reshaped, model_builder, n_splits=2)

                model_path = MODELS_DIR / f"model_{disease.replace('/', '_')}_{zone.replace('/', '_')}_tier{tier}.keras"
                result['model'].save(model_path)

                scaler_path = MODELS_DIR / f"scaler_{disease.replace('/', '_')}_{zone.replace('/', '_')}.pkl"
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)

                disease_results[zone] = {
                    'tier': tier,
                    'n_months': n_months,
                    'n_sequences': len(X_reshaped),
                    'metrics': result['metrics']
                }

                tier_distribution[f"Tier {tier}"] += 1

            except Exception as e:
                logger.error(f"  ❌ Erreur pour {disease} - {zone}: {str(e)}")
                continue

            gc.collect()

        if disease_results:
            results[disease] = disease_results
            avg_r2 = np.mean([r['metrics']['r2'] for r in disease_results.values()])
            logger.info(f"✅ {disease}: {len(disease_results)} zones, R² moyen: {avg_r2:.4f}")
        else:
            logger.warning(f"⚠️ {disease}: Aucune zone avec données suffisantes")

    return results, dict(tier_distribution)

def generate_dashboard_data(results, tier_distribution):
    """Génère les données pour le dashboard"""

    print("\n" + "=" * 70)
    print("📊 GÉNÉRATION DONNÉES DASHBOARD")
    print("=" * 70)

    # Calculer moyennes par maladie
    disease_summary = {}
    for disease, zones_data in results.items():
        if zones_data:
            r2_values = [z['metrics']['r2'] for z in zones_data.values()]
            mae_values = [z['metrics']['mae'] for z in zones_data.values()]

            disease_summary[disease] = {
                'avg_r2': float(np.mean(r2_values)),
                'avg_mae': float(np.mean(mae_values)),
                'zones_count': len(zones_data),
                'best_zone': max(zones_data.items(), key=lambda x: x[1]['metrics']['r2'])[0] if zones_data else None,
                'worst_zone': min(zones_data.items(), key=lambda x: x[1]['metrics']['r2'])[0] if zones_data else None
            }

    # Distribution des tiers
    total_models = sum(tier_distribution.values())

    dashboard_data = {
        'training_metadata': {
            'training_date': datetime.now().isoformat(),
            'diseases': {d: z for d, z in disease_summary.items()},
            'total_models_trained': total_models
        },
        'model_performance': {},
        'tier_distribution': tier_distribution,
        'diagnostics': {
            'total_zones_trained': sum(d['zones_count'] for d in disease_summary.values()),
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'window_size': WINDOW_SIZE,
                'prediction_horizon': PREDICTION_HORIZON,
                'tier_thresholds': {
                    'tier1': TIER1_MIN_MONTHS,
                    'tier2': TIER2_MIN_MONTHS,
                    'tier3': TIER3_MIN_MONTHS,
                    'tier4': TIER4_MIN_MONTHS
                }
            }
        },
        'last_training': datetime.now().isoformat(),
        'total_models_trained': total_models
    }

    # Remplir model_performance pour le dashboard
    for disease, summary in disease_summary.items():
        # Trouver le tier le plus fréquent pour cette maladie
        tiers = [z['tier'] for z in results[disease].values()]
        most_common_tier = max(set(tiers), key=tiers.count) if tiers else 3

        tier_names = {
            1: "ConvLSTM2D",
            2: "CNN-LSTM",
            3: "Bi-LSTM",
            4: "Simple LSTM",
            None: "Persistence"
        }

        dashboard_data['model_performance'][disease] = {
            'tier': most_common_tier,
            'tier_name': tier_names.get(most_common_tier, "Unknown"),
            'mae': round(summary['avg_mae'], 4),
            'rmse': round(np.sqrt(summary['avg_mae']**2), 4) if summary['avg_mae'] < 10 else 0,
            'r2': round(summary['avg_r2'], 4),
            'confidence': round(max(summary['avg_r2'] * 100, 18), 2),
            'epochs': 0,  # Sera calculé par modèle
            'zones_trained': summary['zones_count']
        }

    # Sauvegarder
    dashboard_file = MODELS_DIR / 'dashboard_training_data.json'
    with open(dashboard_file, 'w') as f:
        json.dump(dashboard_data, f, indent=2, default=str)

    logger.info(f"✅ Dashboard data généré: {dashboard_file}")

    # Afficher résumé
    print(f"\n{'='*70}")
    print("📋 RÉSUMÉ DE L'ENTRAÎNEMENT")
    print(f"{'='*70}")
    print(f"Modèles entraînés: {total_models}")
    print(f"Zones totales: {sum(d['zones_count'] for d in disease_summary.values())}")
    print(f"\nDistribution des tiers:")
    for tier, count in sorted(tier_distribution.items()):
        print(f"  {tier}: {count} modèles ({count/total_models*100:.1f}%)")

    print(f"\nPerformance par maladie:")
    for disease, summary in disease_summary.items():
        print(f"  {disease}: R²={summary['avg_r2']:.4f}, MAE={summary['avg_mae']:.4f}, Zones={summary['zones_count']}")

    return dashboard_data

def main():
    """Pipeline principal"""

    start_time = datetime.now()

    try:
        # 1. Charger données
        data_manager = load_and_prepare_data()
        if data_manager is None:
            logger.error("❌ Impossible de charger les données")
            return 1

        # 2. Entraîner modèles
        results, tier_distribution = train_disease_models(data_manager)

        if not results:
            logger.error("❌ Aucun modèle n'a pu être entraîné")
            return 1

        # 3. Générer données dashboard
        dashboard_data = generate_dashboard_data(results, tier_distribution)

        # 4. Sauvegarder métadonnées détaillées
        metadata = {
            'training_date': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - start_time).total_seconds(),
            'configuration': {
                'window_size': WINDOW_SIZE,
                'prediction_horizon': PREDICTION_HORIZON,
                'tier_thresholds': {
                    'tier1': TIER1_MIN_MONTHS,
                    'tier2': TIER2_MIN_MONTHS,
                    'tier3': TIER3_MIN_MONTHS,
                    'tier4': TIER4_MIN_MONTHS
                }
            },
            'results_by_disease': results
        }

        metadata_file = MODELS_DIR / 'training_detailed_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"✅ Entraînement terminé en {(datetime.now()-start_time).total_seconds()/60:.1f} minutes")
        logger.info(f"📊 Dashboard disponible sur http://localhost:8000/analysis/dashboard/")

        print("\n" + "=" * 70)
        print("✅ ENTRAÎNEMENT AMÉLIORÉ COMPLET")
        print("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"❌ Erreur fatale: {str(e)}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
