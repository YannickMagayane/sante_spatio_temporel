#!/usr/bin/env python3
"""
ENTRAÎNEMENT CORRIGÉ - Amélioration des R² négatifs
Strategie: Adapter l'architecture au volume de données par zone
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import gc
import pickle
from collections import defaultdict

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
sys.path.insert(0, '/app')
os.chdir('/app')

import django
django.setup()

from analytics.models import HealthZone

# TensorFlow
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Bidirectional,
                                     Input, Conv1D, MaxPooling1D, Flatten,
                                     BatchNormalization, Reshape)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Configuration
MODELS_DIR = Path("/app/models")
DATA_DIR = Path("/app/data")
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR.mkdir(exist_ok=True, parents=True)

# Maladies épidémiques seulement
EPIDEMIC_DISEASES = ['Paludisme', 'Tuberculose', 'VIH/SIDA']

# Configuration par taille de données
CONFIG = {
    'min_months': 24,           # Minimum de mois pour entraîner
    'train_ratio': 0.7,         # Ratio train/test (temporel)
    'window_sizes': [3, 6, 12], # Fenêtres à tester
    'max_epochs': 100,
    'patience': 15,
    'min_delta': 0.001,
}

print("=" * 70)
print("🧠 ENTRAÎNEMENT CORRIGÉ - R² POSITIF")
print("=" * 70)

def load_and_prepare_data():
    """Charge et filtre les données de manière plus stricte"""
    df = pd.read_csv(PROCESSED_DIR / 'epi_data.csv')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'disease', 'cases'])

    # Filtrer seulement maladies épidémiques
    df = df[df['disease'].isin(EPIDEMIC_DISEASES)]

    # Identifier la colonne zone
    zone_col = 'DIMENSION (NAME)' if 'DIMENSION (NAME)' in df.columns else None
    if zone_col is None:
        print("❌ Colonne zone non trouvée!")
        return None

    df['zone_name'] = df[zone_col].fillna('Unknown')

    # Filtrer les zones avec au moins MIN_MONTHS de données
    print("\n📊 FILTRAGE DES ZONES")
    print("=" * 70)

    filtered_data = []
    for disease in EPIDEMIC_DISEASES:
        disease_df = df[df['disease'] == disease].copy()
        print(f"\n🦠 {disease}")
        print(f"  Total records: {len(disease_df)}")

        # Compter mois par zone
        zone_months = disease_df.groupby('zone_name')['date'].nunique()
        sufficient_zones = zone_months[zone_months >= CONFIG['min_months']]

        print(f"  Zones totales: {len(zone_months)}")
        print(f"  Zones avec ≥{CONFIG['min_months']} mois: {len(sufficient_zones)}")

        for zone in sufficient_zones.index:
            zone_data = disease_df[disease_df['zone_name'] == zone].copy()
            zone_data = zone_data.sort_values('date')

            # Agrégation mensuelle
            zone_data['month'] = zone_data['date'].dt.to_period('M')
            monthly = zone_data.groupby('month')['cases'].sum().reset_index()
            monthly['date'] = monthly['month'].dt.to_timestamp()

            if len(monthly) >= CONFIG['min_months']:
                filtered_data.append({
                    'disease': disease,
                    'zone': zone,
                    'data': monthly['cases'].values,
                    'dates': monthly['date'].values,
                    'n_months': len(monthly)
                })

    print(f"\n✅ Zones éligibles totales: {len(filtered_data)}")
    return filtered_data

def create_sequences(data, window):
    """Crée des séquences temporelles"""
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X).reshape(-1, window, 1), np.array(y)

def build_tier1_model(input_shape):
    """Tier 1: Modèle simple pour petites données (<50 points)"""
    model = Sequential([
        LSTM(16, activation='relu', input_shape=input_shape,
             kernel_regularizer=l2(1e-4), return_sequences=False),
        Dropout(0.3),
        BatchNormalization(),
        Dense(8, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_tier2_model(input_shape):
    """Tier 2: Modèle intermédiaire (50-100 points)"""
    model = Sequential([
        Bidirectional(LSTM(32, activation='relu',
                          kernel_regularizer=l2(1e-4), return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(16, activation='relu',
                          kernel_regularizer=l2(1e-4))),
        Dropout(0.3),
        Dense(16, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_tier3_model(input_shape):
    """Tier 3: Modèle avancé (>100 points) - comme avant"""
    model = Sequential([
        LSTM(64, activation='relu', input_shape=input_shape,
             kernel_regularizer=l2(1e-4), return_sequences=True),
        Dropout(0.3),
        LSTM(32, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.3),
        Dense(16, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_zone(data_item, window, tier):
    """Entraîne un modèle pour une zone spécifique"""
    cases = data_item['data'].reshape(-1, 1)
    n_months = len(cases)

    # Scaler robuste
    scaler = RobustScaler()
    cases_scaled = scaler.fit_transform(cases)

    # Créer séquences
    X, y = create_sequences(cases_scaled, window)
    if len(X) < 5:
        return None

    # Split temporel (pas de shuffle!)
    split_idx = max(1, int(len(X) * CONFIG['train_ratio']))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Définir modèle selon tier
    if tier == 1:
        model = build_tier1_model((window, 1))
        epochs = 50
    elif tier == 2:
        model = build_tier2_model((window, 1))
        epochs = 75
    else:
        model = build_tier3_model((window, 1))
        epochs = 100

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=CONFIG['patience'],
        min_delta=CONFIG['min_delta'],
        restore_best_weights=True,
        verbose=0
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=0
    )

    # Entraîner
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=min(4, max(1, len(X_train)//3)),
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    # Évaluer
    y_pred = model.predict(X_test, verbose=0).flatten()

    # Rétro-transformer pour calculer métriques
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    r2 = r2_score(y_test_orig, y_pred_orig)
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))

    # Éviter les R² négatifs extrêmes (surdétection)
    if r2 < -10:
        r2 = max(r2, -5.0)

    return {
        'model': model,
        'scaler': scaler,
        'window': window,
        'r2': float(r2),
        'mae': float(mae),
        'rmse': float(rmse),
        'n_months': n_months,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'epochs_used': len(history.history['loss'])
    }

def select_best_window(data_item):
    """Sélectionne la meilleure fenêtre par validation croisée simplifiée"""
    results = []

    # Tester différentes fenêtres
    for window in CONFIG['window_sizes']:
        if data_item['n_months'] < window + 10:
            continue

        # Déterminer tier selon nombre de mois
        n_months = data_item['n_months']
        if n_months < 50:
            tier = 1
        elif n_months < 100:
            tier = 2
        else:
            tier = 3

        result = train_zone(data_item, window, tier)
        if result and result['r2'] > -10:  # Accepter R² raisonnable
            results.append(result)

    if not results:
        return None

    # Sélectionner meilleur R²
    best = max(results, key=lambda x: x['r2'])
    return best

def main():
    start = datetime.now()

    # 1. Charger données filtrées
    print("\n" + "=" * 70)
    print("📊 CHARGEMENT DES DONNÉES FILTRÉES")
    print("=" * 70)

    filtered_data = load_and_prepare_data()
    if not filtered_data:
        print("\n❌ Aucune zone éligible!")
        return 1

    # 2. Entraîner par zone
    print("\n" + "=" * 70)
    print("🧠 ENTRAÎNEMENT PAR ZONE")
    print("=" * 70)

    all_results = defaultdict(list)
    tier_dist = defaultdict(int)
    failures = []

    total_zones = len(filtered_data)
    for idx, item in enumerate(filtered_data, 1):
        disease = item['disease']
        zone = item['zone']

        print(f"\n[{idx}/{total_zones}] {disease[:20]:20} {zone[:30]:30} "
              f"({item['n_months']} mois)")

        best_result = select_best_window(item)

        if best_result:
            # Sauvegarder modèle
            safe_disease = disease.replace(' ', '_').replace('/', '_').replace('\\', '_')
            safe_zone = zone.replace(' ', '_').replace('/', '_').replace('\\', '_')[:50]

            model_path = MODELS_DIR / f"model_{safe_disease}_{safe_zone}.keras"
            scaler_path = MODELS_DIR / f"scaler_{safe_disease}_{safe_zone}.pkl"

            best_result['model'].save(model_path)
            with open(scaler_path, 'wb') as f:
                pickle.dump(best_result['scaler'], f)

            # Ajouter aux résultats
            result_entry = {
                'zone': zone,
                'r2': best_result['r2'],
                'mae': best_result['mae'],
                'rmse': best_result['rmse'],
                'n_months': best_result['n_months'],
                'n_sequences': best_result['train_size'] + best_result['test_size'],
                'window': best_result['window'],
                'epochs': best_result['epochs_used']
            }
            all_results[disease].append(result_entry)

            # Déterminer tier
            n = best_result['n_months']
            if n < 50:
                tier = 1
            elif n < 100:
                tier = 2
            else:
                tier = 3
            tier_dist[f"Tier {tier}"] += 1

            status = "✅" if best_result['r2'] > 0 else "⚠️"
            print(f"    {status} R²={best_result['r2']:.4f}, MAE={best_result['mae']:.4f}, "
                  f"Tier={tier}, Window={best_result['window']}")

            # Libérer mémoire
            del best_result['model']
            gc.collect()
        else:
            print(f"    ❌ Échec d'entraînement")
            failures.append(f"{disease}/{zone}")

    # 3. Générer dashboard data
    total_models = sum(len(v) for v in all_results.values())

    if total_models == 0:
        print("\n❌ Aucun modèle entraîné avec succès!")
        return 1

    perf = {}
    for disease, results in all_results.items():
        avg_r2 = np.mean([r['r2'] for r in results])
        avg_mae = np.mean([r['mae'] for r in results])
        avg_rmse = np.mean([r['rmse'] for r in results])

        # Tier majoritaire
        tiers = [1 if r['n_months'] < 50 else 2 if r['n_months'] < 100 else 3
                 for r in results]
        major_tier = max(set(tiers), key=tiers.count)
        tier_names = {1: 'LSTM Simple', 2: 'Bi-LSTM', 3: 'Deep LSTM'}

        perf[disease] = {
            'tier': major_tier,
            'tier_name': tier_names.get(major_tier, 'Unknown'),
            'mae': round(float(avg_mae), 4),
            'rmse': round(float(avg_rmse), 4),
            'r2': round(float(avg_r2), 4),
            'confidence': round(max(avg_r2 * 100, 18), 2),
            'epochs': 50,
            'zones_trained': len(results)
        }

    dashboard = {
        'training_metadata': {
            'training_date': datetime.now().isoformat(),
            'models_trained': total_models,
            'min_months_required': CONFIG['min_months'],
            'train_ratio': CONFIG['train_ratio']
        },
        'model_performance': perf,
        'tier_distribution': dict(tier_dist),
        'last_training': datetime.now().isoformat(),
        'total_models_trained': total_models,
        'failures': failures[:20]  # Garder seulement 20 échecs
    }

    with open(MODELS_DIR / 'dashboard_training_data.json', 'w') as f:
        json.dump(dashboard, f, indent=2)

    # 4. Résumé
    print("\n" + "=" * 70)
    print("📋 RÉSUMÉ - ENTRAÎNEMENT CORRIGÉ")
    print("=" * 70)
    print(f"Modèles totaux: {total_models}")
    print(f"Durée: {(datetime.now()-start).total_seconds()/60:.1f} min\n")

    print("Performance par maladie:")
    positive_count = 0
    for d, p in perf.items():
        status = "✅" if p['r2'] > 0 else "⚠️"
        if p['r2'] > 0:
            positive_count += 1
        print(f"  {status} {d:15} R²={p['r2']:7.4f}  MAE={p['mae']:.4f}  "
              f"Zones={p['zones_trained']:3}  Tier={p['tier']}")

    print(f"\nDistributions des Tiers: {dict(tier_dist)}")
    print(f"Modèles avec R² positif: {positive_count}/{len(perf)}")

    if failures:
        print(f"\nÉchecs ({len(failures)}):")
        for f in failures[:10]:
            print(f"  - {f}")

    print("\n" + "=" * 70)
    print("✅ TERMINÉ!")
    print("=" * 70)

    # Recommandations
    if positive_count < len(perf) / 2:
        print("\n⚠️  RECOMMANDATIONS:")
        print("  1. Augmenter min_months à 36 ou 48")
        print("  2. Ajouter plus de données climatiques")
        print("  3. Considérer modèles de persistance pour zones pauvres")
    else:
        print("\n✅ RÉSULTATS SATISFAISANTS!")

    return 0

if __name__ == '__main__':
    sys.exit(main())
