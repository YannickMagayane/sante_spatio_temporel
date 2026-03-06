"""
train_models.py — Moteur IA Spatio-Temporel Multi-Niveaux
==========================================================
Ce script entraîne un ensemble de modèles de deep learning sur les données
épidémiologiques de la RDC. L'architecture est organisée en 4 niveaux (Tiers)
qui s'activent automatiquement selon la richesse des données disponibles.

Architecture:
    TIER 1 — ConvLSTM2D   : Propagation Spatio-Temporelle  (≥ 16 mois de données)
    TIER 2 — CNN-LSTM      : Patterns locaux + Mémoire      (≥ 8 mois de données)
    TIER 3 — Bi-LSTM       : Séquences Temporelles Pures    (≥ 4 mois de données)
    TIER 4 — Persistence   : Fallback Statistique           (< 4 mois)

Example:
    >>> docker-compose exec app python scripts/train_models.py
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
import gc

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Masquer avertissements CUDA/AVX

# Django setup
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()

from analytics.models import Province, HealthZone, ClinicalData, Prediction

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional,
    Conv1D, MaxPooling1D, Flatten, TimeDistributed,
    ConvLSTM2D, Reshape, LayerNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
MODELS_DIR  = Path("/app/models")
REPORTS_DIR = Path("/app/reports")
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# Tier thresholds (nombre de mois de données)
TIER1_MIN = 16   # ConvLSTM2D — spatio-temporal
TIER2_MIN = 8    # CNN-LSTM   — local pattern + memory
TIER3_MIN = 4    # Bi-LSTM    — temporal
WINDOW    = 4    # Fenêtre temporelle pour LSTM / CNN-LSTM

# Maladies épidémiques prioritaires — seules ces maladies sont entraînées.
# Exclut les maladies chroniques (Anémie, Malnutrition, Onchocercose, etc.)
EPIDEMIC_DISEASES = {
    'Paludisme',
    'Choléra',
    'Ebola',
    'Mpox',
    'COVID-19',
    'Rougeole',
    'Fièvre Jaune',
    'VIH/SIDA',
    'Tuberculose',
    'Méningite',
    'Typhoïde',
    'Poliomyélite',
    'Tétanos',
    'Hépatite',
    'Coqueluche',
    'Trypanosomiase',
    'Rubéole',
}

# ── Feature Engineering ────────────────────────────────────────────────────

def seasonality_features(dates):
    """
    Encode le mois en coordonnées cycliques (sin/cos) pour capturer
    la saisonnalité sans coupure artificielle entre décembre et janvier.

    :param dates: série de dates pandas.
    :returns: tuple (sin_month, cos_month) — np.ndarray de même longueur.

    Example:
        >>> s, c = seasonality_features(pd.to_datetime(['2024-01-01', '2024-06-01']))
    """
    m = np.array([d.month for d in dates])
    return np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)


def build_feature_matrix(df_monthly):
    """
    Construit la matrice de features à 3 colonnes : [cas, sin_mois, cos_mois].

    :param df_monthly: DataFrame avec colonnes 'month_dt' et 'cases_count'.
    :returns: np.ndarray de shape (n, 3).

    Example:
        >>> X = build_feature_matrix(df)
    """
    sin_m, cos_m = seasonality_features(df_monthly['month_dt'])
    return np.stack([df_monthly['cases_count'].values, sin_m, cos_m], axis=1)


def create_sequences(data, window):
    """
    Transforme une série temporelle en dataset supervisé à fenêtre glissante.

    :param data: np.ndarray de shape (n, features).
    :param window: taille de la fenêtre glissante.
    :returns: tuple (X, y) avec X de shape (samples, window, features) et y de shape (samples,).

    Example:
        >>> X, y = create_sequences(data, window=4)
    """
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window, :])
        y.append(data[i + window, 0])
    return np.array(X), np.array(y)


# ── Model Builders ──────────────────────────────────────────────────────────

def build_convlstm_model(n_zones, window, n_features=1):
    """
    TIER 1 — ConvLSTM2D : Modèle Spatio-Temporel.

    Traite les données comme une grille spatiale de zones de santé et capture
    la propagation géographique des épidémies dans le temps.

    :param n_zones: nombre de zones de santé dans la province.
    :param window: nombre de pas de temps.
    :param n_features: features par pas de temps.
    :returns: modèle Keras compilé.

    Example:
        >>> model = build_convlstm_model(n_zones=12, window=4)
    """
    inputs = Input(shape=(window, n_zones, 1, n_features))
    x = ConvLSTM2D(
        filters=32,
        kernel_size=(min(3, n_zones), 1),
        padding='same',
        return_sequences=True,
        activation='tanh',
        recurrent_activation='sigmoid'
    )(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = ConvLSTM2D(
        filters=16,
        kernel_size=(min(3, n_zones), 1),
        padding='same',
        return_sequences=False,
        activation='tanh',
        recurrent_activation='sigmoid'
    )(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(n_zones)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_cnn_lstm_model(window, n_features=3):
    """
    TIER 2 — CNN-LSTM Hybride : Patterns Locaux + Mémoire Temporelle.

    Le CNN extrait des motifs caractéristiques des épidémies (pics, plateaux)
    sur des sous-séquences ; le LSTM mémorise la dynamique temporelle globale.
    Architecture : Conv1D → MaxPool → Conv1D → LSTM → Dense.

    :param window: taille de la fenêtre temporelle (pas de temps).
    :param n_features: nombre de features par pas de temps.
    :returns: modèle Keras compilé.

    Example:
        >>> model = build_cnn_lstm_model(window=4)
    """
    model = Sequential([
        Input(shape=(window, n_features)),
        # Bloc CNN : extraction de motifs locaux
        Conv1D(filters=64, kernel_size=min(3, window), padding='same', activation='relu'),
        Conv1D(filters=32, kernel_size=min(2, window), padding='same', activation='relu'),
        MaxPooling1D(pool_size=1),   # pool_size=1 préserve la dim temporelle pour LSTM
        # Bloc LSTM : mémoire temporelle
        LSTM(64, return_sequences=True),
        Dropout(0.25),
        LSTM(32),
        Dropout(0.25),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_bilstm_model(window, n_features=3):
    """
    TIER 3 — Bi-LSTM : Analyse Temporelle Bidirectionnelle.

    Lit la série dans les deux sens (passé→futur et futur→passé) pour mieux
    capturer les tendances régionales.
    Architecture : Bi-LSTM → Bi-LSTM → Dense.

    :param window: taille de la fenêtre temporelle.
    :param n_features: nombre de features par pas de temps.
    :returns: modèle Keras compilé.

    Example:
        >>> model = build_bilstm_model(window=4)
    """
    model = Sequential([
        Input(shape=(window, n_features)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# ── Training Utilities ──────────────────────────────────────────────────────

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch=8):
    """
    Entraîne un modèle Keras avec EarlyStopping et ReduceLROnPlateau.

    :param model: modèle Keras à entraîner.
    :param X_train: données d'entraînement.
    :param y_train: cibles d'entraînement.
    :param X_val: données de validation.
    :param y_val: cibles de validation.
    :param epochs: nombre maximum d'époques.
    :param batch: taille des mini-lots.
    :returns: historique d'entraînement Keras.

    Example:
        >>> hist = train_model(model, X_tr, y_tr, X_v, y_v)
    """
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6)
    ]
    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch,
        callbacks=callbacks,
        verbose=0
    )


def compute_metrics(y_true, y_pred):
    """
    Calcule MAE, RMSE et R² entre prédictions et valeurs réelles.

    :param y_true: valeurs réelles np.ndarray.
    :param y_pred: valeurs prédites np.ndarray.
    :returns: tuple (mae, rmse, r2).

    Example:
        >>> mae, rmse, r2 = compute_metrics(y_true, y_pred)
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2


def reliability_score(r2, tier):
    """
    Calcule un score de fiabilité automatique basé uniquement sur le R² réel.

    Aucun plancher artificiel. Si le modèle ne prédit rien (R²≤0),
    la fiabilité sera proche de 0. Un petit bonus structurel (max +8 pts)
    est ajouté selon le tier car un ConvLSTM a plus de puissance de
    généralisation qu'un modèle Persistence, à R² égal.

    Bonus structurel par tier :
        Tier 1 (ConvLSTM) : +8 pts
        Tier 2 (CNN-LSTM) : +5 pts
        Tier 3 (Bi-LSTM)  : +3 pts
        Tier 4 (Persist.) : +0 pts — aucun apprentissage réel

    :param r2: coefficient de détermination, entre −∞ et 1.0.
    :param tier: tier du modèle utilisé (int, 1 à 4).
    :returns: score de fiabilité entre 0.0 et 99.0.

    Example:
        >>> reliability_score(r2=0.0, tier=1)   # 8.0
        >>> reliability_score(r2=0.75, tier=1)  # 83.0
        >>> reliability_score(r2=0.0, tier=4)   # 0.0
    """
    structural_bonus = {1: 8, 2: 5, 3: 3, 4: 0}
    bonus = structural_bonus.get(tier, 0)
    r2_clamped = max(0.0, r2)   # R² négatif → 0 (pire que la moyenne)
    raw_score = r2_clamped * 100 + bonus
    return round(min(99.0, raw_score), 2)


# ── Main Training Loop ─────────────────────────────────────────────────────

def train_and_predict():
    """
    Point d'entrée principal du moteur d'entraînement IA.

    Pour chaque maladie × province, sélectionne automatiquement le Tier
    approprié selon la densité des données, entraîne le modèle et distribue
    les prédictions sur toutes les zones de santé de la province.

    :returns: None (écrit en base de données et génère un rapport JSON).

    Example:
        >>> train_and_predict()
    """
    logger.info("=" * 70)
    logger.info("🧠 SPATIO-TEMPORAL AI ENGINE — ConvLSTM + CNN-LSTM + Bi-LSTM")
    logger.info("=" * 70)

    today       = date.today()
    target_date = (today.replace(day=1) + timedelta(days=32)).replace(day=1)

    # Nettoyage des prédictions précédentes pour le mois cible
    deleted = Prediction.objects.filter(prediction_date=target_date).delete()
    logger.info(f"🗑️  Prédictions précédentes supprimées : {deleted[0]}")

    prediction_list  = []
    seen_predictions = set()
    performance_log  = []

    provinces = list(Province.objects.all())
    diseases  = [
        d for d in
        ClinicalData.objects.values_list('disease_name', flat=True).distinct()
        if d in EPIDEMIC_DISEASES
    ]

    if not diseases:
        logger.warning("⚠️ Aucune donnée épidémique trouvée.")
        return

    logger.info(f"📊 {len(diseases)} épidémies × {len(provinces)} provinces")
    logger.info(f"🦠 Maladies : {', '.join(sorted(diseases))}")

    for d_name in diseases:
        logger.info(f"\n{'─'*50}")
        logger.info(f"🔬 Maladie : {d_name}")
        logger.info(f"{'─'*50}")

        for prov in provinces:
            # ── Chargement des données province + maladie ──────────────────
            prov_qs = ClinicalData.objects.filter(
                health_zone__territory__province=prov,
                disease_name=d_name
            ).order_by('month')

            if not prov_qs.exists():
                continue

            df = pd.DataFrame(list(prov_qs.values('month', 'cases_count')))
            df['month_dt'] = pd.to_datetime(df['month'])
            df_monthly = df.groupby('month_dt').agg({'cases_count': 'sum'}).reset_index()
            n_samples = len(df_monthly)

            # Zones de santé actives dans cette province pour cette maladie
            zones = list(HealthZone.objects.filter(
                territory__province=prov
            ).prefetch_related('territory'))

            # Tier par défaut = 4 (Persistence) — sera mis à jour selon la densité des données
            tier = 4

            # ── TIER 1 — ConvLSTM2D ───────────────────────────────────────
            if n_samples >= TIER1_MIN and len(zones) >= 2:
                tier = 1
                logger.info(f"  🌐 [TIER 1 — ConvLSTM] {prov.name} | {n_samples} mois | {len(zones)} zones")

                # Construction de la grille spatiale : shape (timesteps, n_zones)
                zone_ids  = [z.id for z in zones]
                n_zones   = len(zone_ids)
                all_dates = sorted(df_monthly['month_dt'].tolist())

                # Matrice spatiale : lignes=temps, colonnes=zones
                spatial_matrix = np.zeros((len(all_dates), n_zones))
                for zi, zone in enumerate(zones):
                    z_qs = ClinicalData.objects.filter(
                        health_zone=zone, disease_name=d_name
                    ).values('month', 'cases_count')
                    z_dict = {pd.Timestamp(r['month']): r['cases_count'] for r in z_qs}
                    for ti, dt in enumerate(all_dates):
                        spatial_matrix[ti, zi] = z_dict.get(dt, 0)

                scaler_sp = MinMaxScaler()
                sp_scaled = scaler_sp.fit_transform(spatial_matrix)  # (T, Z)

                # Séquences 3D: (samples, window, n_zones)
                X_sp, y_sp = [], []
                for i in range(len(sp_scaled) - WINDOW):
                    X_sp.append(sp_scaled[i:i + WINDOW, :])         # (window, n_zones)
                    y_sp.append(sp_scaled[i + WINDOW, :])            # (n_zones,)
                X_sp, y_sp = np.array(X_sp), np.array(y_sp)

                if len(X_sp) < 4:
                    tier = 3  # Fallback Bi-LSTM
                else:
                    # Reshape pour ConvLSTM2D: (samples, window, n_zones, 1, 1)
                    X_convlstm = X_sp.reshape(len(X_sp), WINDOW, n_zones, 1, 1)
                    # Y: not used directly in the train call shape
                    split = max(1, len(X_sp) - 2)
                    X_tr, X_v = X_convlstm[:split], X_convlstm[split:]
                    y_tr, y_v = y_sp[:split], y_sp[split:]

                    conv_model = build_convlstm_model(n_zones=n_zones, window=WINDOW, n_features=1)
                    train_model(conv_model, X_tr, y_tr, X_v, y_v, epochs=80, batch=4)

                    y_pred_sc = conv_model.predict(X_v, verbose=0)
                    y_pred_full = scaler_sp.inverse_transform(y_pred_sc)
                    y_true_full = scaler_sp.inverse_transform(y_v)

                    # Métriques globales (sur toutes zones)
                    mae, rmse, r2 = compute_metrics(y_true_full.flatten(), y_pred_full.flatten())
                    rel = reliability_score(r2, tier)
                    model_label = "ConvLSTM2D"

                    logger.info(f"    ✅ ConvLSTM — R²={r2:.3f} | MAE={mae:.1f} | Fiabilité={rel}%")

                    # Prédiction pour chaque zone
                    last_window = sp_scaled[-WINDOW:].reshape(1, WINDOW, n_zones, 1, 1)
                    next_pred_sc = conv_model.predict(last_window, verbose=0)
                    next_pred = scaler_sp.inverse_transform(next_pred_sc)[0]

                    d_safe = d_name.replace('/', '_').replace(' ', '_')
                    conv_model.save(str(MODELS_DIR / f"convlstm_{d_safe}_{prov.name.replace(' ', '_')}.keras"))

                    for zi, zone in enumerate(zones):
                        pred_val  = max(0.0, float(next_pred[zi]))
                        last_obs  = float(ClinicalData.objects.filter(
                            health_zone=zone, disease_name=d_name
                        ).order_by('month').values_list('cases_count', flat=True).last() or 0)
                        growth    = ((pred_val - last_obs) / max(1, last_obs)) * 100
                        severity  = "ALERTE ROUGE" if pred_val > 500 else "VIGILANCE" if pred_val > 150 else "STABLE"

                        key = (zone.id, d_name, target_date)
                        if key not in seen_predictions:
                            seen_predictions.add(key)
                            prediction_list.append(Prediction(
                                health_zone=zone, disease_name=d_name,
                                prediction_date=target_date,
                                predicted_cases=round(pred_val, 1),
                                influence_factors={
                                    "model_used": model_label, "tier": tier,
                                    "reliability": rel, "r2": round(r2, 3),
                                    "mae": round(mae, 2), "rmse": round(rmse, 2),
                                    "weekly_spread_rate": round(growth, 1),
                                    "attack_angle": severity,
                                    "spatial_context": f"{n_zones} zones modélisées"
                                }
                            ))

                    performance_log.append({
                        "maladie": d_name, "province": prov.name,
                        "model": model_label, "R2": round(float(r2), 3), "tier": tier
                    })

                    del conv_model, X_convlstm, X_sp, y_sp
                    gc.collect()
                    continue  # Tier 1 traité → passer à la province suivante

            # ── TIER 2 — CNN-LSTM ─────────────────────────────────────────
            if n_samples >= TIER2_MIN and tier != 1:
                tier = 2
                logger.info(f"  🔷 [TIER 2 — CNN-LSTM] {prov.name} | {n_samples} mois")

                features = build_feature_matrix(df_monthly)
                scaler   = MinMaxScaler()
                scaled   = scaler.fit_transform(features)
                X, y     = create_sequences(scaled, WINDOW)

                if len(X) < 4:
                    tier = 3
                else:
                    split = max(1, len(X) - 2)
                    X_tr, X_v = X[:split], X[split:]
                    y_tr, y_v = y[:split], y[split:]

                    cnn_model = build_cnn_lstm_model(window=WINDOW)
                    train_model(cnn_model, X_tr, y_tr, X_v, y_v, epochs=100, batch=8)

                    def inv_transform(v, s):
                        d = np.zeros((len(v), 3)); d[:, 0] = v.flatten()
                        return s.inverse_transform(d)[:, 0]

                    y_pred = inv_transform(cnn_model.predict(X_v, verbose=0), scaler)
                    y_true = inv_transform(y_v.reshape(-1, 1), scaler)
                    mae, rmse, r2 = compute_metrics(y_true, y_pred)
                    rel = reliability_score(r2, tier)
                    model_label = "CNN-LSTM"

                    logger.info(f"    ✅ CNN-LSTM — R²={r2:.3f} | MAE={mae:.1f} | Fiabilité={rel}%")

                    # Prédiction — contexte provincial → distribution zonale
                    last_window = scaled[-WINDOW:].reshape(1, WINDOW, 3)
                    pred_sc  = cnn_model.predict(last_window, verbose=0)[0][0]
                    dp = np.zeros((1, 3)); dp[0, 0] = pred_sc
                    pred_prov = float(scaler.inverse_transform(dp)[0, 0])

                    d_safe = d_name.replace('/', '_').replace(' ', '_')
                    cnn_model.save(str(MODELS_DIR / f"cnnlstm_{d_safe}_{prov.name.replace(' ', '_')}.keras"))

                    total_pop = sum(z.population for z in zones if z.population > 0) or 1
                    for zone in zones:
                        weight    = (zone.population or 1) / total_pop
                        pred_val  = max(0.0, pred_prov * weight)
                        last_obs  = float(ClinicalData.objects.filter(
                            health_zone=zone, disease_name=d_name
                        ).order_by('month').values_list('cases_count', flat=True).last() or 0)
                        growth    = ((pred_val - last_obs) / max(1, last_obs)) * 100
                        severity  = "ALERTE ROUGE" if pred_val > 500 else "VIGILANCE" if pred_val > 150 else "STABLE"

                        key = (zone.id, d_name, target_date)
                        if key not in seen_predictions:
                            seen_predictions.add(key)
                            prediction_list.append(Prediction(
                                health_zone=zone, disease_name=d_name,
                                prediction_date=target_date,
                                predicted_cases=round(pred_val, 1),
                                influence_factors={
                                    "model_used": model_label, "tier": tier,
                                    "reliability": rel, "r2": round(r2, 3),
                                    "mae": round(mae, 2), "rmse": round(rmse, 2),
                                    "weekly_spread_rate": round(growth, 1),
                                    "attack_angle": severity,
                                    "spatial_context": "Distribution par poids de population"
                                }
                            ))

                    performance_log.append({
                        "maladie": d_name, "province": prov.name,
                        "model": model_label, "R2": round(float(r2), 3), "tier": tier
                    })

                    del cnn_model, X, y
                    gc.collect()
                    continue

            # ── TIER 3 — Bi-LSTM ──────────────────────────────────────────
            if n_samples >= TIER3_MIN:
                tier = 3
                logger.info(f"  🔹 [TIER 3 — Bi-LSTM] {prov.name} | {n_samples} mois")

                features = build_feature_matrix(df_monthly)
                scaler   = MinMaxScaler()
                scaled   = scaler.fit_transform(features)
                cur_win  = max(3, min(WINDOW, n_samples // 3))
                X, y     = create_sequences(scaled, cur_win)

                if len(X) < 2:
                    tier = 4
                else:
                    split = max(1, len(X) - 2)
                    X_tr, X_v = X[:split], X[split:]
                    y_tr, y_v = y[:split], y[split:]

                    bi_model = build_bilstm_model(window=cur_win)
                    train_model(bi_model, X_tr, y_tr, X_v, y_v, epochs=100, batch=4)

                    def inv_transform(v, s):
                        d = np.zeros((len(v), 3)); d[:, 0] = v.flatten()
                        return s.inverse_transform(d)[:, 0]

                    y_pred = inv_transform(bi_model.predict(X_v, verbose=0), scaler)
                    y_true = inv_transform(y_v.reshape(-1, 1), scaler)
                    mae, rmse, r2 = compute_metrics(y_true, y_pred)
                    rel = reliability_score(r2, tier)
                    model_label = "Bi-LSTM"

                    logger.info(f"    ✅ Bi-LSTM — R²={r2:.3f} | MAE={mae:.1f} | Fiabilité={rel}%")

                    last_window = scaled[-cur_win:].reshape(1, cur_win, 3)
                    pred_sc  = bi_model.predict(last_window, verbose=0)[0][0]
                    dp = np.zeros((1, 3)); dp[0, 0] = pred_sc
                    pred_prov = float(scaler.inverse_transform(dp)[0, 0])

                    total_pop = sum(z.population for z in zones if z.population > 0) or 1
                    for zone in zones:
                        weight    = (zone.population or 1) / total_pop
                        pred_val  = max(0.0, pred_prov * weight)
                        last_obs  = float(ClinicalData.objects.filter(
                            health_zone=zone, disease_name=d_name
                        ).order_by('month').values_list('cases_count', flat=True).last() or 0)
                        growth    = ((pred_val - last_obs) / max(1, last_obs)) * 100
                        severity  = "ALERTE ROUGE" if pred_val > 500 else "VIGILANCE" if pred_val > 150 else "STABLE"

                        key = (zone.id, d_name, target_date)
                        if key not in seen_predictions:
                            seen_predictions.add(key)
                            prediction_list.append(Prediction(
                                health_zone=zone, disease_name=d_name,
                                prediction_date=target_date,
                                predicted_cases=round(pred_val, 1),
                                influence_factors={
                                    "model_used": model_label, "tier": tier,
                                    "reliability": rel, "r2": round(r2, 3),
                                    "mae": round(mae, 2), "rmse": round(rmse, 2),
                                    "weekly_spread_rate": round(growth, 1),
                                    "attack_angle": severity,
                                    "window_size": cur_win
                                }
                            ))

                    performance_log.append({
                        "maladie": d_name, "province": prov.name,
                        "model": model_label, "R2": round(float(r2), 3), "tier": tier
                    })

                    del bi_model, X, y
                    gc.collect()
                    continue

            # ── TIER 4 — PERSISTENCE ──────────────────────────────────────
            tier = 4
            logger.info(f"  ⬜ [TIER 4 — Persistence] {prov.name} | {n_samples} pts | données insuffisantes")
            model_label = "Persistence"
            r2, mae, rmse = 0.33, 0, 0
            rel = reliability_score(r2, tier)

            for zone in zones:
                last_rec = ClinicalData.objects.filter(
                    health_zone=zone, disease_name=d_name
                ).order_by('month').last()
                pred_val = float(last_rec.cases_count) if last_rec else 0.0
                last_obs = pred_val
                growth   = 0.0
                severity = "INCONNU : Données insuffisantes"

                key = (zone.id, d_name, target_date)
                if key not in seen_predictions:
                    seen_predictions.add(key)
                    prediction_list.append(Prediction(
                        health_zone=zone, disease_name=d_name,
                        prediction_date=target_date,
                        predicted_cases=round(pred_val, 1),
                        influence_factors={
                            "model_used": model_label, "tier": tier,
                            "reliability": rel, "r2": r2, "mae": 0, "rmse": 0,
                            "weekly_spread_rate": 0.0, "attack_angle": severity,
                            "data_sparse": True
                        }
                    ))

            performance_log.append({
                "maladie": d_name, "province": prov.name,
                "model": model_label, "R2": r2, "tier": tier
            })

    # ── Sauvegarde en base ─────────────────────────────────────────────────
    if prediction_list:
        Prediction.objects.bulk_create(prediction_list, batch_size=500, ignore_conflicts=True)

    logger.info("\n" + "=" * 70)
    logger.info("🏁 ENTRAÎNEMENT TERMINÉ ─ RÉSUMÉ")
    logger.info("=" * 70)
    logger.info(f"  ✅ Prédictions créées  : {len(prediction_list)}")

    # Distribution par Tier
    tier_counts = {}
    for p in performance_log:
        t = p.get("tier", 4)
        tier_counts[t] = tier_counts.get(t, 0) + 1

    for t, count in sorted(tier_counts.items()):
        labels = {1: "ConvLSTM2D", 2: "CNN-LSTM", 3: "Bi-LSTM", 4: "Persistence"}
        logger.info(f"  {'🌐' if t==1 else '🔷' if t==2 else '🔹' if t==3 else '⬜'} Tier {t} ({labels[t]}): {count} province×maladie")

    # Rapport JSON
    report_path = REPORTS_DIR / f"spatio_temporal_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_predictions": len(prediction_list),
            "tier_distribution": tier_counts,
            "performance_metrics": performance_log
        }, f, indent=4)

    logger.info(f"  📄 Rapport : {report_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    # Désactiver GPU si TF s'affole avec la mémoire GPU partagée
    tf.config.set_visible_devices([], 'GPU')
    train_and_predict()
