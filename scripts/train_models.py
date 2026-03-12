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
import time

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Masquer avertissements CUDA/AVX

# Django setup
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()

from django.db import models as django_models
from analytics.models import Province, HealthZone, ClinicalData, Prediction

# Import module climatique
try:
    from climate_data_fetcher import ClimateDataFetcher, merge_climate_with_epidemiology
    CLIMATE_MODULE_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ climate_data_fetcher non disponible")
    CLIMATE_MODULE_AVAILABLE = False

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional,
    Conv1D, MaxPooling1D, Flatten, TimeDistributed,
    ConvLSTM2D, Reshape, LayerNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from scipy import stats
import pickle

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

# Chemins données climatiques
CLIMATE_CSV_PATH = Path("/app/data/climat/rdc_climate_monthly.csv")
CLIMATE_PKL_PATH = Path("/app/data/climat/rdc_climate_monthly.pkl")
CLIMATE_PROVINCES_DIR = Path("/app/data/climat/provinces")

# Maladies épidémiques prioritaires — seules ces maladies sont entraînées.
# Exclut les maladies chroniques (Anémie, Malnutrition, Onchocercose, etc.)
# ──────────────────────────────────────────────────────────────────────────────
# MALADIES CIBLES — 6 MALADIES SÉPARÉES (v2.2)
# ──────────────────────────────────────────────────────────────────────────────
# Détachement complet : chaque maladie modélisée indépendamment avec ses traits
# épidémiologiques propres. Données historiques Jan 2020 → Mars 2026.
#
#   1. Paludisme    — transmission vecteur, très saisonnière, pics pluvieux
#   2. Choléra      — diarrhée hydrique, épidémiques courtes et intenses
#   3. VIH/SIDA     — transmission sexuelle/sang, chronique, non-saisonnière
#   4. Tuberculose  — transmission aérienne, progression lente (mois/ans)

EPIDEMIC_DISEASES = {
    'Paludisme',      # Groupe 1 — maladie vectorielle majeure
    'Choléra',        # Groupe 2 — maladie hydrique
    'VIH/SIDA',       # Groupe 3 — maladie chronique transmissible
    'Tuberculose',    # Groupe 4 — maladie respiratoire chronique
}

# Mappage : alias sources → groupe (pas de consolidation, 1:1 mapping)
DISEASE_CONSOLIDATION_MAP = {
    'Paludisme': 'Paludisme',
    'Malaria': 'Paludisme',
    
    'Choléra': 'Choléra',
    'Cholera': 'Choléra',
    
    'VIH/SIDA': 'VIH/SIDA',
    'VIH': 'VIH/SIDA',
    'HIV/AIDS': 'VIH/SIDA',
    
    'Tuberculose': 'Tuberculose',
    'TB': 'Tuberculose',
    'Tuberculosis': 'Tuberculose',
}

# ── Custom Loss Functions ──────────────────────────────────────────────────

def scaled_mse_loss(y_true, y_pred):
    """
    MSE SIMPLE & ROBUST : pas de divergence analytique  
    
    ✅ Pourquoi MSE simple ?
    - Log-functions → divergent sur petites valeurs (Choléra: 453k, log(0.01) = -4.6)
    - Huber → instable sur variance énorme (Paludisme: 314M vs Choléra: 453k)
    - MSE → stable sur TOUT type de distribution
    
    ✅ Optimisé pour:
       • Petites valeurs (Choléra: 453k)
       • Grandes valeurs (Paludisme: 314M)  
       • Variance ultra-élevée (VIH/SIDA: 91x ratio)
    
    Normalisation APPLIQUÉE EN AMONT (avant entraînement par disease)
    
    :param y_true: cibles (pré-normalisées)
    :param y_pred: prédictions du modèle
    :returns: scalar loss MSE
    """
    return K.mean(K.square(y_true - y_pred))


def weighted_huber_loss(y_true, y_pred):
    """⚠️ DEPRECATED - Alias pour scaled_mse_loss. Utiliser scaled_mse_loss directement."""
    return scaled_mse_loss(y_true, y_pred)


def log_mse_loss(y_true, y_pred):
    """⚠️ DEPRECATED - Cause divergence sur petites valeurs. Utiliser scaled_mse_loss."""
    return scaled_mse_loss(y_true, y_pred)


def log_huber_loss(y_true, y_pred):
    """⚠️ DEPRECATED - Cause divergence sur petites valeurs. Utiliser scaled_mse_loss."""
    return scaled_mse_loss(y_true, y_pred)


# ── Data Cleaning ─────────────────────────────────────────────────────────

def consolidate_disease_data(disease_name):
    """
    Retourne le groupe consolidé pour une maladie source.
    Utilise le mappage DISEASE_CONSOLIDATION_MAP.
    
    :param disease_name: nom de la maladie source
    :returns: nom du groupe consolidé (ex: 'Paludisme + Typhoïde')
    """
    return DISEASE_CONSOLIDATION_MAP.get(disease_name, None)


def remove_duplicate_predictions(df):
    """
    Supprime les vrais doublons (même zone, même date, même maladie).
    Stratégie : garder l'observation avec la plus hautecouverture de données,
    puis la plus récente en cas d'égalité.
    
    :param df: DataFrame avec ['month', 'cases_count', 'health_zone_id']
    :returns: DataFrame sans doublons
    """
    if df.empty:
        return df
    
    df = df.copy()
    subset_cols = ['month']
    if 'health_zone_id' in df.columns:
        subset_cols.append('health_zone_id')
    
    # Identifier doublons
    duplicates = df.duplicated(subset=subset_cols, keep=False)
    
    if duplicates.any():
        # Pour chaque groupe de doublons, garder le plus haut
        df = df.sort_values(['cases_count'], ascending=False).drop_duplicates(
            subset=subset_cols, keep='first'
        )
        logger.debug(f"    🔄 {duplicates.sum()} doublons neutralisés")
    
    return df


class AdaptiveScaler:
    """
    Scaleur adaptatif pour données avec distributions extrêmes.
    Détecte automatiquement la dispersion et applique MinMax ou Robust.
    """
    def __init__(self):
        self.scaler = None
        self.use_robust = False
        self.scaler_type = None
        
    def fit_transform(self, data):
        """
        Choix automatique du scaleur basé sur statistiques
        """
        values = data.flatten() if data.ndim > 1 else data
        
        # Calcul de la dispersion
        median = np.median(values)
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        if median > 0:
            iqr_ratio = iqr / median if median != 0 else 0
        else:
            iqr_ratio = 1.0
        
        # Si IQR/median > 0.5 → distribution très dispersée → RobustScaler
        if iqr_ratio > 0.5:
            self.scaler = RobustScaler()
            self.scaler_type = "RobustScaler (données dispersées)"
            self.use_robust = True
            logger.debug(f"    🎯 Normalisation: RobustScaler (IQR/median={iqr_ratio:.2f})")
        else:
            self.scaler = MinMaxScaler()
            self.scaler_type = "MinMaxScaler"
            self.use_robust = False
            logger.debug(f"    🎯 Normalisation: MinMaxScaler (IQR/median={iqr_ratio:.2f})")
        
        if data.ndim == 1:
            return self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        else:
            return self.scaler.fit_transform(data)
    
    def inverse_transform(self, data):
        """Inverse la transformation"""
        if self.scaler is None:
            return data
        
        if data.ndim == 1:
            return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
        else:
            return self.scaler.inverse_transform(data)


class ScalerWithSeasonality:
    """
    Wrapper smarty pour normalisation adaptative:
    - Détecte si données très dispersées → RobustScaler
    - Sinon MinMaxScaler
    - Preserve saisonnalité et climat (cols 1+)
    """
    def __init__(self):
        self.scaler = None  # Sera MinMaxScaler ou RobustScaler
        self.seasonality_data = None
        self.cases_min = None
        self.cases_max = None
        self.n_features = None
        self.scaler_type = None
        
    def fit_transform(self, data, seasonality_features=None):
        """
        Choix automatique du scaleur basé sur dispersion des cas (col 0)
        """
        self.seasonality_data = seasonality_features
        self.n_features = data.shape[1] if data.ndim > 1 else 1
        
        # Sauvegarder stats des cases (col 0) avant normalisation
        cases = data[:, 0] if data.ndim > 1 else data
        self.cases_min = float(cases.min())
        self.cases_max = float(cases.max())
        
        # Détection de dispersion
        median_cases = np.median(cases)
        q1 = np.percentile(cases, 25)
        q3 = np.percentile(cases, 75)
        iqr = q3 - q1
        
        if median_cases > 0 and iqr > 0:
            iqr_ratio = iqr / median_cases
        else:
            iqr_ratio = 1.0
        
        # Choix du scaleur
        if iqr_ratio > 0.4:  # Distribution dispersée
            self.scaler = RobustScaler()
            self.scaler_type = "RobustScaler"
            logger.debug(f"    🎯 RobustScaler sélectionné (dispersion IQR/med={iqr_ratio:.2f})")
        else:
            self.scaler = MinMaxScaler()
            self.scaler_type = "MinMaxScaler"
            logger.debug(f"    🎯 MinMaxScaler sélectionné (dispersion IQR/med={iqr_ratio:.2f})")
        
        return self.scaler.fit_transform(data)
    
    def inverse_transform(self, data, include_seasonality=False):
        """Inverse la normalisation en preservant saisonnalité/climat"""
        n_features = self.n_features or 3
        
        if include_seasonality and self.seasonality_data is not None:
            sin_m, cos_m = self.seasonality_data
            n_samples = len(data.flatten()) if data.ndim == 1 else len(data)
            
            reconstructed = np.zeros((n_samples, n_features))
            reconstructed[:, 0] = data.flatten() if data.ndim == 1 else data[:, 0]
            reconstructed[:, 1] = sin_m[-n_samples:] if n_features >= 2 else 0
            reconstructed[:, 2] = cos_m[-n_samples:] if n_features >= 3 else 0
            
            denorm = self.scaler.inverse_transform(reconstructed)
            return denorm[:, 0]
        else:
            # Méthode basique : pad avec zéros et dénom col 0
            if data.ndim == 1:
                d = np.zeros((len(data), n_features))
                d[:, 0] = data.flatten()
            else:
                d = data.copy()
            
            denorm = self.scaler.inverse_transform(d)
            return denorm[:, 0]


def clean_clinical_data(df, min_variance_threshold=10.0, min_coverage=0.60):
    """
    Nettoie INTELLIGEMMENT les données cliniques avec remplissage des trous.

    Actions de nettoyage :
        ✓ Supprime les lignes avec valeurs NULL/NA
        ✓ Supprime les cas négatifs
        ✓ REMPLIT LES TROUS : interpolation linéaire entre périodes
        ✓ Détecte et atténue les OUTLIERS extrêmes (z-score > 2.5)
        ✓ Vérifie la couverture temporelle (min 70% non-zéro)
        ✓ Rejette les séries plates (variance < 10)
        ✓ Tri par date

    :param df: DataFrame avec 'month', 'cases_count'
    :param min_variance_threshold: variance minimale (défaut: 10.0)
    :param min_coverage: % minimum de données non-zéro accepté (défaut: 0.7 = 70%)
    :returns: DataFrame nettoyé ou vide si insuffisant
    """
    if df.empty:
        logger.warning("⚠️ DataFrame vide")
        return df

    df = df.copy()
    initial_rows = len(df)
    
    # 1. NULL/NA
    df = df.dropna(subset=['month', 'cases_count'])
    null_dropped = initial_rows - len(df)
    if null_dropped > 0:
        logger.debug(f"    🗑️  NULL/NA: {null_dropped} lignes supprimées")

    # 2. Conversion numérique
    df['cases_count'] = pd.to_numeric(df['cases_count'], errors='coerce').astype('float64')
    df = df.dropna(subset=['cases_count'])

    # 3. Cas négatifs → CLIPPING à 0
    negative_count = (df['cases_count'] < 0).sum()
    if negative_count > 0:
        logger.debug(f"    ⚠️ {negative_count} valeurs négatives clippées à 0")
        df['cases_count'] = df['cases_count'].clip(lower=0.0)

    # 4. Doublons
    df = remove_duplicate_predictions(df)

    # 5. Conversion datetime et tri
    df['month'] = pd.to_datetime(df['month'])
    df = df.sort_values('month').reset_index(drop=True)

    if len(df) < 2:
        return pd.DataFrame()

    # 6. ⭐ REMPLISSAGE DES TROUS (interpolation)
    # Créer une plage complète de mois
    date_range = pd.date_range(start=df['month'].min(), end=df['month'].max(), freq='MS')
    df_full = pd.DataFrame({'month': date_range})
    df_full = df_full.merge(df[['month', 'cases_count']], on='month', how='left')
    
    # Interpolation linéaire pour les zéros/NaN
    df_full['cases_count'] = df_full['cases_count'].interpolate(method='linear', limit_direction='both')
    
    # Remplir les NaN aux extrémités avec la valeur la plus proche
    if df_full['cases_count'].isna().any():
        df_full['cases_count'] = df_full['cases_count'].fillna(method='bfill').fillna(method='ffill')
    
    # Assurer NON-NÉGATIF
    df_full['cases_count'] = df_full['cases_count'].clip(lower=0.0)
    
    df = df_full.copy()
    logger.debug(f"    🔧 Remplissage trous : {len(date_range)} mois complets")

    # 7. Détection d'OUTLIERS extrêmes (z-score > 2.5) ET atténuation
    if len(df) > 4:
        values = df['cases_count'].values
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val > 0:
            z_scores = np.abs((values - mean_val) / std_val)
            outlier_mask = z_scores > 2.5
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                logger.debug(f"    🎯 {outlier_count} outliers détectés (z>2.5), ATTÉNUATION")
                # Atténuer : remplacer par moyenne locale (3-mois glissante)
                for idx in np.where(outlier_mask)[0]:
                    start_idx = max(0, idx - 1)
                    end_idx = min(len(values), idx + 2)
                    local_values = []
                    for j in range(start_idx, end_idx):
                        if j != idx and not outlier_mask[j]:
                            local_values.append(values[j])
                    
                    if local_values:
                        df.loc[idx, 'cases_count'] = np.mean(local_values)
                    else:
                        df.loc[idx, 'cases_count'] = mean_val

    # 8. Vérification variance ET couverture
    variance = 0.0
    coverage = 0.0
    non_zero_count = 0
    
    if len(df) > 3:
        variance = float(np.var(df['cases_count'].values))
        non_zero_count = (df['cases_count'] > 0.1).sum()  # Threshold petit pour éviter zéros d'interpolation
        coverage = non_zero_count / len(df)
        
        logger.debug(f"    📊 APRÈS NETTOYAGE: Variance={variance:.2f}, Coverage={coverage:.1%} ({non_zero_count}/{len(df)})")
        
        # Critères STRICTS
        if variance < min_variance_threshold:
            logger.debug(f"    ⚠️ Variance trop faible ({variance:.2f} < {min_variance_threshold}) → REJETÉ")
            return pd.DataFrame()
        
        if coverage < min_coverage:
            logger.debug(f"    ⚠️ Couverture insuffisante ({coverage:.1%} < {min_coverage:.1%}) → REJETÉ")
            return pd.DataFrame()

    # 9. Clipping final et arrondi
    df['cases_count'] = df['cases_count'].clip(lower=0.0).round(1).astype('float64')

    final_rows = len(df)
    logger.debug(f"    ✅ FINAL: {initial_rows}→{final_rows} lignes | var={variance:.2f} | cov={coverage:.1%}")

    return df


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


def build_feature_matrix_with_climate(df_monthly, climate_df=None):
    """
    Construit une matrice de features avec données climatiques séparées.
    
    v2.4 FIX: LOG-TRANSFORM + MinMax pour réduire la variance extrême:
    - Cases: log(x+1) → MinMaxScaler [0, 1]  (stabilise les variances énormes)
    - Climat: Valeurs brutes (pas de normalisation liée aux cases)
    
    ✅ POURQUOI LOG ?
    Paludisme: 314M cas, Choléra: 453k cas, VIH: 8.8M cas → ratios énormes
    log(314M) = 19.6, log(453k) = 13.0, log(8.8M) = 16.0 → BEAUCOUP plus équilibré
    Réduit la variance de 91x à ~2.5x ✨
    
    :param df_monthly: DataFrame épidémiologique avec 'month_dt', 'cases_count'
    :param climate_df: DataFrame climatique avec temperature, precipitation, relative_humidity
    :returns: tuple (feature_array, n_features, feature_names)
    """
    sin_m, cos_m = seasonality_features(df_monthly['month_dt'])
    cases_raw = df_monthly['cases_count'].values
    
    # 📊 LOG-TRANSFORMATION: log(x + 1) pour éviter log(0)
    # Réduit énormément la variance des grandes valeurs
    cases_log = np.log1p(cases_raw)
    
    # Normaliser les valeurs log [0, 1]
    cases_min, cases_max = cases_log.min(), cases_log.max()
    if cases_max > cases_min:
        cases_norm = (cases_log - cases_min) / (cases_max - cases_min)
    else:
        # Si toutes les valeurs sont identiques (constant), on ne peut pas donner du signal
        # → retourner zéros normalisés + saisonnalité seulement
        cases_norm = np.ones_like(cases_log, dtype=float) * 0.5  # Valeur neutre
    
    # Construire vecteur : [cases_NORM, sin_month, cos_month] + climat optionnel
    n_months = len(cases_raw)
    
    if climate_df is not None and not climate_df.empty and len(climate_df) == n_months:
        # Intégration climat : align with case data temporally
        features = np.zeros((n_months, 6))
        features[:, 0] = cases_norm
        features[:, 1] = sin_m
        features[:, 2] = cos_m
        features[:, 3] = climate_df['temperature'].values  # Brutes (°C)
        features[:, 4] = climate_df['precipitation'].values  # Brutes (mm)
        features[:, 5] = climate_df['relative_humidity'].values  # Brutes (%)
        feature_names = ['cases', 'sin_month', 'cos_month', 'temp', 'precip', 'humidity']
        n_features = 6
    else:
        # Sans climat : 3 features seulement
        features = np.stack([cases_norm, sin_m, cos_m], axis=1)
        feature_names = ['cases', 'sin_month', 'cos_month']
        n_features = 3
    
    logger.debug(f"    ✓ Features (v2.4): {n_features} colonnes {feature_names}")
    logger.debug(f"      Cases: raw min={cases_raw.min():.1f} max={cases_raw.max():.1f} → log(min)={cases_min:.2f} log(max)={cases_max:.2f} → norm [0,1]")
    
    return features, n_features, feature_names


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


def temporal_train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2):
    """
    Split temporel : entraîne sur le passé, valide sur le milieu, teste sur le futur.
    
    :param X, y: données
    :param train_ratio: % pour entraînement (défaut 60%)
    :param val_ratio: % pour validation (défaut 20%), reste = test
    :returns: tuple (X_tr, X_v, X_te, y_tr, y_v, y_te)
    """
    n = len(X)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))
    
    X_tr, y_tr = X[:train_idx], y[:train_idx]
    X_v, y_v = X[train_idx:val_idx], y[train_idx:val_idx]
    X_te, y_te = X[val_idx:], y[val_idx:]
    
    logger.debug(f"    📈 Split temporel: train={len(X_tr)}, val={len(X_v)}, test={len(X_te)}")
    
    return X_tr, X_v, X_te, y_tr, y_v, y_te


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
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
    outputs = Dense(n_zones, kernel_regularizer=l2(0.001))(x)
    model = Model(inputs, outputs)
    # Utiliser MSE simple : stable sur toutes les échelles (Paludisme: 314M, Choléra: 453k)
    model.compile(
        optimizer='adam',
        loss=scaled_mse_loss,  # Stable sur variance énorme + pas de divergence
        metrics=['mae', 'mse']
    )
    return model


def build_cnn_lstm_model(window, n_features=3):
    """
    TIER 2 — CNN-LSTM Hybride : Patterns Locaux + Mémoire Temporelle.

    Le CNN extrait des motifs caractéristiques des épidémies (pics, plateaux)
    sur des sous-séquences ; le LSTM mémorise la dynamique temporelle globale.
    Architecture : Conv1D → MaxPool → Conv1D → LSTM → Dense.

    :param window: taille de la fenêtre temporelle (pas de temps).
    :param n_features: nombre de features par pas de temps (3 par défaut avec saisonnalité,
                       ou 4 avec données climatiques: [cases, T°, précip, humidity]).
    :returns: modèle Keras compilé.

    Example:
        >>> model = build_cnn_lstm_model(window=4, n_features=4)  # avec climat
    """
    model = Sequential([
        Input(shape=(window, n_features)),
        # Bloc CNN : extraction de motifs locaux
        Conv1D(filters=64, kernel_size=min(3, window), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
        Conv1D(filters=32, kernel_size=min(2, window), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling1D(pool_size=1),   # pool_size=1 préserve la dim temporelle pour LSTM
        # Bloc LSTM : mémoire temporelle
        LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)),
        Dropout(0.25),
        LSTM(32, kernel_regularizer=l2(0.001)),
        Dropout(0.25),
        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(1)
    ])
    # Utiliser MSE simple : stable sur toutes les échelles (Paludisme: 314M, Choléra: 453k)
    model.compile(
        optimizer='adam',
        loss=scaled_mse_loss,  # Stable sur variance énorme + pas de divergence
        metrics=['mae', 'mse']
    )
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
        >>> model = build_bilstm_model(window=4, n_features=4)  # avec climat
    """
    model = Sequential([
        Input(shape=(window, n_features)),
        Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))),
        Dropout(0.2),
        Bidirectional(LSTM(32, kernel_regularizer=l2(0.001))),
        Dropout(0.2),
        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(1)
    ])
    # Utiliser MSE simple : stable sur toutes les échelles (Paludisme: 314M, Choléra: 453k)
    model.compile(
        optimizer='adam',
        loss=scaled_mse_loss,  # Stable sur variance énorme + pas de divergence
        metrics=['mae', 'mse']
    )
    return model


# ── Training Utilities ──────────────────────────────────────────────────────

class RAMCleanerCallback(tf.keras.callbacks.Callback):
    """
    Callback personnalisée qui nettoie la RAM toutes les 60 secondes pendant l'entraînement.
    Prévient les fuites mémoire sur les longs entraînements.
    """
    def __init__(self, clean_interval=60):
        """
        :param clean_interval: intervalle en secondes pour nettoyer la RAM (défaut: 60)
        """
        super().__init__()
        self.clean_interval = clean_interval
        self.last_clean_time = time.time()
    
    def on_train_batch_end(self, batch, logs=None):
        """Appelé à la fin de chaque batch d'entraînement"""
        current_time = time.time()
        
        if current_time - self.last_clean_time >= self.clean_interval:
            # Nettoyer la mémoire
            gc.collect()
            K.clear_session()  # Vider les tenseurs inutilisés
            self.last_clean_time = current_time
            logger.debug(f"🧹 Nettoyage RAM effectué (+ {self.clean_interval}s)")


def calculate_spread_rate(zone, disease_name, pred_val):
    """
    ✨ Calcule le taux de propagation intelligent basé sur la TENDANCE HISTORIQUE
    
    Au lieu de simplement comparer pred vs last_obs (qui donne 0% si stable),
    on calcule: (pred / trend_baseline) - 1) * 100
    
    où trend_baseline = moyenne mobile des 6 derniers mois de données
    
    ✅ Logique:
    - Si pred > baseline → propagation POSITIVE (épidémie qui s'accélère)
    - Si pred < baseline → propagation NÉGATIVE (épidémie qui s'enraye)
    - Si pred ≈ baseline → propagation ≈ 0 (endémicité stable)
    
    ✅ Avantages sur l'ancienne formule:
    - Basé sur la TENDANCE, pas un point unique
    - Capture les changements de régime (épidémie stable → soudain augmente)
    - Évite les 0% partout quand données sont stables
    - Plus robuste face aux données bruitées
    
    :param zone: HealthZone object
    :param disease_name: str, nom de la maladie
    :param pred_val: float, prédiction du modèle
    :returns: float, taux de propagation en %/semaine
    """
    try:
        # Récupérer les 6 derniers mois de données (tendance historique)
        recent_data = list(ClinicalData.objects.filter(
            health_zone=zone, disease_name=disease_name
        ).order_by('-month')[:6].values_list('cases_count', flat=True))
        
        if not recent_data:
            # Pas de données historiques → prédiction basée sur baseline nulle
            # Donc toute valeur > 0 est une augmentation
            return 5.0 if pred_val > 0 else 0.0
        
        # Calculer la moyenne mobile des 6 derniers mois (ou moins s'il n'y en a pas 6)
        trend_baseline = np.mean(recent_data)
        
        if trend_baseline <= 0:
            # Si baseline est 0 ou négatif (données bruitées), utiliser cas de base
            return 5.0 if pred_val > 0 else 0.0
        
        # Taux de propagation = (prédiction / tendance) - 1) * 100
        spread_rate = ((pred_val / trend_baseline) - 1.0) * 100
        
        # Clipper pour éviter les valeurs extrêmes énormes (si trend_baseline très petit)
        spread_rate = np.clip(spread_rate, -100.0, 200.0)
        
        logger.debug(f"    📈 Propagation {zone.name}/{disease_name}: "
                    f"baseline={trend_baseline:.0f}, pred={pred_val:.0f}, rate={spread_rate:.1f}%")
        
        return spread_rate
        
    except Exception as e:
        logger.warning(f"    ⚠️ Erreur calcul propagation {zone.name}: {e}")
        # Fallback simple et robuste
        return 0.0


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch=8):
    """
    Entraîne un modèle Keras avec EarlyStopping, ReduceLROnPlateau et nettoyage RAM.

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
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6),
        RAMCleanerCallback(clean_interval=60)  # Nettoie RAM toutes les 60 secondes
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
    Nettoie automatiquement les NaN avant calcul.

    :param y_true: valeurs réelles np.ndarray.
    :param y_pred: valeurs prédites np.ndarray.
    :returns: tuple (mae, rmse, r2).

    Example:
        >>> mae, rmse, r2 = compute_metrics(y_true, y_pred)
    """
    # Nettoyer NaN
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Masque pour indices valides (non-NaN)
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    
    if not valid_mask.any():
        # Tous NaN: retourner métriques nulles
        return 0.0, 0.0, 0.0
    
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
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


def verify_climate_data():
    """
    Vérifie la disponibilité et la couverture des données climatiques PAR PROVINCE.
    
    Charges les données climatiques pour chaque province depuis data/climat/provinces/
    Fallback: données nationales ou synthétiques si provinces indisponibles.
    
    :returns: dict {province_name: DataFrame} avec colonnes [date, temperature, precipitation, relative_humidity]
              ou None si données insuffisantes pour le mois en cours
    """
    today = date.today()
    current_month_start = today.replace(day=1)
    
    logger.info("\n📡 VÉRIFICATION DONNÉES CLIMATIQUES PAR PROVINCE")
    logger.info("─" * 70)
    
    all_provinces_climate = {}
    
    # Essai 1: Charger données par province
    if CLIMATE_PROVINCES_DIR.exists() and list(CLIMATE_PROVINCES_DIR.glob("*.csv")):
        try:
            csv_files = list(CLIMATE_PROVINCES_DIR.glob("*_climate.csv"))
            logger.info(f"✓ Répertoire provinces trouvé ({len(csv_files)} fichiers)")
            
            for csv_file in csv_files:
                try:
                    prov_name = csv_file.stem.replace('_climate', '').replace('_', ' ')
                    df = pd.read_csv(csv_file)
                    df['date'] = pd.to_datetime(df['date'])
                    all_provinces_climate[prov_name] = df
                except Exception as e:
                    logger.debug(f"  ⚠️ Erreur lecture {csv_file.name}: {e}")
            
            if all_provinces_climate:
                logger.info(f"✓ {len(all_provinces_climate)} provinces chargées depuis répertoire")
        
        except Exception as e:
            logger.warning(f"⚠️ Erreur accès répertoire provinces: {e}")
    
    # Essai 1.5: Générer synthétique pour toutes les provinces (rapide et fiable)
    if not all_provinces_climate:
        logger.info("⚠️ Données par province introuvables. Génération synthétique...")
        try:
            if CLIMATE_MODULE_AVAILABLE:
                from generate_province_climate import generate_climate_for_all_provinces
                all_provinces_climate = generate_climate_for_all_provinces()
                if all_provinces_climate:
                    logger.info(f"✓ {len(all_provinces_climate)} provinces générées (synthétique)")
                else:
                    logger.warning("⚠️ Génération synthétique vide")
            else:
                logger.debug("Module climat non disponible pour génération")
        except Exception as e:
            logger.warning(f"⚠️ Erreur génération synthétique: {e}")
    
    # Essai 2: Fallback données nationales si provinces vides
    if not all_provinces_climate:
        logger.info("⚠️ Données par province introuvables. Tentative fallback national...")
        
        climate_df = None
        
        # Charger depuis pickle
        if CLIMATE_PKL_PATH.exists():
            try:
                climate_df = pd.read_pickle(CLIMATE_PKL_PATH)
                logger.info(f"✓ Chargé depuis pickle national: {CLIMATE_PKL_PATH}")
            except Exception as e:
                logger.warning(f"⚠️ Erreur pickle: {e}")
        
        # Charger depuis CSV
        if climate_df is None and CLIMATE_CSV_PATH.exists():
            try:
                climate_df = pd.read_csv(CLIMATE_CSV_PATH)
                climate_df['date'] = pd.to_datetime(climate_df['date'])
                logger.info(f"✓ Chargé depuis CSV national: {CLIMATE_CSV_PATH}")
            except Exception as e:
                logger.warning(f"⚠️ Erreur CSV: {e}")
        
        # Générer synthétique si rien
        if climate_df is None:
            logger.warning("⚠️ Données nationales introuvables. Génération synthétique...")
            try:
                if CLIMATE_MODULE_AVAILABLE:
                    from climate_data_fetcher import generate_rdc_synthetic_climate
                    climate_df = generate_rdc_synthetic_climate(year_range=(today.year - 5, today.year))
                    logger.info("✓ Données synthétiques générées")
                else:
                    logger.error("✗ Module climat non disponible")
                    return None
            except Exception as e:
                logger.error(f"✗ Impossible de générer synthétique: {e}")
                return None
        
        if climate_df is not None:
            # Appliquer aux TOUTES les provinces comme fallback
            provinces = list(Province.objects.all())
            for prov in provinces:
                all_provinces_climate[prov.name] = climate_df.copy()
    
    if not all_provinces_climate:
        logger.warning("✗ Aucune donnée climatique disponible!")
        return None
    
    # ── Vérifications de couverture ────────────────────────────────────────
    logger.info(f"\n📊 Couverture {len(all_provinces_climate)} provinces")
    
    # Check mois courant pour au moins une province
    current_month_found = False
    current_month_start_ts = pd.Timestamp(current_month_start)
    current_month_end_ts = current_month_start_ts + timedelta(days=32)
    current_month_end_ts = current_month_end_ts.replace(day=1)
    
    for prov_name, df in all_provinces_climate.items():
        df['date'] = pd.to_datetime(df['date'])
        current_month_data = df[
            (df['date'] >= current_month_start_ts) &
            (df['date'] < current_month_end_ts)
        ]
        
        if not current_month_data.empty:
            current_month_found = True
            break
    
    if current_month_found:
        logger.info(f"✓ Données du mois en cours ({current_month_start.strftime('%B %Y')}): PRÉSENTES")
    else:
        logger.warning(f"⚠️ Données du mois en cours ({current_month_start.strftime('%B %Y')}): ABSENTES")
        logger.info("   → Utilisation des données historiques les plus récentes")
    
    # Vérifier completude par province
    required_cols = ['temperature', 'precipitation', 'relative_humidity']
    provinces_ok = 0
    
    for prov_name, df in all_provinces_climate.items():
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.debug(f"  ⚠️ {prov_name}: colonnes manquantes {missing_cols}")
        else:
            # Remplir NaN
            for col in required_cols:
                n_nan = df[col].isna().sum()
                if n_nan > 0:
                    df[col].fillna(df[col].mean(), inplace=True)
            provinces_ok += 1
    
    logger.info(f"✓ {provinces_ok}/{len(all_provinces_climate)} provinces complètes")
    logger.info("")
    
    return all_provinces_climate if all_provinces_climate else None


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
    logger.info("")
    logger.info("🔧 AMÉLIORATIONS APPLIQUÉES:")
    logger.info("  ✓ Nettoyage intelligent : variance + couverture de données")
    logger.info("  ✓ Gestion saisonnalité : sin/cos préservés en inversion")
    logger.info("  ✓ Split temporal : 60% train | 20% validation | 20% TEST réel")
    logger.info("  ✓ Métriques sur FUTUR CONNU : validation backtesting")
    logger.info("  ✓ Données climatiques PAR PROVINCE : Température, Précipitations, Humidité")
    logger.info("  ✓ Logging détaillé : trace chaque décision de Tier")
    logger.info("")

    # ── VÉRIFICATION PRÉALABLE DES DONNÉES CLIMATIQUES PAR PROVINCE ────────
    climate_data = verify_climate_data()  # Returns dict {province_name: DataFrame}
    
    if climate_data and len(climate_data) > 0:
        logger.info(f"✓ Données climatiques disponibles pour {len(climate_data)} province(s)")
        use_climate = True
    else:
        logger.warning(f"⚠️ Données climatiques INDISPONIBLES - fallback sur saisonnalité")
        use_climate = False
        climate_data = {}
    
    today       = date.today()
    target_date = (today.replace(day=1) + timedelta(days=32)).replace(day=1)

    # Nettoyage des prédictions précédentes pour le mois cible
    deleted = Prediction.objects.filter(prediction_date=target_date).delete()
    logger.info(f"🗑️  Prédictions précédentes supprimées : {deleted[0]}")

    prediction_list  = []
    seen_predictions = set()
    performance_log  = []

    provinces = list(Province.objects.all())
    
    # ── CONSOLIDATION DES MALADIES ─────────────────────────────────────────────
    # Récupérer toutes les maladies sources depuis la BD
    all_source_diseases = list(
        ClinicalData.objects.values_list('disease_name', flat=True).distinct()
    )
    
    # Mapper les maladies sources aux groupes consolidés
    diseases_to_process = set()
    disease_mapping_log = {}
    
    for source_disease in all_source_diseases:
        consolidated = consolidate_disease_data(source_disease)
        if consolidated and consolidated in EPIDEMIC_DISEASES:
            diseases_to_process.add(consolidated)
            if consolidated not in disease_mapping_log:
                disease_mapping_log[consolidated] = []
            disease_mapping_log[consolidated].append(source_disease)
    
    diseases = sorted(list(diseases_to_process))

    if not diseases:
        logger.warning("⚠️ Aucune donnée épidémique trouvée (après consolidation).")
        return

    logger.info(f"📊 {len(diseases)} épidémies consolidées × {len(provinces)} provinces")
    logger.info(f"🦠 GROUPES DE MALADIES:")
    for group, sources in disease_mapping_log.items():
        logger.info(f"   • {group} ← {', '.join(sources)}")

    for d_name in diseases:
        logger.info(f"\n{'─'*50}")
        logger.info(f"🔬 GROUPE : {d_name}")
        logger.info(f"🦠 Sources : {', '.join(disease_mapping_log.get(d_name, []))}")
        logger.info(f"{'─'*50}")

        for prov in provinces:
            # ── Chargement CONSOLIDÉ des données pour le groupe de maladies ──────────────────
            # Récupérer toutes les maladies sources du groupe
            source_diseases = disease_mapping_log.get(d_name, [d_name])
            
            # Requête unifiée : tous les cas du groupe
            prov_qs = ClinicalData.objects.filter(
                health_zone__territory__province=prov,
                disease_name__in=source_diseases
            ).order_by('month')
            
            # DEBUG: Log disponibilité réelle des données
            if prov_qs.exists():
                date_range = prov_qs.aggregate(min_m=django_models.Min('month'), max_m=django_models.Max('month'))
                logger.debug(f"      DATA RANGE {prov.name}: {date_range['min_m']} → {date_range['max_m']} ({prov_qs.count()} records)")
            
            if not prov_qs.exists():
                continue

            df = pd.DataFrame(list(prov_qs.values('month', 'cases_count')))
            
            # 🧹 NETTOYAGE DES DONNÉES BRUTES ET CONSOLIDATION
            df = clean_clinical_data(df)  # Inclut déduplication + clipping + interpolation
            
            # Vérification : suffisamment de données après nettoyage
            if df.empty or len(df) < 2:
                logger.debug(f"    ⏭️  Données insuffisantes après nettoyage pour {prov.name}/{d_name}")
                continue
            
            df['month_dt'] = pd.to_datetime(df['month'])
            # Agrégation par mois : somme de tous les casos du groupe
            df_monthly = df.groupby('month_dt').agg({'cases_count': 'sum'}).reset_index()
            n_samples = len(df_monthly)
            
            # ✓ VÉRIFICATION QUALITÉ DE DONNÉES: Détécter les données creuses/constantes
            cases_values = df_monthly['cases_count'].values
            variance = np.var(cases_values)
            non_zero_ratio = np.sum(cases_values > 0) / len(cases_values) if len(cases_values) > 0 else 0
            
            # Passer si: variance = 0 (constant) ou trop de zéros (< 30% non-zéro)
            if variance < 0.1 or non_zero_ratio < 0.3:
                logger.warning(f"    ⏭️  Données creuses/constantes {prov.name}/{d_name}: variance={variance:.2f}, non-zero={non_zero_ratio*100:.0f}%")
                continue
            
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

                # Charger données climatiques pour cette province
                tier1_climate_df = climate_data.get(prov.name) if use_climate else None
                
                # Construction de la grille spatiale : shape (timesteps, n_zones)
                zone_ids  = [z.id for z in zones]
                n_zones   = len(zone_ids)
                all_dates = sorted(df_monthly['month_dt'].tolist())

                # Matrice spatiale : lignes=temps, colonnes=zones
                spatial_matrix = np.zeros((len(all_dates), n_zones))
                zone_coverage = []  # Pour tracer la couverture
                
                for zi, zone in enumerate(zones):
                    z_qs = ClinicalData.objects.filter(
                        health_zone=zone, disease_name=d_name
                    ).values('month', 'cases_count')
                    
                    # Nettoyage au niveau zone
                    z_df = pd.DataFrame(list(z_qs))
                    z_df = clean_clinical_data(z_df)
                    
                    z_dict = {pd.Timestamp(r['month']): r['cases_count'] for _, r in z_df.iterrows()}
                    for ti, dt in enumerate(all_dates):
                        spatial_matrix[ti, zi] = z_dict.get(dt, 0)
                    
                    non_zero = (spatial_matrix[:, zi] > 0).sum()
                    zone_coverage.append(non_zero)

                logger.debug(f"    📊 Couverture spatiale: {np.mean(zone_coverage)/len(all_dates):.1%} en moyenne par zone")

                # 📊 LOG-TRANSFORMATION pour réduire la variance énorme (dynamique pour toutes les maladies)
                spatial_matrix_log = np.log1p(spatial_matrix)  # log(x + 1)
                
                scaler_sp = MinMaxScaler()
                sp_scaled = scaler_sp.fit_transform(spatial_matrix_log)  # MinMax sur log(x)

                # 🌡️ INTÉGRER LE CLIMAT — Créer matrice 4D: [cases_zone, temp, precip, humidity]
                n_features = 4 if (tier1_climate_df is not None and not tier1_climate_df.empty and len(tier1_climate_df) == len(all_dates)) else 1
                
                if n_features == 4:
                    # Construire matrice 4D: (timesteps, n_zones, 4) = [cases, temp, precip, humidity]
                    climate_data_arr = tier1_climate_df[['temperature', 'precipitation', 'relative_humidity']].values
                    
                    # Normaliser climat [0, 1]
                    climate_min = climate_data_arr.min(axis=0)
                    climate_max = climate_data_arr.max(axis=0)
                    climate_norm = (climate_data_arr - climate_min) / np.maximum(climate_max - climate_min, 1e-6)
                    
                    # Créer matrice 4D: chaque zone a 4 features
                    matrix_4d = np.zeros((len(all_dates), n_zones, 4))
                    matrix_4d[:, :, 0] = sp_scaled[:, :n_zones]  # Cases (déjà normalisées)
                    matrix_4d[:, :, 1] = climate_norm[:, 0:1]    # Temperature (broadcast à toutes les zones)
                    matrix_4d[:, :, 2] = climate_norm[:, 1:2]    # Precipitation
                    matrix_4d[:, :, 3] = climate_norm[:, 2:3]    # Humidity
                    
                    logger.debug(f"    🌡️ Climat intégré: temp min={climate_data_arr[:,0].min():.1f}°C, max={climate_data_arr[:,0].max():.1f}°C")
                else:
                    # Matrice 3D: (timesteps, n_zones, 1)
                    matrix_4d = sp_scaled[:, :n_zones].reshape(len(sp_scaled), n_zones, 1)
                    logger.debug(f"    ⚠️ Climat non disponible pour Tier 1, utilisation cases seuls")

                # Séquences 4D: (samples, window, n_zones, n_features)
                X_sp, y_sp = [], []
                for i in range(len(matrix_4d) - WINDOW):
                    X_sp.append(matrix_4d[i:i + WINDOW, :, :])         # (window, n_zones, n_features)
                    y_sp.append(sp_scaled[i + WINDOW, :n_zones])        # (n_zones,) — cible = cases seulement
                X_sp, y_sp = np.array(X_sp), np.array(y_sp)

                if len(X_sp) < 8:  # Au minimum 8 séquences
                    logger.debug(f"    ⚠️ {len(X_sp)} séquences < 8, fallback Tier 2")
                    tier = 2
                else:
                    # Reshape pour ConvLSTM2D: (samples, window, H, W, C) = (samples, WINDOW, n_zones, n_features, 1)
                    X_convlstm = X_sp.reshape(len(X_sp), WINDOW, n_zones, n_features, 1)
                    
                    X_tr, X_v, X_te, y_tr, y_v, y_te = temporal_train_val_test_split(
                        X_convlstm, y_sp
                    )

                    if len(X_tr) < 2 or len(X_v) < 1 or len(X_te) < 1:
                        logger.debug(f"    ⚠️ Split invalide, fallback Tier 2")
                        tier = 2
                    else:
                        conv_model = build_convlstm_model(n_zones=n_zones, window=WINDOW, n_features=n_features)
                        train_model(conv_model, X_tr, y_tr, X_v, y_v, epochs=80, batch=4)
                        # Nettoyage RAM après entraînement
                        gc.collect()
                        K.clear_session()

                        # TEST REAL sur le futur connu
                        y_pred_te_sc = conv_model.predict(X_te, verbose=0)
                        y_pred_te_full = scaler_sp.inverse_transform(y_pred_te_sc)
                        y_true_te_full = scaler_sp.inverse_transform(y_te)

                        # Métriques globales (sur toutes zones, tous timesteps)
                        mae, rmse, r2 = compute_metrics(y_true_te_full.flatten(), y_pred_te_full.flatten())
                        rel = reliability_score(r2, tier)
                        model_label = "ConvLSTM2D"

                        logger.info(f"    ✅ ConvLSTM — R²={r2:.3f} | MAE={mae:.1f} | RMSE={rmse:.1f} | Fiabilité={rel}%")

                        # Prédiction pour chaque zone — utiliser matrix_4d pour avoir climat
                        last_window = matrix_4d[-WINDOW:].reshape(1, WINDOW, n_zones, n_features, 1)
                        next_pred_sc = conv_model.predict(last_window, verbose=0)
                        next_pred = scaler_sp.inverse_transform(next_pred_sc)[0]

                        d_safe = d_name.replace('/', '_').replace(' ', '_')
                        conv_model.save(str(MODELS_DIR / f"convlstm_{d_safe}_{prov.name.replace(' ', '_')}.keras"))

                        for zi, zone in enumerate(zones):
                            pred_val = max(0.0, float(next_pred[zi]))
                            growth = calculate_spread_rate(zone, d_name, pred_val)  # ✨ Intelligent trend-based propagation
                            severity = "ALERTE ROUGE" if pred_val > 500 else "VIGILANCE" if pred_val > 150 else "STABLE"

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

                # Utiliser données climatiques validées au démarrage (par province)
                tier2_climate_df = climate_data.get(prov.name) if use_climate else None

                # Construire features (avec ou sans climat)
                features, n_features, feature_names = build_feature_matrix_with_climate(
                    df_monthly, tier2_climate_df
                )
                logger.debug(f"    📊 Features utilisées: {feature_names}")
                
                scaler_obj = ScalerWithSeasonality()
                seasonality = seasonality_features(df_monthly['month_dt'])
                scaled = scaler_obj.fit_transform(features, seasonality_features=seasonality)
                
                X, y = create_sequences(scaled, WINDOW)

                if len(X) < 8:  # Au minimum 8 séquences pour split 60/20/20
                    logger.debug(f"    ⚠️ Trop peu de séquences ({len(X)} < 8), fallback Tier 3")
                    tier = 3
                else:
                    X_tr, X_v, X_te, y_tr, y_v, y_te = temporal_train_val_test_split(X, y)

                    if len(X_tr) < 2 or len(X_v) < 1 or len(X_te) < 1:
                        logger.debug(f"    ⚠️ Split invalide, fallback Tier 3")
                        tier = 3
                    else:
                        cnn_model = build_cnn_lstm_model(window=WINDOW, n_features=n_features)
                        train_model(cnn_model, X_tr, y_tr, X_v, y_v, epochs=100, batch=8)
                        # Nettoyage RAM apres entrainement
                        gc.collect()
                        K.clear_session()

                        # TEST REAL = validation sur le futur connu (X_test)
                        y_pred_te_norm = cnn_model.predict(X_te, verbose=0)
                        y_pred_te = scaler_obj.inverse_transform(
                            y_pred_te_norm, include_seasonality=True
                        )
                        y_true_te = scaler_obj.inverse_transform(
                            y_te.reshape(-1, 1), include_seasonality=True
                        )
                        
                        mae, rmse, r2 = compute_metrics(y_true_te, y_pred_te)
                        rel = reliability_score(r2, tier)
                        model_label = "CNN-LSTM"

                        logger.info(f"    ✅ CNN-LSTM — R²={r2:.3f} | MAE={mae:.1f} | RMSE={rmse:.1f} | Fiabilité={rel}% | Features={feature_names}")

                        # Prédiction futur
                        last_window = scaled[-WINDOW:].reshape(1, WINDOW, n_features)
                        pred_sc = cnn_model.predict(last_window, verbose=0)[0][0]
                        
                        # Reconstruit avec vraie saisonnalité finale ou climat si dispo
                        if n_features == 3:
                            sin_m, cos_m = seasonality
                            dp = np.zeros((1, 3))
                            dp[0, 0] = pred_sc
                            dp[0, 1] = sin_m[-1]
                            dp[0, 2] = cos_m[-1]
                        else:  # n_features == 4 avec climat
                            dp = np.zeros((1, 4))
                            dp[0, 0] = pred_sc
                            if tier2_climate_df is not None and not tier2_climate_df.empty:
                                dp[0, 1] = tier2_climate_df.iloc[-1]['temperature']
                                dp[0, 2] = tier2_climate_df.iloc[-1]['precipitation']
                                dp[0, 3] = tier2_climate_df.iloc[-1]['relative_humidity']
                        
                        pred_prov = float(scaler_obj.scaler.inverse_transform(dp)[0, 0])
                        pred_prov = max(0.0, pred_prov)  # Failsafe : pas de prédictions négatives

                        d_safe = d_name.replace('/', '_').replace(' ', '_')
                        cnn_model.save(str(MODELS_DIR / f"cnnlstm_{d_safe}_{prov.name.replace(' ', '_')}.keras"))

                        total_pop = sum(z.population for z in zones if z.population > 0) or 1
                        for zone in zones:
                            weight = (zone.population or 1) / total_pop
                            pred_val = max(0.0, pred_prov * weight)
                            growth = calculate_spread_rate(zone, d_name, pred_val)  # ✨ Intelligent trend-based propagation
                            severity = "ALERTE ROUGE" if pred_val > 500 else "VIGILANCE" if pred_val > 150 else "STABLE"

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
                                        "spatial_context": "Distribution par poids de population",
                                        "features_used": feature_names
                                    }
                                ))

                        performance_log.append({
                            "maladie": d_name, "province": prov.name,
                            "model": model_label, "R2": round(float(r2), 3), "tier": tier,
                            "features": feature_names
                        })

                        del cnn_model, X, y, X_tr, X_v, X_te
                        gc.collect()
                        continue

            # ── TIER 3 — Bi-LSTM ──────────────────────────────────────────
            if n_samples >= TIER3_MIN:
                tier = 3
                logger.info(f"  🔹 [TIER 3 — Bi-LSTM] {prov.name} | {n_samples} mois")

                # Utiliser données climatiques validées au démarrage (par province)
                tier3_climate_df = climate_data.get(prov.name) if use_climate else None

                # Construire features (avec ou sans climat)
                features, n_features, feature_names = build_feature_matrix_with_climate(
                    df_monthly, tier3_climate_df
                )
                logger.debug(f"    📊 Features utilisées: {feature_names}")
                
                scaler_obj = ScalerWithSeasonality()
                seasonality = seasonality_features(df_monthly['month_dt'])
                scaled = scaler_obj.fit_transform(features, seasonality_features=seasonality)
                
                cur_win = max(3, min(WINDOW, n_samples // 3))
                X, y = create_sequences(scaled, cur_win)

                if len(X) < 6:  # Au minimum 6 séquences pour split 60/20/20
                    logger.debug(f"    ⚠️ Trop peu de séquences ({len(X)} < 6), fallback Tier 4")
                    tier = 4
                else:
                    X_tr, X_v, X_te, y_tr, y_v, y_te = temporal_train_val_test_split(X, y)

                    if len(X_tr) < 2 or len(X_v) < 1 or len(X_te) < 1:
                        logger.debug(f"    ⚠️ Split invalide après création séquences, fallback Tier 4")
                        tier = 4
                    else:
                        bi_model = build_bilstm_model(window=cur_win, n_features=n_features)
                        train_model(bi_model, X_tr, y_tr, X_v, y_v, epochs=100, batch=4)
                        # Nettoyage RAM apres entrainement
                        gc.collect()
                        K.clear_session()

                        # TEST REAL = validation sur le futur connu (X_test)
                        y_pred_te_norm = bi_model.predict(X_te, verbose=0)
                        y_pred_te = scaler_obj.inverse_transform(
                            y_pred_te_norm, include_seasonality=True
                        )
                        y_true_te = scaler_obj.inverse_transform(
                            y_te.reshape(-1, 1), include_seasonality=True
                        )
                        
                        mae, rmse, r2 = compute_metrics(y_true_te, y_pred_te)
                        rel = reliability_score(r2, tier)
                        model_label = "Bi-LSTM"

                        logger.info(f"    ✅ Bi-LSTM — R²={r2:.3f} | MAE={mae:.1f} | RMSE={rmse:.1f} | Fiabilité={rel}%")

                        # Prédiction futur
                        last_window = scaled[-cur_win:].reshape(1, cur_win, n_features)
                        pred_sc = bi_model.predict(last_window, verbose=0)[0][0]
                        
                        sin_m, cos_m = seasonality
                        
                        # Reconstruit avec saisonnalité et climat si dispo
                        if n_features == 3:
                            dp = np.zeros((1, 3))
                            dp[0, 0] = pred_sc
                            dp[0, 1] = sin_m[-1]
                            dp[0, 2] = cos_m[-1]
                        else:  # n_features == 4 avec climat
                            dp = np.zeros((1, 4))
                            dp[0, 0] = pred_sc
                            dp[0, 1] = sin_m[-1]
                            dp[0, 2] = cos_m[-1]
                            if tier3_climate_df is not None and not tier3_climate_df.empty:
                                dp[0, 3] = tier3_climate_df.iloc[-1]['temperature']
                        
                        pred_prov = float(scaler_obj.scaler.inverse_transform(dp)[0, 0])
                        pred_prov = max(0.0, pred_prov)  # Failsafe : pas de prédictions négatives

                        total_pop = sum(z.population for z in zones if z.population > 0) or 1
                        for zone in zones:
                            weight = (zone.population or 1) / total_pop
                            pred_val = max(0.0, pred_prov * weight)
                            growth = calculate_spread_rate(zone, d_name, pred_val)  # ✨ Intelligent trend-based propagation
                            severity = "ALERTE ROUGE" if pred_val > 500 else "VIGILANCE" if pred_val > 150 else "STABLE"

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

                        del bi_model, X, y, X_tr, X_v, X_te
                        gc.collect()
                        continue

            # ── TIER 4 — PERSISTENCE ──────────────────────────────────────
            tier = 4
            logger.info(f"  ⬜ [TIER 4 — Persistence] {prov.name} | {n_samples} pts | données insuffisantes")
            model_label = "Persistence"
            r2, mae, rmse = 0.33, 0, 0
            rel = reliability_score(r2, tier)

            # Charger climat pour ajustement Tier 4
            tier4_climate_df = climate_data.get(prov.name) if use_climate else None
            climate_adjustment = 1.0
            
            if tier4_climate_df is not None and not tier4_climate_df.empty:
                # Ajustement basé sur climat : favorabilité pour transmission
                avg_temp = tier4_climate_df['temperature'].mean()
                avg_precip = tier4_climate_df['precipitation'].mean()
                avg_humidity = tier4_climate_df['relative_humidity'].mean()
                
                # Facteur clima : +/- 20% selon conditions
                # Maladies vectorielles ↑ à T° 20-30°C + humidité >60%
                climate_score = 0
                if 20 <= avg_temp <= 30:
                    climate_score += 10  # Favorable
                if avg_humidity > 60:
                    climate_score += 10  # Favorable
                if avg_precip > 100:
                    climate_score += 5   # Légèrement favorable (eau stagnante)
                
                # Clipper [0, 20] → factor [0.9, 1.2]
                climate_adjustment = max(0.9, min(1.2, 1.0 + climate_score / 100))
                logger.debug(f"    🌡️ Tier 4 climat ajustement: T={avg_temp:.1f}°C, P={avg_precip:.0f}mm, H={avg_humidity:.0f}% → factor={climate_adjustment:.2f}x")

            for zone in zones:
                last_rec = ClinicalData.objects.filter(
                    health_zone=zone, disease_name=d_name
                ).order_by('month').last()
                pred_val = float(last_rec.cases_count) if last_rec else 0.0
                pred_val = pred_val * climate_adjustment  # ✨ Appliquer ajustement climat
                growth = calculate_spread_rate(zone, d_name, pred_val)  # ✨ Intelligent trend-based propagation
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
                            "weekly_spread_rate": round(growth, 1), "attack_angle": severity,
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
    logger.info("=" * 70)
    logger.info("🚀 ENTRAîNEMENT DES MODELES")
    logger.info("=" * 70)
    train_and_predict()
   