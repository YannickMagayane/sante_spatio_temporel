import os
import django
import numpy as np
import logging
import pandas as pd
from datetime import date, datetime, timedelta
import random
import json
from pathlib import Path

# Setup Django Environment
import sys
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from analytics.models import Province, Territory, HealthZone, ClinicalData, Prediction
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = Path("/app/models")
REPORTS_DIR = Path("/app/reports")
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

DEFAULT_WINDOW_SIZE = 4 # Reduced from 6 for HDX data compatibility
MIN_TOTAL_SAMPLES = 8   # Minimum to attempt LSTM (split 6/2 for example)

def create_windowed_dataset(data, window_size=6):
    """
    Transforme une série temporelle en dataset supervisé.
    Format: (samples, window_size, features)
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size), :])
        y.append(data[i + window_size, 0]) # On prédit le premier feature (cases_count)
    return np.array(X), np.array(y)

def build_advanced_model(input_shape):
    """
    Architecture Bidirectionnelle LSTM avec Dropout pour la régularisation.
    """
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def get_seasonality_features(dates):
    """
    Génère des features cycliques (Sin/Cos) pour encoder le mois de l'année.
    """
    months = np.array([d.month for d in dates])
    sin_month = np.sin(2 * np.pi * months / 12)
    cos_month = np.cos(2 * np.pi * months / 12)
    return sin_month, cos_month

def train_and_predict():
    logger.info("🧠 NEURALCORE ADVANCED TRAINING: SPATIO-TEMPORAL REFACTOR")
    
    today = date.today()
    target_date = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
        
    Prediction.objects.all().delete()
    prediction_list = []
    seen_predictions = set()
    performance_log = []
    
    provinces = Province.objects.all()
    health_zones = HealthZone.objects.all()
    diseases = ClinicalData.objects.values_list('disease_name', flat=True).distinct()
    
    if not diseases:
        logger.warning("⚠️ No clinical data found.")
        return

    for d_name in diseases:
        logger.info(f"🔮 Training for: {d_name}")
        for prov in provinces:
            # Récupération des données agrégées par province pour l'apprentissage global
            prov_data = ClinicalData.objects.filter(
                health_zone__territory__province=prov, 
                disease_name=d_name
            ).order_by('month')
            
            if not prov_data.exists():
                continue

            # Conversion en DataFrame pour manipulation aisée
            df = pd.DataFrame(list(prov_data.values('month', 'cases_count')))
            df['month_dt'] = pd.to_datetime(df['month'])
            df = df.sort_values('month_dt')
            
            # Agrégation par mois (au cas où plusieurs zones ont des données le même mois)
            df_monthly = df.groupby('month_dt').agg({'cases_count': 'sum'}).reset_index()
            
            # 1. Selection du modèle selon la densité (N samples)
            n_samples = len(df_monthly)
            
            # --- TIER 1: BI-LSTM (N >= 8) ---
            if n_samples >= MIN_TOTAL_SAMPLES:
                current_window = max(3, min(DEFAULT_WINDOW_SIZE, n_samples // 3))
                logger.info(f"🤖 [BI-LSTM] {prov.name} / {d_name}: Window size {current_window}")
                
                sin_m, cos_m = get_seasonality_features(df_monthly['month_dt'])
                features = np.stack([df_monthly['cases_count'].values, sin_m, cos_m], axis=1)

                test_size = 2
                split_idx = n_samples - test_size - current_window
                
                train_data = features[:split_idx + current_window]
                test_data = features[split_idx:]

                scaler = MinMaxScaler(feature_range=(0, 1))
                train_scaled = scaler.fit_transform(train_data)
                test_scaled = scaler.transform(test_data)

                X_train, y_train = create_windowed_dataset(train_scaled, current_window)
                X_test, y_test = create_windowed_dataset(test_scaled, current_window)

                if len(X_train) < 2: 
                     model_type_label = "Persistence"
                     r2, mae, rmse = 0.2, 0, 0
                else:
                    model = build_advanced_model((current_window, 3))
                    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=4, callbacks=[early_stop], verbose=0)

                    y_pred_scaled = model.predict(X_test, verbose=0)
                    def inv(v, s):
                        d = np.zeros((len(v), 3)); d[:, 0] = v.flatten()
                        return s.inverse_transform(d)[:, 0]

                    y_pred = inv(y_pred_scaled, scaler)
                    y_true = inv(y_test.reshape(-1, 1), scaler)
                    
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    r2 = r2_score(y_true, y_pred)
                    
                    y_naive = inv(X_test[:, -1, 0].reshape(-1, 1), scaler)
                    mae_naive = mean_absolute_error(y_true, y_naive)
                    gain = ((mae_naive - mae) / mae_naive * 100) if mae_naive > 0 else 5.0
                    model_type_label = "Bi-LSTM"
                    
                    d_safe = d_name.replace('/', '_').replace(' ', '_')
                    model.save(str(MODELS_DIR / f"model_adv_{d_safe}_{prov.name.replace(' ', '_')}.keras"))

            # --- TIER 2: LINEAR TREND (4 <= N < 8) ---
            elif n_samples >= 4:
                logger.info(f"📈 [LINEAR TREND] {prov.name} / {d_name}: Statistical estimation ({n_samples} pts)")
                X_lin = np.arange(n_samples).reshape(-1, 1)
                y_lin = df_monthly['cases_count'].values
                lin_model = LinearRegression().fit(X_lin, y_lin)
                
                r2 = lin_model.score(X_lin, y_lin)
                mae = mean_absolute_error(y_lin, lin_model.predict(X_lin))
                rmse = np.sqrt(mean_squared_error(y_lin, lin_model.predict(X_lin)))
                gain = float(lin_model.coef_[0] / max(1, np.mean(y_lin)) * 100) if len(lin_model.coef_) > 0 else 0.0
                model_type_label = "Linear Trend"
                
            # --- TIER 3: PERSISTENCE (N < 4) ---
            else:
                logger.info(f"📉 [FALLBACK] {prov.name} / {d_name}: Persistence forecast ({n_samples} pts)")
                model_type_label = "Persistence"
                r2, mae, rmse, gain = 0.1, 0, 0, 2.0
                last_val = df_monthly['cases_count'].iloc[-1] if n_samples > 0 else 0

            performance_log.append({
                "maladie": d_name, "province": prov.name, "model": model_type_label, "R2": round(float(r2), 3)
            })

            # --- Prédictions par Zone de Santé ---
            for zone in health_zones.filter(territory__province=prov):
                z_data = ClinicalData.objects.filter(health_zone=zone, disease_name=d_name).order_by('month')
                z_count = z_data.count()
                
                if model_type_label == "Bi-LSTM" and z_count >= current_window:
                    z_df = pd.DataFrame(list(z_data.values('month', 'cases_count')))
                    z_df['dt'] = pd.to_datetime(z_df['month'])
                    s, c = get_seasonality_features(z_df['dt'])
                    z_f = np.stack([z_df['cases_count'].values, s, c], axis=1)
                    
                    lx = scaler.transform(z_f[-current_window:])
                    px = model.predict(lx.reshape(1, current_window, 3), verbose=0)[0][0]
                    dp = np.zeros((1, 3)); dp[0, 0] = px
                    pred_val = float(scaler.inverse_transform(dp)[0, 0])
                    reliability = max(55, min(99, r2 * 100))
                
                elif model_type_label == "Linear Trend" and z_count > 0:
                    pred_val = lin_model.predict([[n_samples]])[0]
                    reliability = max(45, min(75, r2 * 100))
                else:
                    last_rec = z_data.last()
                    pred_val = float(last_rec.cases_count) if last_rec else 0.0
                    reliability = 40.0 if last_rec else 0.0

                # Calcul du taux de propagation (Growth Rate)
                last_obs = float(z_data.last().cases_count) if z_data.exists() else 0.0
                growth_rate = ((pred_val - last_obs) / max(1, last_obs)) * 100
                
                if pred_val > 500: severity = "ALERTE ROUGE : Déploiement d'urgence"
                elif pred_val > 150: severity = "VIGILANCE : Surveillance accrue"
                else: severity = "STABLE : Monitoring nominal"

                pred_key = (zone.id, d_name, target_date)
                if pred_key not in seen_predictions:
                    seen_predictions.add(pred_key)
                    prediction_list.append(Prediction(
                        health_zone=zone, disease_name=d_name, prediction_date=target_date,
                        predicted_cases=round(max(0, pred_val), 1),
                        influence_factors={
                            "reliability": round(reliability, 2), "mae": round(float(mae), 2),
                            "r2": round(float(r2), 3), "model_used": model_type_label,
                            "attack_angle": severity, "data_sparse": n_samples < 8,
                            "weekly_spread_rate": round(float(growth_rate), 1),
                            "innovation_gain": round(float(gain), 1) if 'gain' in locals() else 0.0,
                            "rmse": round(float(rmse), 2)
                        }
                    ))

    if prediction_list:
        Prediction.objects.bulk_create(prediction_list, batch_size=500)
    logger.info(f"✅ Created {len(prediction_list)} predictions")
    
    # Generate Report
    report_path = REPORTS_DIR / f"advanced_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "total_predictions": len(prediction_list),
        "performance_metrics": performance_log
    }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=4)
    
    logger.info(f"✅ ADVANCED REPORT GENERATED: {report_path}")

if __name__ == "__main__":
    train_and_predict()
