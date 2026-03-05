# 🏥 HealthDRC Intelligence v3.0 (NeuralCore Adaptive)

[English Version](#english-version) | [Version Française](#version-française)

---

<a name="english-version"></a>
# 🌍 English Version

## 📖 Project Overview
**HealthDRC Intelligence** is a state-of-the-art spatio-temporal prediction platform designed for epidemiological surveillance in the Democratic Republic of Congo (DRC). It leverages Deep Learning (Bi-LSTM) and Climate data to forecast disease outbreaks (Malaria, Measles, Tuberculosis, etc.) at the Health Zone level.

### 🧠 The "NeuralCore" Engine
The system uses an **Adaptive Multi-Tier Pipeline** to ensure 100% geographical coverage, even in regions with sparse historical data:
1.  **Deep Learning (Bi-LSTM)**: Used for long, rich time series ($N \ge 8$). Captures complex seasonality and non-linear trends.
2.  **Statistical Linear Trend**: Used for annual or shorter series ($4 \le N < 8$).
3.  **Persistence Fallback**: Guarantees a baseline forecast for zones with extremely rare data ($N < 4$).

### 🚀 Key Features
-   **NeuralCore Adaptive Training**: Automated training cycles that select the best model per province.
-   **Intelligence Hub**: A contextual recommendation engine that explains AI decisions in plain language.
-   **Premium Dashboard**: Real-time visualization using Heatmaps, Geo-localized markers, and predictive metrics ($R^2$, MAE, RMSE).
-   **Sparse Data Shield**: High-precision indicators and transparency banners (`⚠️ Sparse Data`) for ethically sound scientific reporting.

---

## 🏗️ Technical Stack
-   **Backend**: Python 3.14+, Django LTS ORM.
-   **AI/ML**: TensorFlow 2.16+ (Bidirectional LSTM), Scikit-Learn.
-   **Frontend**: HTMX, Alpine.js, TailwindCSS (CDN), Folium (Maps).
-   **Orchestration**: Docker, Traefik (Reverse Proxy), PostgreSQL.

---

## 📂 Project Structure
```text
health/
├── bin/          # Quick-launch bash scripts (up, down, logs, sync)
├── analytics/    # Django Core: AI logic, Views, Intelligence Engine
├── scripts/      # Data Pipelines: HDX Ingestion, NeuralCore Training, Diagnostics
├── templates/    # Responsive Premium Dashboard (UI/UX)
├── data/         # [Git Ignored] Local storage for HDX, Climate & Shapefiles
└── models/       # [Git Ignored] Trained .keras and .pkl models
```

---

<a name="version-française"></a>
# 🇫🇷 Version Française

## 📖 Présentation du Projet
**HealthDRC Intelligence** est une plateforme de pointe dédiée à l'analyse spatio-temporelle et à la surveillance épidémiologique en République Démocratique du Congo (RDC). Elle utilise le Deep Learning (Bi-LSTM) et les données climatiques pour prédire l'évolution des maladies (Paludisme, Rougeole, Tuberculose, etc.) à l'échelle des Zones de Santé.

### 🧠 Le Moteur "NeuralCore"
Le système repose sur un **Pipeline Adaptatif à 3 Paliers** pour garantir une couverture géographique à 100%, même dans les régions avec peu de données historiques :
1.  **Deep Learning (Bi-LSTM)** : Pour les séries temporelles riches ($N \ge 8$). Capture les saisonnalités complexes.
2.  **Tendance Linéaire Statistique** : Pour les séries courtes ou annuelles ($4 \le N < 8$).
3.  **Repli par Persistance** : Assure une prévision de base pour les zones à données ultra-rares ($N < 4$).

### 🚀 Fonctionnalités Clés
-   **Entraînement Adaptatif NeuralCore** : Sélection automatique du meilleur modèle selon la densité de données par province.
-   **Hub d'Intelligence** : Un moteur de recommandation contextuelle qui explique les décisions de l'IA en langage clair.
-   **Dashboard Premium** : Visualisation en temps réel via Heatmaps, marqueurs géolocalisés et métriques prédictives ($R^2$, MAE, RMSE).
-   **Transparence Scientifique** : Affichage systématique du score de précision et badge `⚠️ Faible Densité` pour une lecture éthique des résultats.

---

## �️ Stack Technique
-   **Backend** : Python 3.14+, Django LTS ORM.
-   **AI/ML** : TensorFlow 2.16+ (Bidirectional LSTM), Scikit-Learn.
-   **Frontend** : HTMX, Alpine.js, TailwindCSS (CDN), Folium (Cartographie).
-   **Infrastructure** : Docker, Traefik (Proxy inverse), PostgreSQL.

---

## 🚀 Guide de Lancement / Setup
1.  **Démarrer l'infrastructure** : `./bin/up`
2.  **Lancer l'entraînement IA** : `docker-compose exec app python3 scripts/train_models.py`
3.  **Accéder au site** : `http://localhost:8000` (ou `http://health.localhost`)

---
*Developed for health excellence in DR Congo by Yannick Magayane.*
*Optimisé pour la précision scientifique et la résilience des données.*
