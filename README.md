# HealthDRC Intelligence v3.0 (NeuralCore Adaptive) 🛰️🏥

**Analyse Spatio-Temporelle, Deep Learning & Surveillance Épidémiologique en RDC**
**Spatio-Temporal Analysis, Deep Learning & Epidemiological Surveillance in DRC**

[English Version](#english-version) | [Version Française](#version-française)

---

<a name="english-version"></a>
# � English Version

## 📖 Project Overview
**HealthDRC Intelligence** is a cutting-edge spatio-temporal prediction platform designed for epidemiological surveillance in the Democratic Republic of Congo (DRC). It leverages Deep Learning (Bi-LSTM) and Climate data to monitor and forecast disease outbreaks (Malaria, Measles, Tuberculosis, etc.) across the **519 Health Zones** of the country.

### 🧠 The "NeuralCore" Engine
The system uses an **Adaptive Multi-Tier Pipeline** to ensure 100% geographical coverage, even in regions with sparse historical data:
1.  **Deep Learning (Bi-LSTM)**: Optimized for long, rich time series ($N \ge 8$). Captures complex seasonality and non-linear trends using Bidirectional LSTM layers.
2.  **Statistical Linear Trend**: Fallback for shorter series ($4 \le N < 8$), ensuring reliable projections where deep learning would overfit.
3.  **Persistence Engine**: Guarantees a baseline forecast for zones with extremely rare data ($N < 4$).

### 🚀 Key Features
-   **Interactive Heatmaps**: Real-time spatial visualization of health threats across the entire DRC territory.
-   **NeuralCore Adaptive Training**: Automated training cycles that dynamically select the best model architecture per province.
-   **Intelligence Hub**: A contextual recommendation engine that explains AI decisions (spread rates, precision, $R^2$) in plain language.
-   **Sparse Data Shield**: High-precision indicators and transparency banners (`⚠️ Sparse Data`) for ethically sound scientific reporting.
-   **Automated Ingestion**: Pipeline capable of processing CSV, XLSX, ZIP, and GeoJSON files from international sources (HDX, WHO).

---

## 🏗️ Technical Stack
-   **Backend**: Django 5.1 (Python 3.12+), Postgres 16.
-   **AI/ML**: TensorFlow 2.16+ (Bidirectional LSTM), Scikit-Learn (Linear Regression), NumPy, Pandas.
-   **Frontend**: HTMX (Dynamic interactions), Alpine.js (State management), TailwindCSS (CDN), Folium (Maps).
-   **Infrastructure**: Docker & Docker-Compose, Traefik v3.1 (Reverse Proxy).

---

## 🚀 Quick Setup

### 📦 Prerequisites
- Docker & Docker-Compose installed.

### �️ Installation
```bash
# Clone the project
git clone https://github.com/YannickMagayane/sante_spatio_temporel
cd sante_spatio_temporel

# Launch with Docker
./bin/up  # or docker-compose up -d --build

# Initialize Data & Models
docker-compose exec app python scripts/seed_all_zones.py    # Setup geo-registry
docker-compose exec app python scripts/ingest_hdx.py         # Ingest clinical data
docker-compose exec app python scripts/train_models.py      # Launch NeuralCore Engine
```

### � Access
Open `http://localhost:8000` or `http://health.localhost` in your browser.

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

## � Présentation du Projet
**HealthDRC Intelligence** est une plateforme de pointe dédiée à l'analyse spatio-temporelle et à la surveillance épidémiologique en République Démocratique du Congo (RDC). Elle utilise le Deep Learning (Bi-LSTM) et les données climatiques pour prédire l'évolution des maladies (Paludisme, Rougeole, Tuberculose, etc.) à l'échelle des **519 Zones de Santé**.

### 🧠 Le Moteur "NeuralCore"
Le système repose sur un **Pipeline Adaptatif à 3 Paliers** pour garantir une couverture géographique à 100%, même dans les régions avec peu de données historiques :
1.  **Deep Learning (Bi-LSTM)** : Pour les séries temporelles riches ($N \ge 8$). Analyse les cycles passés et la saisonnalité via des neurones bidirectionnels.
2.  **Tendance Linéaire Statistique** : Pour les séries courtes ou annuelles ($4 \le N < 8$), évitant le sur-apprentissage.
3.  **Repli par Persistance** : Assure une prévision de base pour les zones à données ultra-rares ($N < 4$).

### 🚀 Fonctionnalités Clés
-   **Cartographie Heatmap interactive** : Visualisation spatiale des menaces sanitaires en temps réel.
-   **Entraînement Adaptatif NeuralCore** : Sélection automatique du meilleur modèle selon la densité de données par province.
-   **Hub d'Intelligence** : Un moteur de recommandation qui transforme les métriques complexes ($R^2$, MAE) en conseils de santé publique.
-   **Transparence Scientifique** : Affichage systématique du score de précision et badge `⚠️ Faible Densité` pour une lecture éthique des résultats.
-   **Ingestion Automatisée** : Pipeline capable de traiter des fichiers hétérogènes (CSV, XLSX, GeoJSON) provenant du HDX ou de l'OMS.

---

## 🛠️ Stack Technique
-   **Backend** : Django 5.1 (Python 3.12+), Postgres 16.
-   **AI/ML** : TensorFlow/Keras (Bi-LSTM), Scikit-Learn, NumPy, Pandas.
-   **Frontend** : HTMX, Alpine.js, TailwindCSS (CDN), Folium (Cartographie).
-   **Infrastructure** : Docker, Traefik v3.1 (Proxy inverse).

---

## � Guide de Lancement

### 🛠️ Installation
```bash
# Cloner le projet
git clone https://github.com/YannickMagayane/sante_spatio_temporel
cd sante_spatio_temporel

# Lancement
./bin/up

# Initialisation des données et de l'IA
docker-compose exec app python scripts/seed_all_zones.py    # Création du répertoire géo
docker-compose exec app python scripts/ingest_hdx.py         # Ingestion des données cliniques
docker-compose exec app python scripts/train_models.py      # Lancement du moteur NeuralCore
```

---
*Developed for health excellence in DR Congo by Yannick Magayane.*
*Optimisé pour la précision scientifique et la résilience des données.*
