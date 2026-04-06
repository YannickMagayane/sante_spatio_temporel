# 🏥 HealthDRC Intelligence v3.0 — Spatio-Temporal Epidemic Forecasting

![Status: Production](https://img.shields.io/badge/Status-Production-brightgreen)
![Docker](https://img.shields.io/badge/Containerized-Docker--Compose-blue)
![Python: 3.9+](https://img.shields.io/badge/Python-3.9+-blue)
![TensorFlow: 2.16+](https://img.shields.io/badge/TensorFlow-2.16+-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Author: Yannick Magayane](https://img.shields.io/badge/Author-Yannick%20Magayane-blue)

---

## 📖 Table des Matières / Table of Contents

- [English Version](#-english-version)
- [Version Française](#-version-française)

---

<a name="-english-version"></a>
# 🇬🇧 English Version

## 📊 Project Overview

**HealthDRC Intelligence** is an advanced spatio-temporal prediction platform for epidemiological surveillance in the Democratic Republic of Congo (DRC). It uses **Multi-Tier Deep Learning Architecture** and **Climate Integration** to forecast disease outbreaks (Malaria, Cholera, Tuberculosis, HIV/AIDS) across **519 Health Zones**.

### 🎯 Key Features

✅ **Adaptive Multi-Tier Architecture** — Automatically selects the best model based on data availability
✅ **Climate Integration** — Temperature, precipitation, humidity factored into all predictions
✅ **100% Geographic Coverage** — 519 health zones, 26 provinces, full DRC territory
✅ **Real-time Dashboard** — Interactive heatmaps, charts, and analytics
✅ **Trend-based Propagation** — Detects actual epidemic spread using 6-month baselines
✅ **Docker-Containerized** — Complete system runs in Docker Compose
✅ **NeuralCore Intelligence** — Explains AI decisions in plain language
✅ **Task Management** — Stop long-running training/ingestion jobs with one click
✅ **Training Reset** — Clean slate functionality to restart training from scratch  

---

## 🚀 Quick Start with Docker

### Prerequisites
- Docker & Docker-Compose (v2.0+) installed
- 8GB RAM minimum
- 20GB disk space

### Launch in 3 Steps

```bash
# 1️⃣ Clone the project
git clone https://github.com/YannickMagayane/health-drc-intelligence.git
cd health-drc-intelligence

# 2️⃣ Start all services with Docker Compose
docker-compose up -d --build

# 3️⃣ Access the dashboard
# Open: http://localhost:8000
# Login: admin / admin_2026
```

### Check System Status

```bash
# View running containers
docker-compose ps

# Check logs
docker-compose logs -f app

# Stop all services
docker-compose down
```

---

## 🧠 Architecture: The 4-Tier System

The system intelligently selects the best model architecture based on available data:

### 📊 Tier Selection Logic

```
Zone Analysis:
├─ N_months ≥ 16 AND N_zones ≥ 2
│  └─> TIER 1: ConvLSTM2D (Spatio-Temporal)
│
├─ N_months ≥ 8 AND Climate Data Available
│  └─> TIER 2: CNN-LSTM (Patterns + Climate)
│
├─ N_months ≥ 4
│  └─> TIER 3: Bi-LSTM (Temporal Only)
│
└─ N_months < 4
   └─> TIER 4: Persistence (Statistical Fallback)
```

### **TIER 1 — ConvLSTM2D** 🌐 (Spatio-Temporal)

**When Activated**: ≥16 months + ≥2 health zones

**Architecture**:
```
Input: 4D Matrix (batch, timesteps, zones, features)
       Example: (32, 4 months, 30 zones, 4 features)
       Features: [cases, temperature, precipitation, humidity]

Layer 1: ConvLSTM2D(32 filters, kernel=3×1, padding='same')
         └─> Captures spatial patterns across zones
         └─> Understands temporal dependencies
         └─> Output: (batch, 4, 30, 32 filters)

Layer 2: ConvLSTM2D(64 filters, padding='same')
         └─> Deeper pattern extraction
         └─> Output: (batch, 4, 30, 64 filters)

Layer 3: Flatten → Dense(128) → Dropout(0.3)
Layer 4: Dense(64) → ReLU
Layer 5: Dense(1) → Output prediction

Parameters: ~500K weights
Training Time: 80 epochs × ~5 minutes = ~6.7 hours per zone
```

**Strengths**:
- Detects zone-to-zone propagation patterns
- Captures spatial clustering of diseases
- Best performance: R² = 0.95-1.0

**Example**: Malaria in Haut-Katanga (30 zones × 180 months)

---

### **TIER 2 — CNN-LSTM** 🌦️ (Patterns + Climate)

**When Activated**: ≥8 months + Climate data available

**Architecture**:
```
Input Features: 6D vector
  [cases, sin(month), cos(month), temperature, precipitation, humidity]

Layer 1: Conv1D(32 filters, kernel_size=3)
         └─> Extracts temporal patterns
         └─> Output: (batch, timesteps, 32)

Layer 2: LSTM(64 units, return_sequences=True)
         └─> Long-short-term memory
         └─> Captures seasonal cycles
         └─> Output: (batch, timesteps, 64)

Layer 3: LSTM(32 units, return_sequences=False)
         └─> Final temporal integration
         └─> Output: (batch, 32)

Layer 4: Dense(64) → ReLU → Dropout(0.3)
Layer 5: Dense(1) → Output prediction

Parameters: ~180K weights
Training Time: 100 epochs × ~3 minutes = ~5 hours per zone
```

**Strengths**:
- Integrates climate explicitly as features
- Excellent for seasonal diseases
- Good for moderate data density

**Example**: Cholera with climate data

---

### **TIER 3 — Bi-LSTM** ⏰ (Temporal)

**When Activated**: ≥4 months

**Architecture**:
```
Input: Univariate time series (cases only)

Layer 1: Bidirectional LSTM(64 units)
         └─> Forward LSTM: past → future
         └─> Backward LSTM: future → past
         └─> Captures bidirectional dependencies
         └─> Output: (batch, 128) [64 forward + 64 backward]

Layer 2: Dropout(0.3) → Dense(64) → ReLU
Layer 3: Dense(1) → Output prediction

Parameters: ~45K weights
Training Time: 60 epochs × ~1 minute = ~1 hour per zone
```

**Strengths**:
- Understands past and future context
- Effective for non-linear temporal patterns
- Lower computational cost than Tier 1-2

**Example**: Tuberculosis, HIV/AIDS

---

### **TIER 4 — Persistence** 📊 (Statistical Fallback)

**When Activated**: <4 months

**Logic**:
```
baseline = last_observation
climate_factor = 1.0

IF (20°C ≤ temp ≤ 28°C AND humidity > 60%):
    climate_factor = 1.3  # +30% boost for favorable conditions
ELIF (15°C ≤ temp ≤ 32°C):
    climate_factor = 1.1  # +10% boost for acceptable conditions

prediction = baseline × climate_factor

Parameters: 0 (statistical baseline)
Training Time: Instant
```

**Strengths**:
- Guarantees 100% geographic coverage
- No risk of overfitting on sparse data
- Provides reliable baseline

**Example**: New health zones with minimal history

---

## 🎓 Training Deep Dive

### Epoch Configuration

An **epoch** is one complete pass through all training data. Here's how we configure them:

```yaml
# Tier-specific epoch counts (optimized for convergence)
Tier 1 (ConvLSTM):    80 epochs  # Complex model, needs more iterations
Tier 2 (CNN-LSTM):   100 epochs  # Climate features require thorough learning
Tier 3 (Bi-LSTM):     60 epochs  # Simpler, converges faster
Tier 4 (Persistence): 0 epochs   # No training needed (statistical only)

# Total training time: ~52 hours for complete RDC coverage (519 zones)
```

### Training Pipeline

```bash
# Launch training with Docker (all-in-one command)
docker-compose exec app python scripts/train_models.py

# With options:
docker-compose exec app python scripts/train_models.py \
  --diseases "Malaria,Cholera" \
  --provinces all \
  --epochs auto \
  --use-climate yes
```

### What Happens During Training

1. **Data Loading** (5 min): Fetch clinical data + climate from database
2. **Preprocessing** (10 min): Normalization, gap-filling, outlier detection
3. **Tier Selection** (1 min): Analyze N_months, N_zones → pick best Tier
4. **Model Building** (2 min): Compile ConvLSTM/CNN-LSTM/Bi-LSTM
5. **Training Loop** (varies by Tier):
   - Tier 1: 80 epochs × ~5 min/epoch = 6.7 hours
   - Tier 2: 100 epochs × ~3 min/epoch = 5 hours
   - Tier 3: 60 epochs × ~1 min/epoch = 1 hour
6. **Evaluation** (5 min): Calculate R², MAE, Propagation Rate
7. **Saving** (2 min): Store .keras models + metrics in database

**Total per province**: ~2-4 hours (parallelized across 26 provinces)

---

## � Dashboard Usage

Access the dashboard at `http://localhost:8000/analysis/disease/`

### Available Views

```
Dashboard Graphs:
├─ Chart 1: Top 10 Affected Zones
│  └─ Shows predicted cases + model tier used
│
├─ Chart 2: Model Performance
│  └─ R² and Reliability % by Tier
│
├─ Chart 3: Temporal Evolution
│  └─ Historical (2014+) + Recent predictions
│
├─ Chart 4: Tier Distribution
│  └─ How many predictions use each Tier
│
├─ Chart 5: Climate Impact
│  └─ Pearson correlation (temp/precip vs cases)
│
└─ Heatmap: Geographic Distribution
   └─ Color intensity by cases + climate boost
```

### Task Management

Access the task management interface at `http://localhost:8000/tasks/`

- **View running/completed tasks** with real-time progress
- **Stop long-running jobs** (training/ingestion) with one click
- **Auto-refresh** every 3 seconds for running tasks
- **View logs** (output and errors) for any task

```
Task Types:
├─ 🧠 Entraînement des Modèles (training)
├─ 📥 Ingestion des Données HDX (data sync)
└─ 📦 Ingestion Données Climatiques (climate data)
```

### Filters

```bash
# By disease
?disease=Malaria          # or Cholera, Tuberculosis, HIV_AIDS, all

# By province
?province=Haut-Katanga    # or any of 26 provinces, all

# Combined
?disease=Cholera&province=Kinshasa
```

---

## 🔧 Complete Project Structure

```
health-drc-intelligence/
├── README.md                         # This file (Quick start + Overview)
├── README_MAIN.md                    # Complete technical documentation
├── .gitignore                        # Git ignore rules (excludes models, data, .docx)
├── docker-compose.yml                # Docker service orchestration ⭐
├── requirements.txt                  # Python dependencies
├── manage.py                         # Django management
│
├── 📁 core/                          # Django configuration
│   ├── settings.py                   # Global settings, DB config
│   ├── urls.py                       # Main URL patterns
│   ├── asgi.py / wsgi.py             # ASGI/WSGI entry points
│
├── 📁 analytics/                     # AI & Dashboard Application
│   ├── views.py                      # Dashboard views + API logic
│   ├── models.py                     # DB models (Prediction, ClinicalData)
│   ├── intelligence.py               # NeuralCore explanation engine
│   ├── urls.py                       # API endpoints
│   ├── migrations/                   # Database migrations
│   ├── management/commands/          # Custom Django commands
│   ├── templates/analytics/
│   │   ├── dashboard.html            # Main dashboard UI
│   │   ├── disease_detail.html
│   │   └── province_analysis.html
│   └── static/css/ & js/             # Styles & scripts
│
├── 📁 scripts/                       # Training & Data Pipeline
│   ├── train_models.py               # ⭐ MAIN TRAINING ENGINE
│   ├── ingest_hdx.py                 # HDX data import
│   ├── seed_all_zones.py             # Geographic setup
│   ├── diagnose_data.py              # Data quality checks
│   └── climate_data_fetcher.py       # Climate data download
│
├── 📁 data/                          # Local data storage (GITIGNORED)
│   ├── raw_hdx/                      # Raw WHO data
│   ├── processed/                    # Cleaned data
│   ├── climat/
│   │   ├── rdc_climate_monthly.pkl
│   │   └── provinces/                # Climate per province
│   └── geospatial/                   # Shapefiles, GeoJSON
│
├── 📁 models/                        # Trained models (GITIGNORED)
│   ├── convlstm_*.keras              # Tier 1 models
│   ├── cnnlstm_*.keras               # Tier 2 models
│   ├── bilstm_*.keras                # Tier 3 models
│   └── metadata.json
│
├── 📁 docs/                          # Detailed documentation
│   ├── ARCHITECTURE_MODELES_COMPLET.md
│   ├── DOCUMENTATION_TRAIN_TEST.md
│   ├── RAPPORT_Paludisme_Cholera_DataComparison.md
│   └── ... (7 more documents)
│
├── 📁 reports/                       # Word reports (GITIGNORED)
│   ├── RAPPORT_PALUDISME_CHOLERA_COMPLET.docx
│   └── ... (4 more Word docs)
│
├── 📁 compose/                       # Docker service configs
│   ├── app/Dockerfile
│   ├── postgres/Dockerfile
│   └── traefik/config.yml
│
└── 📁 bin/                           # Quick launch scripts
    ├── up         # Start all services
    ├── down       # Stop all services
    ├── logs       # View live logs
    └── sync       # Sync data with remote
```

---

## 🎯 Common Commands

### Start/Stop Services

```bash
# Start all services (builds if needed)
docker-compose up -d --build

# View status
docker-compose ps

# View real-time logs
docker-compose logs -f

# Stop all services
docker-compose down

# Remove volumes (WARNING: deletes database!)
docker-compose down -v
```

### Training Models

```bash
# Train all diseases, all provinces (full training)
docker-compose exec app python scripts/train_models.py

# Train specific disease
docker-compose exec app python scripts/train_models.py --diseases Malaria

# Train specific province
docker-compose exec app python scripts/train_models.py --provinces Katanga

# Train with custom epochs
docker-compose exec app python scripts/train_models.py --epochs 80
```

### Training Reset (Clean Slate)

⚠️ **Warning**: This deletes all trained models, predictions, and training history.

```bash
# Reset everything (models, predictions, task history)
docker-compose exec app python scripts/reset_training.py

# After reset, the dashboard will show 0 statistics
# Launch a new training to rebuild predictions
docker-compose exec app python scripts/train_models.py
```

### Data Management

```bash
# Check data quality
docker-compose exec app python scripts/diagnose_data.py --verbose

# Ingest new WHO data
docker-compose exec app python scripts/ingest_hdx.py

# Download climate data
docker-compose exec app python scripts/climate_data_fetcher.py
```

### Database Access

```bash
# Enter Django shell
docker-compose exec app python manage.py shell

# Create superuser (admin)
docker-compose exec app python manage.py createsuperuser

# Run migrations
docker-compose exec app python manage.py migrate

# Create backup
docker-compose exec postgres pg_dump -U postgres health_drc > backup.sql
```

---

## 📈 Performance Metrics

### Last Training Results (March 12, 2026)

| Disease | Tier | R² | Reliability | Propagation |
|---------|------|-----|-------------|------------|
| Malaria | 1 | 0.997 | 98% | +2.5% |
| Cholera | 2 | 0.994 | 97% | +1.2% |
| TB | 3 | 0.993 | 96% | -0.8% |
| HIV/AIDS | 3 | 0.999 | 99% | +0.1% |

### Geographic Coverage

- **Health Zones**: 519/519 (100%)
- **Provinces**: 26/26 (100%)
- **Predictions/Month**: 2,056

---

## 🌍 API Endpoints

```bash
# Global statistics
curl http://localhost:8000/api/stats.json

# Predictions by disease
curl http://localhost:8000/api/predictions/?disease=Malaria

# Clinical data history
curl http://localhost:8000/api/clinical/?zone=Kambove&start=2020-01-01

# System alerts
curl http://localhost:8000/api/alerts/
```

---

<a name="-version-française"></a>
# 🇫🇷 Version Française

## 📊 Vue d'Ensemble du Projet

**HealthDRC Intelligence** est une plateforme avancée de prédiction spatio-temporelle pour la surveillance épidémiologique en République Démocratique du Congo. Elle utilise une **Architecture Multi-Tier de Deep Learning** et une **Intégration Climatique** pour prévoir l'évolution des épidémies (Paludisme, Choléra, Tuberculose, VIH/SIDA) à travers les **519 Zones de Santé**.

### 🎯 Caractéristiques Principales

✅ **Architecture Multi-Tier Adaptative** — Sélectionne automatiquement le meilleur modèle selon les données disponibles
✅ **Intégration Climatique** — Température, précipitations, humidité intégrées à toutes les prédictions
✅ **Couverture Géographique 100%** — 519 zones de santé, 26 provinces, tout le territoire
✅ **Dashboard Temps Réel** — Heatmaps interactives, graphiques, analytics
✅ **Taux de Propagation Intelligent** — Détecte la vraie propagation épidémique via baselines 6 mois
✅ **Containerisé Docker** — Système complet fonctionne dans Docker Compose
✅ **Intelligence NeuralCore** — Explique les décisions IA en langage simple
✅ **Gestion des Tâches** — Arrêter les entraînements/ingestions longs en un clic
✅ **Reset d'Entraînement** — Fonctionnalité de remise à zéro complète pour recommencer  

---

## 🚀 Démarrage Rapide avec Docker

### Prérequis
- Docker & Docker-Compose (v2.0+) installés
- 8GB RAM minimum
- 20GB espace disque

### Lancer en 3 Étapes

```bash
# 1️⃣ Cloner le projet
git clone https://github.com/YannickMagayane/health-drc-intelligence.git
cd health-drc-intelligence

# 2️⃣ Démarrer tous les services avec Docker Compose
docker-compose up -d --build

# 3️⃣ Accéder au dashboard
# Ouvrir: http://localhost:8000
# Login: admin / admin_2026
```

### Vérifier le Statut

```bash
# Voir les conteneurs actifs
docker-compose ps

# Voir les logs
docker-compose logs -f app

# Arrêter tous les services
docker-compose down
```

---

## 🧠 Architecture: Le Système 4-Tiers

Le système sélectionne intelligemment la meilleure architecture selon les données disponibles:

### 📊 Logique de Sélection des Tiers

```
Analyse Zone:
├─ N_mois ≥ 16 ET N_zones ≥ 2
│  └─> TIER 1: ConvLSTM2D (Spatio-Temporel)
│
├─ N_mois ≥ 8 ET Données Climat Disponibles
│  └─> TIER 2: CNN-LSTM (Patterns + Climat)
│
├─ N_mois ≥ 4
│  └─> TIER 3: Bi-LSTM (Temporel Pur)
│
└─ N_mois < 4
   └─> TIER 4: Persistence (Fallback Statistique)
```

### **TIER 1 — ConvLSTM2D** 🌐 (Spatio-Temporel)

**Activation**: ≥16 mois + ≥2 zones de santé

**Architecture**:
```
Entrée: Matrice 4D (batch, timesteps, zones, features)
        Exemple: (32, 4 mois, 30 zones, 4 features)
        Features: [cases, température, précipitations, humidité]

Couche 1: ConvLSTM2D(32 filtres, kernel=3×1, padding='same')
          └─> Capture patterns spatiaux entre zones
          └─> Comprend dépendances temporelles
          └─> Sortie: (batch, 4, 30, 32 filtres)

Couche 2: ConvLSTM2D(64 filtres, padding='same')
          └─> Extraction patterns plus profonde
          └─> Sortie: (batch, 4, 30, 64 filtres)

Couche 3: Flatten → Dense(128) → Dropout(0.3)
Couche 4: Dense(64) → ReLU
Couche 5: Dense(1) → Prédiction

Paramètres: ~500K poids
Temps d'entraînement: 80 epochs × ~5 min = ~6.7 heures par zone
```

**Forces**:
- Détecte patterns de propagation zone→zone
- Capture clustering spatial des maladies
- Meilleure performance: R² = 0.95-1.0

**Exemple**: Paludisme en Haut-Katanga (30 zones × 180 mois)

---

### **TIER 2 — CNN-LSTM** 🌦️ (Patterns + Climat)

**Activation**: ≥8 mois + Données climat disponibles

**Architecture**:
```
Features d'Entrée: Vecteur 6D
  [cases, sin(mois), cos(mois), température, précipitations, humidité]

Couche 1: Conv1D(32 filtres, kernel_size=3)
          └─> Extrait patterns temporels
          └─> Sortie: (batch, timesteps, 32)

Couche 2: LSTM(64 unités, return_sequences=True)
          └─> Memory court-terme/long-terme
          └─> Capture cycles saisonniers
          └─> Sortie: (batch, timesteps, 64)

Couche 3: LSTM(32 unités, return_sequences=False)
          └─> Intégration temporelle finale
          └─> Sortie: (batch, 32)

Couche 4: Dense(64) → ReLU → Dropout(0.3)
Couche 5: Dense(1) → Prédiction

Paramètres: ~180K poids
Temps d'entraînement: 100 epochs × ~3 min = ~5 heures par zone
```

**Forces**:
- Intègre climat explicitement comme features
- Excellent pour maladies saisonnières
- Bon pour densité données modérée

**Exemple**: Choléra avec données climatiques

---

### **TIER 3 — Bi-LSTM** ⏰ (Temporel)

**Activation**: ≥4 mois

**Architecture**:
```
Entrée: Série temporelle univariée (cases uniquement)

Couche 1: Bidirectional LSTM(64 unités)
          └─> LSTM Forward: passé → futur
          └─> LSTM Backward: futur → passé
          └─> Capture dépendances bidirectionnelles
          └─> Sortie: (batch, 128) [64 forward + 64 backward]

Couche 2: Dropout(0.3) → Dense(64) → ReLU
Couche 3: Dense(1) → Prédiction

Paramètres: ~45K poids
Temps d'entraînement: 60 epochs × ~1 min = ~1 heure par zone
```

**Forces**:
- Comprend contexte passé ET futur
- Efficace pour patterns temporels non-linéaires
- Coût computationnel réduit vs Tier 1-2

**Exemple**: Tuberculose, VIH/SIDA

---

### **TIER 4 — Persistence** 📊 (Fallback Statistique)

**Activation**: <4 mois

**Logique**:
```
baseline = dernière_observation
facteur_climat = 1.0

SI (20°C ≤ temp ≤ 28°C ET humidité > 60%):
    facteur_climat = 1.3  # +30% pour conditions favorables
SINON SI (15°C ≤ temp ≤ 32°C):
    facteur_climat = 1.1  # +10% pour conditions acceptables

prédiction = baseline × facteur_climat

Paramètres: 0 (baseline statistique)
Temps d'entraînement: Instant
```

**Forces**:
- Garantit couverture 100% géographique
- Pas de risque sur-apprentissage
- Baseline robuste

**Exemple**: Nouvelles zones avec historique minimal

---

## 🎓 Entraînement Détaillé

### Configuration des Epochs

Un **epoch** est une passe complète sur toutes les données d'entraînement. Voici la configuration:

```yaml
# Epochs par Tier (optimisés pour convergence)
Tier 1 (ConvLSTM):    80 epochs   # Modèle complexe, besoin plus itérations
Tier 2 (CNN-LSTM):   100 epochs   # Features climat nécessite apprentissage thorough
Tier 3 (Bi-LSTM):     60 epochs   # Plus simple, converge plus vite
Tier 4 (Persistence): 0 epochs    # Pas d'entraînement (statistique seulement)

# Temps entraînement total: ~52 heures pour couverture RDC (519 zones)
```

### Pipeline d'Entraînement

```bash
# Lancer entraînement avec Docker (une commande tout-en-un)
docker-compose exec app python scripts/train_models.py

# Avec options:
docker-compose exec app python scripts/train_models.py \
  --diseases "Paludisme,Choléra" \
  --provinces all \
  --epochs auto \
  --use-climate yes
```

### Que se passe-t-il Pendant l'Entraînement

1. **Chargement Données** (5 min): Fetch données cliniques + climat
2. **Prétraitement** (10 min): Normalisation, remplissage gaps, détection outliers
3. **Sélection Tier** (1 min): Analyse N_mois, N_zones → choix Tier
4. **Construction Modèle** (2 min): Compile ConvLSTM/CNN-LSTM/Bi-LSTM
5. **Boucle Entraînement** (varie par Tier):
   - Tier 1: 80 epochs × ~5 min/epoch = 6.7 heures
   - Tier 2: 100 epochs × ~3 min/epoch = 5 heures
   - Tier 3: 60 epochs × ~1 min/epoch = 1 heure
6. **Évaluation** (5 min): Calcule R², MAE, Taux Propagation
7. **Sauvegarde** (2 min): Stock modèles .keras + métriques en DB

**Total par province**: ~2-4 heures (parallélisé sur 26 provinces)

---

## 📊 Utilisation du Dashboard

Accédez au dashboard à `http://localhost:8000/analysis/disease/`

### Vues Disponibles

```
Graphiques Dashboard:
├─ Graphique 1: Top 10 Zones Affectées
│  └─ Affiche prédictions cases + tier modèle utilisé
│
├─ Graphique 2: Performance Modèles
│  └─ R² et % Fiabilité par Tier
│
├─ Graphique 3: Évolution Temporelle
│  └─ Historique (2014+) + prédictions récentes
│
├─ Graphique 4: Distribution Tiers
│  └─ Combien prédictions utilisent chaque Tier
│
├─ Graphique 5: Impact Climatique
│  └─ Corrélation Pearson (temp/precip vs cases)
│
└─ Heatmap: Distribution Géographique
   └─ Intensité couleur par cases + boost climat
```

### Gestion des Tâches

Accédez à l'interface de gestion des tâches sur `http://localhost:8000/tasks/`

- **Visualiser les tâches** en cours/complétées avec progression en temps réel
- **Arrêter les tâches longues** (entraînement/ingestion) d'un simple clic
- **Rafraîchissement automatique** toutes les 3 secondes pour les tâches actives
- **Consulter les logs** (sortie et erreurs) de n'importe quelle tâche

```
Types de Tâches:
├─ 🧠 Entraînement des Modèles (training)
├─ 📥 Ingestion des Données HDX (sync données)
└─ 📦 Ingestion Données Climatiques (climat)
```

### Filtres

```bash
# Par maladie
?disease=Paludisme         # ou Choléra, Tuberculose, VIH_SIDA, all

# Par province
?province=Haut-Katanga     # ou l'une des 26 provinces, all

# Combiné
?disease=Choléra&province=Kinshasa
```

---

## 🔧 Structure Complète Projet

```
health-drc-intelligence/
├── README.md                         # Ce fichier (Quick start + Vue d'ensemble)
├── README_MAIN.md                    # Documentation technique complète
├── .gitignore                        # Règles ignore Git (exclut models, data, .docx)
├── docker-compose.yml                # Orchestration services Docker ⭐
├── requirements.txt                  # Dépendances Python
├── manage.py                         # Management Django
│
├── 📁 core/                          # Configuration Django
│   ├── settings.py                   # Paramètres globaux, config DB
│   ├── urls.py                       # Patterns URL principaux
│   ├── asgi.py / wsgi.py             # Points d'entrée ASGI/WSGI
│
├── 📁 analytics/                     # Application IA & Dashboard
│   ├── views.py                      # Vues Dashboard + logique API
│   ├── models.py                     # Modèles DB (Prediction, ClinicalData)
│   ├── intelligence.py               # Moteur explication NeuralCore
│   ├── urls.py                       # Endpoints API
│   ├── migrations/                   # Migrations DB
│   ├── management/commands/          # Commandes Django custom
│   ├── templates/analytics/
│   │   ├── dashboard.html            # UI Dashboard principal
│   │   ├── disease_detail.html
│   │   └── province_analysis.html
│   └── static/css/ & js/             # Styles & scripts
│
├── 📁 scripts/                       # Entraînement & Pipeline Données
│   ├── train_models.py               # ⭐ MOTEUR ENTRAÎNEMENT PRINCIPAL
│   ├── ingest_hdx.py                 # Import données HDX
│   ├── seed_all_zones.py             # Setup géographique
│   ├── diagnose_data.py              # Vérifications qualité données
│   └── climate_data_fetcher.py       # Téléchargement données climat
│
├── 📁 data/                          # Stockage données local (GITIGNORED)
│   ├── raw_hdx/                      # Données brutes OMS
│   ├── processed/                    # Données nettoyées
│   ├── climat/
│   │   ├── rdc_climate_monthly.pkl
│   │   └── provinces/                # Climat par province
│   └── geospatial/                   # Shapefiles, GeoJSON
│
├── 📁 models/                        # Modèles entraînés (GITIGNORED)
│   ├── convlstm_*.keras              # Modèles Tier 1
│   ├── cnnlstm_*.keras               # Modèles Tier 2
│   ├── bilstm_*.keras                # Modèles Tier 3
│   └── metadata.json
│
├── 📁 docs/                          # Documentation détaillée
│   ├── ARCHITECTURE_MODELES_COMPLET.md
│   ├── DOCUMENTATION_TRAIN_TEST.md
│   ├── RAPPORT_Paludisme_Cholera_DataComparison.md
│   └── ... (7 documents supplémentaires)
│
├── 📁 reports/                       # Rapports Word (GITIGNORED)
│   ├── RAPPORT_PALUDISME_CHOLERA_COMPLET.docx
│   └── ... (4 docs Word supplémentaires)
│
├── 📁 compose/                       # Configs services Docker
│   ├── app/Dockerfile
│   ├── postgres/Dockerfile
│   └── traefik/config.yml
│
└── 📁 bin/                           # Scripts lancement rapide
    ├── up         # Démarrer tous services
    ├── down       # Arrêter tous services
    ├── logs       # Voir logs en direct
    └── sync       # Sync données avec remote
```

---

## 🎯 Commandes Courantes

### Démarrer/Arrêter Services

```bash
# Démarrer tous les services (build si nécessaire)
docker-compose up -d --build

# Voir statut
docker-compose ps

# Voir logs en direct
docker-compose logs -f

# Arrêter tous services
docker-compose down

# Supprimer volumes (ATTENTION: supprime DB!)
docker-compose down -v
```

### Entraîner Modèles

```bash
# Entraîner toutes maladies, toutes provinces (entraînement complet)
docker-compose exec app python scripts/train_models.py

# Entraîner maladie spécifique
docker-compose exec app python scripts/train_models.py --diseases Paludisme

# Entraîner province spécifique
docker-compose exec app python scripts/train_models.py --provinces Katanga

# Entraîner avec epochs custom
docker-compose exec app python scripts/train_models.py --epochs 80
```

### Reset d'Entraînement (Remise à Zéro)

⚠️ **Attention**: Supprime tous les modèles entraînés, prédictions et historique.

```bash
# Réinitialiser complètement (modèles, prédictions, historique)
docker-compose exec app python scripts/reset_training.py

# Après le reset, le dashboard affiche 0 statistiques
# Lancez un nouvel entraînement pour reconstruire les prédictions
docker-compose exec app python scripts/train_models.py
```

### Gestion Données

```bash
# Vérifier qualité données
docker-compose exec app python scripts/diagnose_data.py --verbose

# Ingérer nouvelles données OMS
docker-compose exec app python scripts/ingest_hdx.py

# Télécharger données climat
docker-compose exec app python scripts/climate_data_fetcher.py
```

### Accès Base Données

```bash
# Entrer Django shell
docker-compose exec app python manage.py shell

# Créer utilisateur admin
docker-compose exec app python manage.py createsuperuser

# Lancer migrations
docker-compose exec app python manage.py migrate

# Créer backup
docker-compose exec postgres pg_dump -U postgres health_drc > backup.sql
```

---

## 📈 Métriques de Performance

### Résultats Dernier Entraînement (12 Mars 2026)

| Maladie | Tier | R² | Fiabilité | Propagation |
|---------|------|-----|-----------|------------|
| Paludisme | 1 | 0.997 | 98% | +2.5% |
| Choléra | 2 | 0.994 | 97% | +1.2% |
| TB | 3 | 0.993 | 96% | -0.8% |
| VIH/SIDA | 3 | 0.999 | 99% | +0.1% |

### Couverture Géographique

- **Zones de Santé**: 519/519 (100%)
- **Provinces**: 26/26 (100%)
- **Prédictions/Mois**: 2,056

---

## 🌍 Endpoints API

```bash
# Statistiques globales
curl http://localhost:8000/api/stats.json

# Prédictions par maladie
curl http://localhost:8000/api/predictions/?disease=Paludisme

# Historique données cliniques
curl http://localhost:8000/api/clinical/?zone=Kambove&start=2020-01-01

# Alertes système
curl http://localhost:8000/api/alerts/
```

---

## 📚 Documentation Additionnelle

Pour plus de détails sur l'architecture, l'entraînement et l'analyse:

- **[README_MAIN.md](README_MAIN.md)** — Documentation technique complète
- **[docs/ARCHITECTURE_MODELES_COMPLET.md](docs/ARCHITECTURE_MODELES_COMPLET.md)** — Architecture détaillée ConvLSTM/CNN-LSTM
- **[docs/DOCUMENTATION_TRAIN_TEST.md](docs/DOCUMENTATION_TRAIN_TEST.md)** — Pipeline d'entraînement
- **[docs/ANALYSE_GRAPHIQUES_PROVINCES.md](docs/ANALYSE_GRAPHIQUES_PROVINCES.md)** — Analyse provinciale
- **[docs/RAPPORT_Paludisme_Cholera_DataComparison.md](docs/RAPPORT_Paludisme_Cholera_DataComparison.md)** — Comparaisons maladies
- **[docs/RAPPORT_TOUTES_MALADIES_DataComparison.md](docs/RAPPORT_TOUTES_MALADIES_DataComparison.md)** — Vue globale 4 maladies

---

## 🤝 Support & Contribution

- **Issues/Bugs**: Ouvrir GitHub Issue ou Discussion
- **Rapports Word**: Voir dossier [reports/](reports/)

---

## 👨‍💻 Author

**Yannick Magayane**
- 🏛️ HealthDRC Intelligence Creator
- 🌍 Spatio-Temporal Epidemic Forecasting Specialist
- 📧 [GitHub](https://github.com/YannickMagayane)

---

**v3.0.0** | Production | Mars 2026  
**Dernière mise à jour**: 12 Mars 2026  
**Status**: ✅ Production
