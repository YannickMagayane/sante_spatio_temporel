# HealthDRC: Spatio-Temporal Health Intelligence Platform 🛰️🏥

**HealthDRC** is a cutting-edge spatio-temporal analysis system designed for the Democratic Republic of Congo (DRC). It leverages Deep Learning to monitor, analyze, and predict epidemic outbreaks across the 519 Health Zones of the country.

---

## 🇫🇷 Version Française

### 🎯 Objectifs
HealthDRC vise à fournir une surveillance épidémiologique en temps réel en intégrant diverses sources de données (HDX, OMS, reliefweb) et en utilisant des modèles d'IA avancés pour anticiper les foyers de contagion.

### 🚀 Fonctionnalités Clés
- **Cartographie Heatmap interactive** : Visualisation spatiale des menaces sanitaires sur l'ensemble du territoire RDC.
- **Moteur IA Prédictif** : Utilisation de modèles **ConvLSTM**, **CNN** et **LSTM** pour prédire les cas futurs avec une haute fiabilité.
- **Ingestion Automatisée** : Pipeline capable de traiter des fichiers CSV, XLSX, ZIP et GeoJSON provenant de sources internationales.
- **Tableau de Bord Premium** : Interface moderne basée sur TailwindCSS, HTMX et Alpine.js pour une expérience utilisateur fluide.

### 🛠️ Architecture Technique
- **Backend** : Django 5.1 (Python 3.12).
- **IA/ML** : TensorFlow/Keras pour les modèles spatio-temporels.
- **Infrastructure** : Docker & Docker-Compose.
- **Reverse Proxy** : Traefik v3.1.
- **Base de données** : PostgreSQL 16.

### 📦 Installation Rapide
```bash
# Cloner le projet
git clone https://github.com/YannickMagayane/sante_spatio_temporel
cd sante_spatio_temporel

# Lancer avec Docker
docker-compose up -d --build

# Initialiser les données
docker-compose exec app python scripts/seed_all_zones.py
docker-compose exec app python scripts/ingest_hdx.py
docker-compose exec app python scripts/train_models.py
```

---

## 🇺🇸 English Version

### 🎯 Objectives
HealthDRC aims to provide real-time epidemiological surveillance by integrating various data sources (HDX, WHO, reliefweb) and using advanced AI models to anticipate contagion hotspots.

### 🚀 Key Features
- **Interactive Heatmap Mapping**: Spatial visualization of health threats across the entire DRC territory.
- **Predictive AI Engine**: Leveraging **ConvLSTM**, **CNN**, and **LSTM** models to predict future cases with high reliability.
- **Automated Ingestion**: Pipeline capable of processing CSV, XLSX, ZIP, and GeoJSON files from international sources.
- **Premium Dashboard**: Modern interface based on TailwindCSS, HTMX, and Alpine.js for a seamless user experience.

### 🛠️ Technical Architecture
- **Backend**: Django 5.1 (Python 3.12).
- **AI/ML**: TensorFlow/Keras for spatio-temporal modeling.
- **Infrastructure**: Docker & Docker-Compose.
- **Reverse Proxy**: Traefik v3.1.
- **Database**: PostgreSQL 16.

### 📦 Quick Setup
```bash
# Clone the project
git clone <repository_url>
cd health

# Launch with Docker
docker-compose up -d --build

# Initialize data
docker-compose exec app python scripts/seed_all_zones.py
docker-compose exec app python scripts/ingest_hdx.py
docker-compose exec app python scripts/train_models.py
```

---
**Développé par Yannick Magayane pour la résilience sanitaire de la RDC.**
