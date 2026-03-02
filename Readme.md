# Analyse prédictive de la dynamique spatio-temporelle des maladies infectieuses en milieu urbain en RDC

**Approche comparative : modèles LSTM vs CNN intégrant les réalités épidémiologiques locales**

---

## 📋 Description du projet

Ce projet vise à développer des modèles de deep learning (LSTM, CNN, et hybrides) pour **prédire la propagation spatio-temporelle** de maladies infectieuses (paludisme, choléra, rougeole, mpox, Ebola) sur l'ensemble des **26 provinces de la RDC**.

Les modèles intègrent des données :
- **Environnementales** (climat via ERA5-Land)
- **Démographiques** (densité, infrastructures via OSM)
- **Sanitaires** (cas historiques via DPS/PNLP/INRB)
- **Temps réel** (alertes via web scraping Twitter/X et OMS)

pour produire des **cartes de risque interactives** et des **alertes précoces**.

## 🚀 Objectifs

1. Collecter et structurer des données hétérogènes (climat, géospatiales, épidémio, web scraping).
2. Développer des modèles spatio-temporels comparatifs :
   - **LSTM pur** : séquences temporelles par zone de santé
   - **CNN pur** : images spatio-temporelles multi-canaux
   - **Modèles hybrides** : ConvLSTM, CNN+LSTM
3. Évaluer les performances (RMSE, MAE, AUC) et interpréter les facteurs de risque (SHAP).
4. Visualiser les prédictions sur des **cartes interactives** (Folium/Leaflet).
5. Automatiser la veille sanitaire via **web scraping** (Twitter/X, OMS).

---

## 📂 Structure du projet

```text
health/
│
├── bin/                              # Scripts de gestion Docker
│   ├── up                            # Lancer tous les conteneurs
│   ├── down                          # Arrêter les conteneurs
│   ├── logs                          # Afficher les logs en temps réel
│   └── terminal                      # Accès terminal dans un conteneur
│
├── compose/                          # Configuration Docker
│   ├── app/
│   │   └── Dockerfile                # Image Python 3.12 + GDAL + deps
│   ├── cle_api/                      # Clé API Copernicus (non versionnée)
│   │   └── .cdsapirc                 # Clé API Copernicus ERA5
│   └── traefik/                      # Configuration du reverse proxy Traefik
│
├── config/                           # Fichiers de configuration
│   └── preprocess.yaml               # Paramètres de prétraitement (villes, maladies, etc.)
│
├── data/                             # Données brutes et traitées
│   ├── climat/                       # Données ERA5-Land (NetCDF/CSV)
│   ├── geospatial/                   # Shapefiles OSM, limites administratives
│   ├── sanitaires/                   # Données de cas par ville (CSV)
│   ├── scraped/                      # Données issues du web scraping
│   └── processed/                    # Données fusionnées prêtes pour ML
│
├── scripts/                          # Scripts Python exécutables
│   ├── download_climate.py           # Téléchargement ERA5 (API Copernicus)
│   ├── download_geodata.py           # Téléchargement OSM / HDX
│   ├── geo_reference.py              # Base de données géo des 26 provinces & géocodage
│   ├── web_scraping.py               # Scraping Twitter (X), OMS & ReliefWeb avec géocodage précis
│   ├── preprocess.py                 # Nettoyage, fusion, séquences LSTM, images CNN
│   ├── train_models.py               # Entraînement LSTM / CNN / ConvLSTM
│   ├── evaluate.py                   # Métriques (RMSE, MAE, AUC, F1) et SHAP
│   ├── generate_dashboard_data.py    # Génération des API JSON pour le dashboard
│   └── visualize.py                  # Cartes Folium, SHAP plots, tendances
│
├── models/                           # Modèles sauvegardés (.h5, .pkl)
├── reports/                          # Métriques JSON, graphiques, cartes HTML
├── frontend/                         # Dashboard interactif (Leaflet + Alpine.js)
│   └── index.html                    # Tableau de bord avec carte, filtres et alertes
│
├── .env                              # Variables d'environnement locales
├── .gitignore                        # Fichiers exclus du versionnement
├── docker-compose.yml                # Orchestration (Traefik + App)
├── requirements.txt                  # Dépendances Python
└── Readme.md                         # Ce fichier
```

---

## 🔧 Installation

### 🐳 Via Docker Compose (Recommandé)

Le projet tourne entièrement dans **Docker** avec **Traefik** comme reverse proxy. Aucune installation locale n'est nécessaire.

1. **Configurer les clés API** (voir section [Configuration des clés API](#-configuration-des-clés-api))

2. **Lancer l'application** :
   ```bash
   ./bin/up
   ```
   > Ceci construit l'image Docker et lance les services (App + Traefik).
   > L'application est accessible sur `http://health.localhost`.

3. **Voir les logs** :
   ```bash
   ./bin/logs
   ```

4. **Arrêter l'application** :
   ```bash
   ./bin/down
   ```

### 🐍 Via Environnement Virtuel (Alternative locale)

1. **Prérequis** : Python 3.9+, GDAL installé, compte Copernicus.
2. **Cloner et installer** :
   ```bash
   git clone https://github.com/YannickMagayane/sante_spatio_temporel.git
   cd sante_spatio_temporel
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

---

## 🔑 Configuration des clés API

### 1. API Copernicus (données climatiques ERA5)

La clé est déjà configurée dans `compose/cle_api/.cdsapirc` et montée automatiquement dans le conteneur Docker.

> 📌 **Pour obtenir votre propre clé :**
> 1. Inscrivez-vous sur [Copernicus CDS](https://cds.climate.copernicus.eu/user/register)
> 2. Une fois connecté, allez sur [CDSAPI setup](https://cds.climate.copernicus.eu/how-to-api)
> 3. Copiez la clé affichée et remplacez-la dans `compose/cle_api/.cdsapirc`

### 2. Twitter/X (web scraping)

**Aucune clé API nécessaire.** Le projet utilise `ntscraper` pour scraper Twitter/X sans passer par l'API officielle. Le scraping fonctionne directement via des instances Nitter publiques.

### 3. Variables d'environnement

Le fichier `.env` à la racine est déjà créé :
```env
DEBUG=1
PYTHONUNBUFFERED=1

```

---

## 📥 Acquisition des données

### 1. Données climatiques (ERA5-Land)

Le script `scripts/download_climate.py` télécharge les données mensuelles via l'**API Copernicus**.

```bash
python scripts/download_climate.py \
   --start 2020 --end 2025 \
   --regions all \
   --output data/climat/
```

**Note** : Le système couvre désormais les 26 provinces de la RDC.

**Variables téléchargées** : température à 2m, précipitations totales, humidité relative.
**Format** : NetCDF4 → convertis en CSV par le prétraitement.

### 2. Données géospatiales (OpenStreetMap / HDX)

Le script `scripts/download_geodata.py` récupère :
- Limites administratives et zones de santé
- Routes, bâtiments, points d'eau
- Densité de population

**Ressources** :
- Zones de santé : [HDX RDC](https://data.humdata.org/dataset/zone-de-sante-beni-nord-kivu-rdc)
- Limites provinciales : [HDX](https://data.humdata.org/group/cod)

### 3. Données sanitaires (cas de maladies)

**⚠️ Important :** ces données ne sont pas publiques. Contactez officiellement :
- **Divisions Provinciales de la Santé (DPS)** de chaque province cible
- **Programme National de Lutte contre le Paludisme (PNLP)**
- **Institut National de Recherche Biomédicale (INRB)** — [site web](http://www.inrb.cd/)

**Demandez** : séries temporelles mensuelles 2021–2025 par zone de santé, avec cas confirmés/suspects/décès pour chaque maladie.

Placez les fichiers CSV dans `data/sanitaires/` (ex: `kinshasa_palu_2021_2025.csv`).

### 4. 🌐 Web Scraping (Twitter/X & OMS)

Le script `scripts/web_scraping.py` automatise la collecte de données en temps réel pour compléter les sources officielles :

**Plateformes scrapées** :
| Plateforme | Méthode | Type de données |
|---|---|---|
| **Twitter (X)** | ntscraper / API v2 | Tweets d'alertes sanitaires, signalements citoyens |
| **OMS (WHO)** | BeautifulSoup / requests | Rapports de situation, bulletins épidémiologiques |
| **ReliefWeb** | API REST | Alertes humanitaires, rapports de terrain |

**Mots-clés utilisés** :

| Catégorie | Exemples de mots-clés |
|---|---|
| Maladies | `paludisme`, `choléra`, `rougeole`, `mpox`, `ebola` |
| Alertes | `épidémie`, `éclosion`, `cas suspect`, `alerte sanitaire` |
| Géographie | **26 Provinces de la RDC** (Kinshasa, Nord-Kivu, Haut-Katanga, etc.) |
| Précision | Localisation automatique via `geo_reference.py` |

**Utilisation** :
```bash
# Scraper Twitter pour des alertes choléra à Kinshasa
python scripts/web_scraping.py --platform twitter --keywords "choléra épidémie" --city Kinshasa

# Scraper l'OMS pour les bulletins RDC
python scripts/web_scraping.py --platform oms --keywords "RDC épidémie" --city "Nord-Kivu"
```

> Les données scrapées sont nettoyées, dédupliquées et fusionnées avec les sources officielles lors du prétraitement.

---

## 🔄 Prétraitement des données

Le script `scripts/preprocess.py` réalise :

1. **Nettoyage** : gestion des valeurs manquantes, harmonisation des dates.
2. **Agrégation spatiale** : grille 500m × 500m ou zones de santé comme unités spatiales.
3. **Séquences temporelles** (pour LSTM) : fenêtres de longueur T (ex: 12 mois) avec variables climatiques + cas passés.
4. **Images spatio-temporelles** (pour CNN) : tenseurs `(time, height, width, channels)` où chaque canal = une variable.
5. **Intégration web scraping** : fusion des signaux Twitter/OMS avec les données structurées.

```bash
python scripts/preprocess.py --config config/preprocess.yaml
```

---

## 🧠 Modèles de deep learning

Trois architectures sont implémentées dans `scripts/train_models.py` :

### LSTM pur
- **Entrée** : séquence temporelle pour une zone de santé.
- **Architecture** : 2 couches LSTM (128 → 64 unités) + couches denses.
- **Sortie** : prédiction du nombre de cas pour les 1–4 prochaines semaines.
- **Usage** : capture des tendances temporelles (saisonnalité, pics épidémiques).

### CNN pur
- **Entrée** : images spatio-temporelles multi-canaux (12 mois × variables).
- **Architecture** : Conv2D + BatchNorm + MaxPooling + Flatten + Dense.
- **Sortie** : carte de prédiction de risque par pixel.
- **Usage** : capture des corrélations spatiales (propagation géographique).

### Modèles hybrides (ConvLSTM)
- **Architecture** : couches `ConvLSTM2D` combinant convolutions spatiales et mémoire temporelle.
- **Usage** : modèle le plus complet, capture simultanément espace et temps.

### 🔬 Détails Méthodologiques

L'approche repose sur une comparaison rigoureuse de trois paradigmes de Deep Learning :

1.  **Paradigme Temporel (LSTM)** :
    *   **Concept** : Analyse les cycles saisonniers (ex: pic de paludisme après la saison des pluies).
    *   **Input** : Vecteurs de dimension `(Batch, Time_Steps, Features)`. Les features incluent les anomalies de température, de précipitations et l'historique des cas.
    *   **Force** : Excellente mémoire à long terme des cycles épidémiques.

2.  **Paradigme Spatial (CNN)** :
    *   **Concept** : Traite la RDC comme une "image" où chaque pixel contient des informations climatiques.
    *   **Input** : Tenseurs `(Batch, Height, Width, Channels)`. Les couches de convolution détectent les "foyers" (clusters) et leur diffusion vers les zones voisines.
    *   **Force** : Capture la propagation géographique (ex: progression d'Ebola le long d'un axe routier).

3.  **Paradigme Hybride (ConvLSTM)** :
    *   **Concept** : Le "Gold Standard" de ce projet. Il remplace les multiplications de matrices du LSTM par des convolutions.
    *   **Objectif** : Prédire non seulement **quand** l'épidémie va frapper, mais aussi **vers où** elle va se déplacer.

---

## 📊 Évaluation et interprétation

Le script `scripts/evaluate.py` calcule :

| Métrique | Usage |
|---|---|
| **RMSE** | Erreur quadratique moyenne des prédictions |
| **MAE** | Erreur absolue moyenne |
| **AUC** | Performance de la classification en risque épidémique |
| **F1-score** | Équilibre précision/rappel |
| **SHAP** | Variables les plus influentes (interprétabilité) |

```bash
python scripts/evaluate.py --model lstm --test_data data/processed/test.csv
```

---

## 🌍 Visualisation interactive

### Cartes de risque (Folium/Leaflet)

Le script `scripts/visualize.py` génère des **cartes interactives** :
- Cercles proportionnels aux cas prédits par zone de santé
- Couches de risque colorées (vert → rouge)
- Filtrage par maladie, période et ville

```bash
# Carte de risque pour janvier 2025
python scripts/visualize.py --type maps --period 2025-01

# Analyse SHAP pour le modèle LSTM
python scripts/visualize.py --type shap --model lstm
```

### Tableau de bord (Alpine.js + Leaflet)

Le dossier `frontend/` contient un dashboard interactif (`index.html`) permettant de :
- 🗺️ Choisir une maladie et une semaine sur une carte
- 📊 Afficher les tendances temporelles par ville
- ⚠️ Consulter les alertes générées par le web scraping
- 📈 Comparer les performances des modèles LSTM vs CNN

---

## 🛠️ Stack technique

| Composant | Technologie |
|---|---|
| **Conteneurisation** | Docker + Docker Compose |
| **Reverse proxy** | Traefik v3.1 |
| **Backend** | Python 3.14, Django LTS |
| **Deep learning** | TensorFlow / Keras |
| **Données** | pandas, numpy, xarray, netCDF4 |
| **Géospatial** | geopandas, rasterio, shapely, osmnx |
| **Visualisation** | matplotlib, seaborn, folium, plotly |
| **Web scraping** | BeautifulSoup, ntscraper, requests |
| **API climat** | cdsapi (Copernicus ERA5) |
| **Interprétabilité** | SHAP, ELI5 |
| **Frontend** | Alpine.js, Leaflet, TailwindCSS (CDN) |

---

## 📝 Livrables attendus

1. Un jeu de données structuré (CSV) avec les séries temporelles par province et maladie.
2. Des modèles entraînés et comparés (LSTM vs CNN vs ConvLSTM).
3. Un rapport d'analyse (PDF) détaillant la méthodologie, les résultats et les limites.
4. Des **cartes interactives** de visualisation des prédictions.
5. Un pipeline de **web scraping** pour la veille sanitaire en temps réel.

## 📌 Prochaines étapes

| # | Étape | Statut |
|---|---|---|
| 1 | Structurer le projet et configurer Docker Compose | ✅ Fait |
| 2 | Configurer la clé API Copernicus ERA5 | ✅ Fait |
| 3 | Implémenter les scripts (scraping, climat, modèles, évaluation, visualisation) | ✅ Fait |
| 4 | Construire le dashboard interactif Premium (Leaflet + Alpine.js + HTMX) | ✅ Fait |
| 5 | Étendre la couverture aux 26 provinces de la RDC | ✅ Fait |
| 6 | Implémenter le géocodage de précision pour le scraping | ✅ Fait |
| 7 | Supprimer les données simulées (Zéro-Simulation Policy) | ✅ Fait |
| 8 | Télécharger les données climatiques ERA5 pour toute la RDC | ⏳ À lancer |
| 9 | Acquérir les données sanitaires officielles (DPS, PNLP, INRB) | ⏳ Contacts en cours |
| 10 | Entraîner et comparer les modèles sur l'ensemble du territoire | ⏳ En cours |
| 11 | Rédiger le rapport final et préparer la soutenance | ⏳ Phase finale |

---

## 🌐 Fonctionnement du Web Scraping

Le web scraping est le mécanisme de **veille sanitaire en temps réel** du projet. Il complète les données officielles (souvent publiées avec retard) par des signaux précoces détectés en ligne.

### Principe général

```text
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Twitter/X   │    │   OMS (WHO)  │    │  ReliefWeb   │
│  (ntscraper) │    │(BeautifulSoup│    │  (API REST)  │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       └───────────┬───────┴───────────────────┘
                   ▼
          ┌────────────────┐
          │  web_scraping.py│  ← Mots-clés + Villes
          └────────┬───────┘
                   ▼
          ┌────────────────┐
          │ data/scraped/  │  ← CSV horodatés
          └────────┬───────┘
                   ▼
          ┌────────────────┐
          │  preprocess.py │  ← Fusion avec données officielles
          └────────┬───────┘
                   ▼
          ┌────────────────┐
          │ Modèles (LSTM, │
          │ CNN, ConvLSTM) │
          └────────────────┘
```

### 1. Twitter/X — via `ntscraper` (sans clé API)

**Comment ça marche :**
- `ntscraper` scrape Twitter via des **instances Nitter publiques** (miroirs open-source de Twitter)
- Aucune authentification API nécessaire
- On recherche des tweets contenant nos mots-clés + nom de ville

**Commande :**
```bash
python scripts/web_scraping.py --platform twitter --keywords "choléra épidémie" --city Kinshasa
```

**Ce qu'on récupère :**
- Texte du tweet, date, lien source.
- **Géocodage Précis** : Extraction automatique de la ville et de la province.
- **Coordonnées GPS** : Attribution de `lat/lon` pour chaque alerte via le module `geo_reference`.

### 2. OMS (WHO) — via `BeautifulSoup`

**Comment ça marche :**
- On scrape la page [Disease Outbreak News](https://www.who.int/emergencies/disease-outbreak-news) de l'OMS
- BeautifulSoup parse le HTML pour extraire les titres d'articles
- Filtrage par mots-clés (`RDC`, `Congo`, nom de maladie) et ville

**Commande :**
```bash
python scripts/web_scraping.py --platform oms --keywords "RDC épidémie" --city Beni
```

**Ce qu'on récupère :**
- Titres des bulletins épidémiologiques, dates, liens

### 3. ReliefWeb — via API REST

**Comment ça marche :**
- ReliefWeb (portail humanitaire de l'ONU) offre une **API REST publique** sans authentification
- On requête `api.reliefweb.int/v1/reports` avec nos mots-clés filtrés sur la RDC

**Commande :**
```bash
python scripts/web_scraping.py --platform reliefweb --keywords "ebola" --city Beni
```

**Ce qu'on récupère :**
- Rapports de terrain, alertes humanitaires, titres, dates

### 4. Scraper toutes les plateformes en une fois

```bash
python scripts/web_scraping.py --platform all --keywords "paludisme épidémie" --city Kinshasa
```

### 5. Résultat

Chaque exécution génère un fichier CSV horodaté dans `data/scraped/` :

```text
data/scraped/
├── scraped_20250115_143022.csv
├── scraped_20250116_091510.csv
└── ...
```

Chaque CSV contient les colonnes : `date`, `text`, `source`, `city`, `platform`, `keywords`, `scraped_at`.

Ces données sont ensuite **fusionnées automatiquement** avec les données officielles lors de l'étape de prétraitement (`scripts/preprocess.py`).

## 🧠 Explication des modèles (LSTM vs CNN)

L'intelligence artificielle de ce système repose sur une analyse à deux dimensions : **le temps** et **l'espace**.

### 1. Le LSTM (Long Short-Term Memory) : **L'aspect Temporel**
*   **Rôle** : Analyser les séries chronologiques (le passé).
*   **Fonctionnement** : Il étudie les cycles de données sur plusieurs mois (ex: précipitations et cas de paludisme). 
*   **Objectif** : Apprendre que si de fortes pluies surviennent en octobre, un pic épidémique est probable en novembre. Le LSTM mémorise ces décalages temporels.

### 2. Le CNN (Convolutional Neural Network) : **L'aspect Spatial**
*   **Rôle** : Analyser la géographie et les voisinages (la carte).
*   **Fonctionnement** : Il traite la carte de la RDC comme une "image" de santé multi-couches.
*   **Objectif** : Comprendre que si une épidémie explose dans une province (ex: Nord-Kivu), le risque augmente mathématiquement dans les provinces voisines à cause des mouvements de population. Le CNN détecte des "motifs" de propagation géographique.

### 🚀 Modèle Hybride : **Spatio-Temporel**
En combinant les deux, le système prédit non seulement **quAND** une épidémie risque d'arriver (LSTM), mais aussi précisément **OÙ** elle va se propager (CNN). C'est la synergie de ces deux modèles qui permet une alerte précoce et précise sur l'ensemble du territoire national.

---

## 📞 Contacts

- **Yannick Magayane** :
  - 📱 Tel : +243 979 068 311
  - 📧 Mail : yannickmagayaneyannick@gmail.com

- **Ressources institutionnelles** :
  - **INRB** : [www.inrb.cd](http://www.inrb.cd/)
  - **PNLP** : pnlp@minisanterdc.cd
  - **Copernicus CDS** : [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/)

---

> **Remarque** : Ce README est évolutif. Il sera mis à jour au fil de l'avancement du projet. 🚀
