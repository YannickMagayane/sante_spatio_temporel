# 🏥 HealthDRC Intelligence v2.0
**Analyse Spatio-Temporelle & Surveillance Épidémiologique en RDC**

---

## 🏗️ État Actuel : Ce qui marche & ce qui ne marche pas

### ✅ Ce qui fonctionne (OK)
- **Modèles Hybrides (ConvLSTM-Spatio)** : L'IA est maintenant robuste et corrélée temporellement et spatialement.
- **Entraînement Automatisé** : Le "Data Train" est prêt et se lance automatiquement toutes les **24 heures**.
- **Visualisation Intelligente** :
    - **Global** : HeatMap (Carte de chaleur) fluide sur toute la RDC.
    - **Local** : Zoom automatique et isolation des zones de santé lors du choix d'une province.
- **Dynamisme des Données** : Les indicateurs (Age -15 ans, Ruralité, Reliabilité) sont calculés en temps réel depuis la base de données.

### ⚠️ Ce qui est en cours / Nécessite attention
- **Accès Dashboard (Bad Gateway)** : Traefik rencontre parfois des difficultés à joindre le container applicatif lors des redémarrages massifs (corrigé via simplification des labels).
- **Données Climatiques ERA5** : Nécessite l'acceptation manuelle des licences sur le site de Copernicus pour un téléchargement complet sans erreurs 403.

---

## 📂 Architecture & Organisation

```text
health/
├── bin/                              # Commandes de contrôle (up, down, logs, sync)
├── analytics/                        # Moteur Django & Intelligence Artificielle
├── scripts/                          # Pipelines de données (Train, Scraping, Sync)
├── templates/                        # Interface Dashboard Premium
├── data/                             # Stockage des données (HDX, Climat, Sanitaires)
└── compose/                          # Configuration Docker (App, Traefik, DB)
```

---

## 🚀 Guide Rapide de Lancement

1. **Lancer le Pilier** :
   ```bash
   ./bin/up
   ```
2. **Accéder au Dashboard** :
   Ouvrez votre navigateur sur `http://health.localhost` ou `http://localhost:8000`.

3. **Forcer une Synchronisation Rapide** :
   ```bash
   docker-compose exec app python3 scripts/train_models.py
   ```

---

## 📊 Les Modèles : LSTM vs CNN

- **LSTM (Temporel)** : Analyse les cycles passés pour prédire **QUAND** un pic épidémique arrivera.
- **CNN (Spatial)** : Analyse la carte pour prédire **OÙ** la maladie va se déplacer.
- **ConvLSTM (Hybride)** : Le "Cœur" du système, capturant les deux dimensions pour une précision maximale.

---

## 🔄 Automatisation 24h
L'orchestrateur (`scripts/orchestrator.py`) assure la fraîcheur des données :
- **T+0h** : Synchronisation HDX.
- **T+6h** : MAJ Climat.
- **T+12h** : Entraînement IA.
- **T+24h** : Répétition.

---
*Développé pour l'excellence sanitaire en RDC par Yannick Magayane.*
