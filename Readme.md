# 🏥 HealthDRC Intelligence v3.0 (NeuralCore Adaptive)
**Analyse Spatio-Temporelle, Deep Learning & Surveillance Épidémiologique en RDC**

---

## 🏗️ État Actuel : Intelligence Adaptative & Robustesse

### ✅ Architecture NeuralCore (Nouveautés v3.0)
- **Modèles Triple-Tier (Intelligence Adaptative)** : Le pipeline d'entraînement ne "skip" plus aucune zone. Il adapte sa complexité selon la densité des données HDX :
    - **🤖 [BI-LSTM]** : Deep Learning bidirectionnel pour séries temporelles riches ($N \ge 8$).
    - **📈 [LINEAR TREND]** : Régression statistique Scikit-Learn pour séries courtes ($4 \le N < 8$).
    - **📉 [PERSISTENCE]** : Modèle de repli par persistance pour zones à très faible historique.
- **Moteur d'Intelligence v3** : Analyse sémantique des prédictions. L'IA explique désormais *pourquoi* elle a choisi un modèle et quel est le degré de confiance ($R^2$, MAE, RMSE).
- **Dashboard Premium v3** :
    - **Score de Précision IA ($R^2$)** : Affichage transparent de la fiabilité historique.
    - **Alertes de Densité** : Badge automatique `⚠️ Faible Densité` pour les prédictions basées sur peu de points.
    - **Propagation & Gain** : Calcul dynamique du taux de croissance épidémique.

### ⚠️ Précautions Techniques
- **Git & Big Data** : Les dossiers `data/` et `models/` sont strictement exclus du repo via `.gitignore` (exigences GitHub < 100Mo).
- **Données HDX** : Les séries temporelles annuelles sont automatiquement gérées par le fallback linéaire.

---

## 📂 Architecture du Projet

```text
health/
├── bin/                              # Scripts de contrôle (up, down, logs, sync)
├── analytics/                        # Moteur Django, Models & Intelligence (IA)
├── scripts/                          # Pipelines (Train adaptatif, Ingest HDX, Diag)
├── templates/                        # Interface Dashboard ultra-moderne (Tailwind/Alpine)
├── data/                             # [IGNORÉ GIT] Données brutes et extraites (HDX, Geospatial)
└── models/                           # [IGNORÉ GIT] Modèles .keras entraînés
```

---

## 🚀 Guide de Lancement

1. **Démarrer l'infrastructure** :
   ```bash
   ./bin/up
   ```
2. **Lancer l'entraînement intelligent (NeuralCore)** :
   ```bash
   docker-compose exec app python3 scripts/train_models.py
   ```
3. **Accéder au Dashboard** :
   Ouvrez `http://localhost:8000` ou `http://health.localhost`.

---

## 📊 Méthodologie IA : NeuralCore Adaptive

- **Bi-LSTM (Réseaux Récurrents)** : Analyse les tendances passées et la saisonnalité cyclique (Sin/Cos) pour prédire les pics futurs.
- **R² Score Optimization** : Chaque modèle est évalué sur un split temporel strict (pas de shuffle) pour garantir la validité scientifique.
- **Adaptive Windowing** : La fenêtre temporelle glissante s'ajuste dynamiquement à la longueur de la série disponible par province.

---

## 🔄 Pipeline de Synchronisation
L'orchestrateur assure la cohérence des données :
- **Ingestion** : `scripts/ingest_hdx.py` - Injection intelligente des données WHO/HDX.
- **Diagnostic** : `scripts/diag_data.py` - Vérification de la cohérence spatiale.
- **IA** : `scripts/train_models.py` - Cycle complet d'entraînement et prédiction 2026.

---
*Moteur de prédiction spatio-temporelle optimisé pour la République Démocratique du Congo.*
*Développé par Yannick Magayane.*
