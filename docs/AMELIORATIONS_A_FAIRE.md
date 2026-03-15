# 🚀 Améliorations à Faire - Plan Complet

**Date**: 13 mars 2026  
**État actuel**: Thèse 7.5/10 | Article 6.5/10 | Business 8.5/10  
**Objectif**: Thèse 9.2/10 | Article 8.5/10 | Business 9.2/10

---

## 📊 RÉSUMÉ EXÉCUTIF

| Phase | Durée | Impact | Priorité |
|-------|-------|--------|----------|
| **Phase 1: Validation Empirique** | 2 semaines | +1.7 thèse, +1.5 article | 🔴 CRITIQUE |
| **Phase 2: Explainabilité + Causalité** | 2 semaines | +0.5 thèse, +0.5 article | 🟠 IMPORTANT |
| **Phase 3: Monitoring + Polish** | 1 semaine | +0.2 business | 🟡 OPTIONNEL |

**Total temps**: ~5 semaines intensive  
**Exclus**: Authentification (demandée par utilisateur)

---

# 🔴 PHASE 1: VALIDATION EMPIRIQUE (2 SEMAINES) - CRITIQUE

## 1.1 Backtest Walk-Forward Validation (5 jours)

### Objectif
Prouver empiriquement que CNN-LSTM fonctionne sur **toutes les années 2016-2024**

### Tâches

#### 1.1.1 Créer script de backtesting
```
Fichier: analytics/validation/backtest_comparison.py
Contenu:
- Walk-forward validation (Train 2010-Y, Test Y) pour Y in [2016, 2024]
- Comparer 4 modèles:
  * Baseline = Moyenne dernière année
  * ARIMA(2,1,2)
  * Prophet
  * CNN-LSTM (existant)
  
Sortie:
  * CSV: validation/results_{disease}.csv
  * Métriques: RMSE, MAE, MAPE
  * Improvement %: (ARIMA - CNN-LSTM) / ARIMA
```

**Effort**: 8h  
**Dépendances**: statsmodels, prophet installés  

#### 1.1.2 Générer résultats pour 5 maladies
```
Maladies à tester:
- VIH/SIDA
- Paludisme
- Choléra
- Coqueluche
- Anémie

Sortie attendue: Table finale
┌──────────────────────────────────────────────────────┐
│ Maladie    │ CNN RMSE (avg) │ ARIMA RMSE │ Gain       │
├──────────────────────────────────────────────────────┤
│ VIH/SIDA   │     45         │    78      │ 42%        │
│ Paludisme  │     52         │    91      │ 43%        │
│ Choléra    │     38         │    65      │ 42%        │
│ Coqueluche │     61         │   102      │ 40%        │
│ Anémie     │     49         │    84      │ 42%        │
└──────────────────────────────────────────────────────┘
```

**Effort**: 4h (script exécution)

#### 1.1.3 Créer visualisations backtest
```
Fichier: validation/visualizations_backtest.py

Graphiques:
1. Line plot: RMSE par année (CNN vs ARIMA)
2. Heatmap: Disease x Year x RMSE (CNN/ARIMA)
3. Bar chart: Distribution amélioration % par maladie
4. Scatter: Actual vs Predicted (CNN vs ARIMA)

Format: PNG base64 (pour rapports Word/web)
```

**Effort**: 6h  
**Sortie**: 4 PNG haute qualité

---

## 1.2 Test Spatio-Temporel (3 jours)

### Objectif
Prouver que **composante spatiale améliore vraiment les prédictions**

### Tâches

#### 1.2.1 CNN-LSTM vs LSTM (ablation architecture)
```
Fichier: analytics/ablations/test_spatial_importance.py

Comparaison:
- LSTM-only (sans convolution)
- CNN-LSTM (avec convolution 2D sur matrice provinces)
- LSTM-only RMSE vs CNN-LSTM RMSE

Résultat: Montrer que CNN ajoute X% d'amélioration
```

**Effort**: 6h

#### 1.2.2 Analyser corrélations spatiales
```
Fichier: analytics/spatial_analysis/proximity_correlation.py

Analyse:
- Pour chaque paire provinces (Kinshasa, Kasai):
  * Corrélation croisée = cross_correlation(t_lag)
  * Lag optimal = délai de transmission
  * Force de corrélation

Exemple résultat:
  Kasai → Kinshasa: lag=2 mois, corrélation=0.68
  Tshopo → Kinshasa: lag=1 mois, corrélation=0.45

Sortie: Matrice corrélations spatiales + heatmap
```

**Effort**: 4h

---

## 1.3 Validation Croisée Temporelle (2 jours)

### Tâches

#### 1.3.1 Time-series cross-validation
```
Fichier: analytics/validation/time_series_cv.py

Méthode:
- 5-fold time-series cross-validation
- Pas de shuffle (respect temporalité)
- Résultat: variance de performance

Sortie:
  Fold 1: RMSE = 45 ± 3
  Fold 2: RMSE = 48 ± 2
  ...
  Moyenne: RMSE = 46 ± 2.5 (95% IC)
```

**Effort**: 4h

---

### ✅ Deliverables Phase 1

- [ ] `analytics/validation/backtest_comparison.py` - Script complet
- [ ] `validation/results_{disease}.csv` - Résultats 5 maladies
- [ ] `validation/visualizations_backtest.py` - 4 graphiques principaux
- [ ] Table comparative final (RMSE, MAE, MAPE)
- [ ] `analytics/ablations/test_spatial_importance.py`
- [ ] Matrice corrélations spatiales + heatmap
- [ ] Documentation: VALIDATION_RESULTS.md (dans docs/)

**Impact**: Thèse 7.5 → 8.8 (+1.3) | Article 6.5 → 8.0 (+1.5)

---

# 🟠 PHASE 2: EXPLAINABILITÉ + CAUSALITÉ (2 SEMAINES)

## 2.1 SHAP Explainability (4 jours)

### Objectif
Répondre: "Pourquoi le modèle prédit 150 cas pour Kinshasa en mai?"

### Tâches

#### 2.1.1 SHAP force plots par prédiction
```
Fichier: analytics/explainability/shap_analysis.py

Fonctionnalité:
- Pour chaque prédiction, calculer SHAP values
- Force plot: "Feature contribution à la prédiction"
- Exemple:
  Kinshasa mai 2024: Prédiction = 150 cas
  ├─ Historique Kinshasa (avril) = +45 cas
  ├─ Kasai voisin (avril) = +30 cas
  ├─ Température = +15 cas
  ├─ Saisonnalité (mai) = +40 cas
  └─ Autres = +20 cas

Sortie: PNG interactif (Plotly)
```

**Effort**: 8h

#### 2.1.2 Feature importance temporelle
```
Fichier: analytics/explainability/temporal_feature_importance.py

Analyse:
- Importance des features par período:
  * Décembre-mars: Température dominante?
  * Avril-novembre: Saisonnalité? Spatiale?
  
Heatmap: Feature x Month x Importance
```

**Effort**: 4h

---

## 2.2 Granger Causality (4 jours)

### Objectif
Prouver relations causales: **Température → Paludisme** vs **Corrélation**

### Tâches

#### 2.2.1 Granger causality test
```
Fichier: analytics/causal/granger_causality_test.py

Test pour chaque maladie:
- Température → Cas? (p-value < 0.05?)
- Précipitation → Cas? 
- Humidité → Cas?

Résultat par maladie:
┌─────────────────────────────────────────┐
│ Maladie    │ Température │ Précipitation │
├─────────────────────────────────────────┤
│ Paludisme  │ ✅ p=0.001  │ ✅ p=0.02    │
│ Choléra    │ ❌ p=0.45   │ ✅ p=0.001   │
│ VIH/SIDA   │ ❌ p=0.67   │ ❌ p=0.89    │
└─────────────────────────────────────────┘
```

**Effort**: 6h

#### 2.2.2 Transfer Entropy (optionnel, premium)
```
Fichier: analytics/causal/transfer_entropy.py

Alternative à Granger, plus robuste
- Capture relations non-linéaires
- Lagué temporal dependencies

Effort: 6h (optionnel)
```

---

## 2.3 Résiduals Analysis (2 jours)

### Tâches

#### 2.3.1 Analyser erreurs du modèle
```
Fichier: analytics/diagnostics/residuals_analysis.py

Analyse:
- Distribution des erreurs (normale?)
- Autocorrélation des résidus (Durbin-Watson)
- Erreurs systématiques par saison/région

Sortie: 
- Distribution plot
- ACF plot (autocorrélation)
- Errore par province (identifier provinces difficiles)
```

**Effort**: 4h

---

### ✅ Deliverables Phase 2

- [ ] `analytics/explainability/shap_analysis.py` - SHAP force plots
- [ ] Visualisations SHAP (PNG/HTML)
- [ ] `analytics/causal/granger_causality_test.py`
- [ ] Tableau causalité (Température/Précipitar → Maladies)
- [ ] `analytics/diagnostics/residuals_analysis.py`
- [ ] Documentation: EXPLAINABILITY.md

**Impact**: Thèse 8.8 → 9.2 (+0.4) | Article 8.0 → 8.5 (+0.5)

---

# 🟡 PHASE 3: MONITORING + POLISHING (1-2 SEMAINES)

## 3.1 Model Monitoring (3 jours)

### Objectif
Détecter quand le modèle "dérive" (perd en précision)

### Tâches

#### 3.1.1 Concept drift detection
```
Fichier: analytics/monitoring/concept_drift.py

Implémentation:
- Rolling RMSE (fenêtre 30j)
- Alert si RMSE augmente > 30%
- Log: "Model performance dégradée le 13/03/2026"

Sortie: 
- Dashboard monitoring (visualiser drift)
- JSON logs pour alertes
```

**Effort**: 5h

#### 3.1.2 Data quality checks
```
Fichier: analytics/monitoring/data_quality.py

Checks:
- Données manquantes? (% par province)
- Outliers détectés? (zscore > 3)
- Saut de données? (date gaps)

Sortie: Daily report (email/dashboard)
```

**Effort**: 4h

---

## 3.2 Documentation Scientifique (3 jours)

### Tâches

#### 3.2.1 Écrire METHODOLOGY.md
```
Fichier: docs/METHODOLOGY.md

Sections:
1. Data Collection & Preprocessing
2. Spatio-Temporal Architecture
   - CNN: Raison convolution 2D
   - LSTM: Séquences temporelles
   - Hybrid: Combinaison
3. Climate Integration
4. Validation Strategy
   - Walk-forward (2016-2024)
   - Temporal CV
   - Results: Table comparative
5. Limitations & Future Work
```

**Effort**: 6h

#### 3.2.2 Écrire RESULTS.md
```
Fichier: docs/RESULTS.md

Sections:
1. Main findings (table comparative)
2. By-disease analysis (5 maladies)
3. Spatial correlation analysis
4. SHAP interpretability examples
5. Granger causality findings
```

**Effort**: 4h

---

## 3.3 Articles Template (2 jours)

### Tâches

#### 3.3.1 Article 1 skeleton
```
Fichier: docs/ARTICLE_1_ML_APPROACH.md

Outline:
1. Title: "CNN-LSTM for Disease Forecasting..."
2. Abstract (150 words)
3. Introduction (page 1)
4. Data (page 1)
5. Methods (pages 2-3)
6. Results (page 2) - INCLUDE TABLE 1
7. Discussion (page 1)
8. Conclusion
9. References

Todo: Fill in with your validation results
```

**Effort**: 5h

#### 3.3.2 Article 2 skeleton
```
Fichier: docs/ARTICLE_2_CAUSAL_CLIMATE.md

Outline:
1. Title: "Climate-Disease Causality..."
2. Abstract
3. Background (climate+disease literature)
4. Methods (Granger, Transfer entropy)
5. Results - TABLE 2: Causal findings
6. Policy implications
7. Discussion
8. References

Todo: Fill with granger results
```

**Effort**: 4h

---

### ✅ Deliverables Phase 3

- [ ] `analytics/monitoring/concept_drift.py`
- [ ] `analytics/monitoring/data_quality.py`
- [ ] `docs/METHODOLOGY.md` - Complet, prêt publication
- [ ] `docs/RESULTS.md` - Résultats résumés
- [ ] `docs/ARTICLE_1_ML_APPROACH.md` - Template rempli
- [ ] `docs/ARTICLE_2_CAUSAL_CLIMATE.md` - Template rempli

**Impact**: Business 8.5 → 9.2 (+0.7) | Preparedness pour publication

---

# 📋 CHECKLIST GLOBALE

## Thèse (7.5 → 9.2)
- [ ] Backtesting 5 ans (2016-2024) ← PRIORITÉ 1
- [ ] Benchmark vs ARIMA/Prophet ← PRIORITÉ 1
- [ ] Table comparative (RMSE, MAE, MAPE)
- [ ] Ablation: CNN vs LSTM vs architecture
- [ ] Corrélations spatiales analysées
- [ ] Granger causality test
- [ ] SHAP explainability examples
- [ ] METHODOLOGY.md documenté
- [ ] Résiduals analysis complet

## Article (6.5 → 8.5)
- [ ] TABLE 1: Résultats comparatifs
- [ ] Narrative: "Why CNN-LSTM > ARIMA"
- [ ] Granger causality TABLE 2
- [ ] SHAP visualizations
- [ ] Article 1 skeleton filled
- [ ] Article 2 skeleton filled

## Business (8.5 → 9.2)
- [ ] Monitoring system (concept drift)
- [ ] Data quality checks
- [ ] Daily monitoring dashboard
- [ ] Alert system implémenté
- [ ] Logs + observabilité

---

# 🗓️ PLANNING TEMPOREL RECOMMANDÉ

```
SEMAINE 1 (13-17 mars):
├─ Lundi-Mardi: Backtest script (1.1.1)
├─ Mercredi: Générer résultats (1.1.2)
├─ Jeudi-Vendredi: Visualisations (1.1.3)
└─ Score: Thèse 7.5 → 8.2

SEMAINE 2 (20-24 mars):
├─ Lundi-Mardi: Spatial importance (1.2.1)
├─ Mercredi: Corrélations spatiales (1.2.2)
├─ Jeudi-Vendredi: SHAP explainability (2.1.1)
└─ Score: Thèse 8.2 → 8.8

SEMAINE 3 (27-31 mars):
├─ Lundi-Mardi: Granger causality (2.2.1)
├─ Mercredi: Residuals analysis (2.3.1)
├─ Jeudi: Documentation (3.2)
├─ Vendredi: Monitoring setup (3.1)
└─ FINAL Score: Thèse 8.8 → 9.2

SEMAINE 4 (3-7 avril):
├─ Rédaction articles (Article 1 + 2)
└─ Prêt pour soumission
```

---

# 🔧 DÉPENDANCES À INSTALLER

```bash
pip install statsmodels  # ARIMA
pip install prophet      # Facebook Prophet
pip install shap         # SHAP explainability
pip install scipy        # Granger causality
pip install numpy scipy pandas scikit-learn

# Optionnel mais recommandé:
pip install plotly       # Visualisations interactives
pip install seaborn      # Heatmaps
```

---

# 📚 RESSOURCES & LIENS

## Librairies de référence
- ARIMA: `from statsmodels.tsa.arima.model import ARIMA`
- Granger: `from statsmodels.tsa.api import grangercausalitytests`
- SHAP: `import shap`
- Prophet: `from prophet import Prophet`

## Documentation utile
- Walk-forward validation: https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split
- Granger causality: https://en.wikipedia.org/wiki/Granger_causality
- SHAP: https://shap.readthedocs.io/

---

# 📊 EXPECTED OUTCOMES (Exemple)

Une fois tout terminé, vous pourrez présenter:

### Table 1: Comparative Model Performance (2016-2024)
```
┌──────────────────────────────────────────────────────────────┐
│ Model          │ RMSE  │ MAE   │ MAPE   │ Inference  │       │
├──────────────────────────────────────────────────────────────┤
│ Baseline       │ 120   │ 85    │ 15.2%  │ N/A        │       │
│ ARIMA(2,1,2)   │  84   │ 62    │ 11.8%  │ 0.5s       │       │
│ Prophet        │  78   │ 58    │ 11.2%  │ 0.8s       │       │
│ LSTM           │  65   │ 48    │  9.5%  │ 0.3s       │       │
│ CNN-LSTM       │  49   │ 36    │  7.1%  │ 0.2s       │ ⭐⭐⭐ │
├──────────────────────────────────────────────────────────────┤
│ CNN Improvement │ 42%↓  │ 42%↓  │ 40%↓   │           │       │
└──────────────────────────────────────────────────────────────┘
```

### Table 2: Granger Causality Results
```
┌──────────────────────────────────┐
│ Disease    │ Temp → │ Precip →  │
├──────────────────────────────────┤
│ Paludisme  │ ✅ ***│ ✅ *     │
│ Choléra    │ ❌    │ ✅ ***   │
│ VIH/SIDA   │ ❌    │ ❌       │
└──────────────────────────────────┘
* p<0.05, ** p<0.01, *** p<0.001
```

### Narrative pour article
> "CNN-LSTM captures inter-provincial transmission patterns that univariate ARIMA cannot, achieving 42% improvement in RMSE. Spatio-temporal architecture leverages both historical temporal dynamics and geographic proximity of neighboring regions. Granger causality analysis reveals that climate factors significantly influence malaria transmission (p<0.001), supporting the integration of meteorological data into epidemiological forecasting models."

---

# ❌ CE QUI N'EST PAS INCLUS (Comme demandé)

- Système d'authentification/permissions
- API documentation (Swagger)
- Advanced monitoring (Prometheus/Grafana)

Ces éléments peuvent être ajoutés ultérieurement si nécessaire.

---

# 📞 SUPPORT

Pour chaque tâche, voir:
- Code snippets dans ce document
- Fichiers existants: `analytics/disease_analyzer.py`, `analytics/models.py`
- GitHub issues: Créer tickeet pour chaque phase

---

**Dernier update**: 13 mars 2026  
**Status**: 🔴 À faire  
**ETA Complétion**: 7 avril 2026 (4 semaines intensive)
