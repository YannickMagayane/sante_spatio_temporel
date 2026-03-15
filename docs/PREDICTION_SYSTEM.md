# 📊 Système de Prédiction Progressive

## Vue d'ensemble

Le système a été refactorisé pour permettre une **prédiction progressive sans réentraînement**. Cela signifie:

1. **Première exécution (ou réentraînement forcé)**: 
   - Entraîne les modèles IA (ConvLSTM, CNN-LSTM, Bi-LSTM)
   - Fait les prédictions progressives pour chaque année de données
   - Sauvegarde les métadonnées d'entraînement

2. **Exécutions suivantes**:
   - Détecte que les modèles existent déjà
   - Fait les prédictions rapidement sans réentraîner (Tier 4 - Persistence)
   - Met à jour les prédictions pour chaque année jusqu'à aujourd'hui

## Architecture

### Métadonnées (`/app/models/training_metadata.json`)

Garde trace de:
- `last_training_date`: Date du dernier entraînement
- `force_retrain`: Flag pour forcer un réentraînement (défaut: False)

### Processus d'exécution

```
train_and_predict.py
│
├─ Vérifier should_retrain()
│  │
│  ├─ TRUE → Entraînement + Prédictions Progressives
│  │  ├─ train_and_predict(train_mode=True)  [Entraîne tous les modèles]
│  │  ├─ Sauvegarde métadonnées
│  │  └─ do_progressive_predictions()  [Prédit pour chaque année]
│  │
│  └─ FALSE → Prédictions Progressives uniquement
│     └─ do_progressive_predictions()  [Utilise Tier 4 - Persistence]
│
└─ Retour
```

### Modes de train_and_predict()

#### Mode 1: Entraînement + Prédiction (par défaut)
```python
train_and_predict(train_mode=True, verbose=True)
```
- Entraîne les modèles IA (peut prendre 30-60min)
- Crée des prédictions pour le mois prochain
- Affiche tous les logs

#### Mode 2: Prédiction uniquement
```python
train_and_predict(prediction_date=date(2024, 1, 1), train_mode=False, verbose=False)
```
- N'entraîne PAS
- Utilise les données historiques (Tier 4 - Persistence)
- Crée des prédictions rapidement (quelques secondes)
- Logs minimaux

## Prédictions progressives par année

Pour chaque année de données:
- **Années passées** (2014-2025): Prédiction pour le 1er janvier de l'année suivante
- **Année actuelle** (2026): Prédictions pour les 2 prochains mois

Exemple:
```
2014 → Prédiction pour 2015-01-01
2015 → Prédiction pour 2016-01-01
...
2025 → Prédiction pour 2026-01-01
2026 → Prédictions pour 2026-04-01, 2026-05-01 (à partir de mars 2026)
```

## Utilisation

### Lancer le système complet

```bash
docker-compose exec app python scripts/train_models.py
```

Cela va:
1. Vérifier s'il y a un réentraînement nécessaire
2. Si oui: Entraîner + faire prédictions progressives
3. Si non: Faire juste les prédictions progressives

### Forcer un réentraînement

Modifier `/app/models/training_metadata.json`:
```json
{
  "last_training_date": "2026-03-13T...",
  "force_retrain": true
}
```

Ou supprimer le fichier complètement pour réinitialiser.

## Conditions de réentraînement

Le système réentraîne si:
1. Aucun fichier de métadonnées trouvé
2. Aucun modèle Keras existant (`/app/models/*.keras`)
3. `force_retrain: true` dans les métadonnées
4. Les données ont été mises à jour après le dernier entraînement

## Performance

### Temps d'exécution

- **Entraînement complet**: 30-60 minutes (premiers 16+ mois par maladie/province)
- **Prédiction Tier 4**: 2-5 secondes par année

### Stockage

- Modèles Keras: ~50-200MB (ConvLSTM, CNN-LSTM, Bi-LSTM)
- Métadonnées: < 1KB
- Rapport JSON: ~1-5MB

## Tiers (modèles utilisés)

| Tier | Modèle | Données min | Utilisation |
|------|--------|------------|------------|
| 1 | ConvLSTM2D | 16+ mois | Spatio-temporal, données riches |
| 2 | CNN-LSTM | 8+ mois | Patterns locaux + mémoire |
| 3 | Bi-LSTM | 4+ mois | Séquences temporelles |
| 4 | Persistence | <4 mois | Fallback basique (dernière valeur) |

### Mode Prédiction Sans Réentraînement

Quand `train_mode=False`:
- Utilise toujours **Tier 4 (Persistence)**
- Base: dernière valeur connue + volatilité historique
- Fiabilité: 33% (conforme à Tier 4 standard)
- Pas d'entraînement = très rapide

## Exemple: Flux complet

```
🚀 Jour 1: Premier lancement
  └─ should_retrain() = TRUE
     ├─ train_and_predict(train_mode=True) [30-60min]
     ├─ Modèles entraînés et sauvegardés
     └─ do_progressive_predictions()  [5-10min]
        ├─ 2014 → Prédiction 2015-01
        ├─ 2015 → Prédiction 2016-01
        ├─ ...
        └─ 2026 → Prédictions 2026-04, 2026-05
        
🚀 Jour 2: Redémarrage du conteneur
  └─ should_retrain() = FALSE
     └─ do_progressive_predictions()  [5-10 sec]
        ├─ 2014 → Prédiction 2015-01 (mise à jour rapide Tier 4)
        ├─ 2015 → Prédiction 2016-01
        ├─ ...
        └─ 2026 → Prédictions 2026-04, 2026-05
```

## Surveillance

Les prédictions sont enregistrées pour chaque mois cible:
- Table `Prediction`: contient les prédictions par zone
- Colonne `influence_factors`: détails du modèle utilisé (tier, R², fiabilité, etc.)

## Notes d'implémentation

- Les scalers (MinMax, Robust) sont appliqués avant l'entraînement
- Les données manquantes sont interpolées linéairement
- La saisonnalité est encodée en sin/cos (préservée lors de l'inversion)
- Les données climatiques sont intégrées si disponibles par province
- Log-transformation appliquée pour réduire les variances énormes (Paludisme: 314M vs Choléra: 453k)
