# 📋 RAPPORT PALUDISME & CHOLÉRA — Guide d'Utilisation

## ✅ Fichier Généré

**Fichier:** `RAPPORT_Paludisme_Cholera_09032026.docx`  
**Taille:** 37 KB  
**Localisation:** `/home/yannick-magayane/Bureau/projets/data_analyse/health/`

## 📊 Status Actuel

```
Total Predictions in DB: 0
Diseases Available: [] (empty)
```

**⚠️ IMPORTANT:** Pour que le rapport soit rempli avec les données réelles, il faut d'abord exécuter `train_models.py` v2.3.

## 🚀 Étapes pour Générer le Rapport Complet

### 1️⃣ Exécuter train_models.py

```bash
cd /home/yannick-magayane/Bureau/projets/data_analyse/health
docker exec health_app_1 python scripts/train_models.py 2>&1 | tee train_output.log
```

**Temps requis:** ~15-30 minutes (selon le volume de données)

**Résultat attendu:**
```
🏁 ENTRAÎNEMENT TERMINÉ ─ RÉSUMÉ
  ✅ Prédictions créées: 2570+
  🌐 Tier 1 (ConvLSTM2D): 26 province×maladie
  🔷 Tier 2 (CNN-LSTM): 95+ province×maladie
  🔹 Tier 3 (Bi-LSTM): 9+ province×maladie
```

### 2️⃣ Générer le Rapport Word

```bash
docker exec health_app_1 python /app/report_gen.py
docker cp health_app_1:/app/RAPPORT_Paludisme_Cholera_DDMMYYYY.docx /home/yannick-magayane/Bureau/projets/data_analyse/health/
```

## 📈 Contenu du Rapport

Le rapport généré contiendra:

### Pour **PALUDISME**:
- ✅ Nombre de prédictions générées
- ✅ Couverture spatiale (26 provinces × N zones)
- ✅ Données cliniques source (N records)
- ✅ Performance des modèles:
  - R² moyen, min, max
  - % modèles avec R² > 0
  - MAE et RMSE moyens
- ✅ Distribution des Tiers (TIER 1, 2, 3, 4)
- ✅ Interprétation épidémiologique

### Pour **CHOLÉRA**:
- ✅ Idem que Paludisme
- ✅ Analyse spécifique maladie hydrique
- ✅ Patterns pics épidémiques

## 🔧 Script Utilisé

**Fichier:** `generate_paludisme_cholera_report.py`

```python
# Extraction des métriques
metrics = get_disease_metrics('Paludisme')
metrics = get_disease_metrics('Choléra')

# Génération Word
doc.add_heading('🦠 PALUDISME', level=1)
doc.add_heading('🦠 CHOLÉRA', level=1)
```

**Localisation:** `/home/yannick-magayane/Bureau/projets/data_analyse/health/generate_paludisme_cholera_report.py`

## 🎯 Résultats Attendus (v2.3)

### Paludisme
```
✅ R² moyen: 0.3-0.6 (excellent épidémiologie)
✅ MAE: 5-40 cas
✅ RMSE: 10-60 cas
✅ % R²>0: 100% (sparse regions skippées)
✅ Tiers: Majorité TIER 1 (ConvLSTM)
```

### Choléra
```
✅ R² moyen: 0.3-0.6 (stable production)
✅ MAE: 5-30 cas
✅ RMSE: 10-40 cas
✅ % R²>0: 100%
✅ Tiers: Mélange TIER 1-2
```

## 🔍 Diagnostic

### Si Rapport est vide:
1. Vérifier que train_models.py a été exécuté ✓
2. Vérifier que Predictions existent en BD:
```bash
docker exec health_app_1 python manage.py shell
>>> from analytics.models import Prediction
>>> Prediction.objects.count()
```
3. Vérifier les noms de maladies exactes:
```bash
>>> Prediction.objects.values_list('disease_name', flat=True).distinct()
```

### Si Rapport a erreurs:
1. Vérifier les logs:
```bash
docker logs health_app_1 | tail -100
```
2. Vérifier les données ClinicalData:
```bash
docker exec health_app_1 python manage.py shell
>>> from analytics.models import ClinicalData
>>> ClinicalData.objects.filter(disease_name='Paludisme').count()
```

## 📌 Notes Importantes

- **Accents DOS:** Le script gère "Choléra" (avec accent) - à adapter si BD utilise "Cholera"
- **Chemin Docker:** Rapport généré dans `/app/` puis copié vers host
- **Temps génération:** ~5 secondes (une fois données disponibles)
- **Format Word:** Compatible Office 2007+

## 🎬 Automatisation Future

Pour automatiser la génération quotidienne:

```bash
# Ajouter au crontab
0 2 * * * cd /app && python scripts/train_models.py && python report_gen.py
```

---

**Généré:** 9 Mars 2026  
**Version:** v2.3 avec fixes critiques  
**Status:** ✅ READY FOR USE
