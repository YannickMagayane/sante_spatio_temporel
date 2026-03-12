# 🔧 SOLUTION: PERSISTANCE DES DONNÉES & INTERFACE D'ENTRAÎNEMENT

## PROBLÈME 1: Les données disparaissent à chaque redémarrage

### Causes probables:
1. **PostgreSQL ne persiste pas** → Vérifier les volumes Docker
2. **Migrations Django non exécutées** → Tables non créées
3. **SQLite utilisé au lieu de PostgreSQL** → Données stockées dans un conteneur éphémère

### SOLUTION: Configuration Correcte de Docker

Votre `docker-compose.yml` a **déjà** les volumes configurés:
```yaml
volumes:
  postgres_data:

services:
  db:
    volumes:
      - postgres_data:/var/lib/postgresql/data
```

✅ **C'est correct!** Mais vous devez vous assurer que:

#### 1️⃣ Les migrations Django s'exécutent au démarrage

Ajoutez ceci dans votre `Dockerfile` ou `compose/app/Dockerfile`:

```dockerfile
# Dans le Dockerfile existant, ajoutez la ligne suivante AVANT le CMD:
RUN python manage.py migrate --noinput

CMD ["gunicorn", "core.wsgi:application", "--bind", "0.0.0.0:8000"]
```

#### 2️⃣ Vérifiez que `.env` utilise PostgreSQL

Assurez-vous que `.env` contient:
```env
DATABASE_URL=postgres://user:password@db:5432/health_db
POSTGRES_DB=health_db
POSTGRES_USER=health
POSTGRES_PASSWORD=secure_password_here
```

**NOT SQLite!**

#### 3️⃣ Utilisez les commandes correctes pour nettoyer/redémarrer

```bash
# DONT (supprime les données):
docker-compose down

# DO (garde les données):
docker-compose stop
docker-compose start

# Si vous devez tout reconstruire:
docker-compose down -v  # ⚠️ SUPPRIME LES DONNÉES
docker volume create health_postgres_data  # Créer le volume nommé
docker-compose up -d
```

---

## PROBLÈME 2: Interface manquante pour entraîner & injecter les données

### J'ai créé une solution complète:

#### ✅ Implémentation faite:

1. **Modèle `TaskRun`** → Suit l'exécution des tâches
2. **Module `task_runner.py`** → Exécute les scripts en arrière-plan (threading)
3. **Vues Django** → Boutons dans l'interface pour lancer les tâches
4. **Template `task_list.html`** → Affiche l'historique et la progression
5. **URLs configurées** → `/tasks/`, `/tasks/launch/training/`, `/tasks/launch/ingest/`

---

## 🚀 MISE EN PLACE IMMÉDIATE

### ÉTAPE 1: Appliquer les migrations

```bash
# Accédez au conteneur
docker exec -it health_app_1 bash

# Exécutez les migrations
python manage.py migrate

# Vérifiez que le modèle TaskRun est créé
python manage.py showmigrations
```

### ÉTAPE 2: Tester les boutons dans l'interface

1. Allez sur `http://localhost:8000/` (le dashboard)
2. Vous verrez une nouvelle section **"⚙️ Gestion des Données & Modèles"**
3. Cliquez sur **"🧠 Entraîner les Modèles"** ou **"📥 Injecter Données HDX"**
4. Les tâches s'exécutent en arrière-plan sans bloquer l'interface
5. Allez sur `http://localhost:8000/tasks/` pour voir l'historique

### ÉTAPE 3: Vérifier la persistance

```bash
# Redémarrez le Docker
docker-compose down
docker-compose up -d

# Les données doivent TOUJOURS être là!
# Vérifiez via le dashboard
```

---

## 📊 Comment fonctionne le système de tâches?

### Flow complet:

1. **Clic sur bouton** → POST request vers Django
2. **Vue Django** → Crée un `TaskRun` et lance un thread
3. **Thread** → Exécute le script Python en arrière-plan
4. **Capture de logs** → La sortie est stockée en DB
5. **Interface** → Affiche la progression et les résultats

### Avantages:
- ✅ L'interface ne se bloque pas
- ✅ Historique persistant des tâches
- ✅ Possibilité de relancer facilement
- ✅ Logs disponibles pour diagnostiquer les erreurs

---

## 🔒 Garantir la Persistance - CHECKLIST

```bash
# 1. Vérifier que le volume Docker existe et est utilisé
docker volume ls | grep postgres_data

# 2. Vérifier que les données sont vraiment persistées
docker exec health_db_1 psql -U health health_db -c "SELECT COUNT(*) FROM analytics_clinicaldata;"

# 3. Vérifier les migrations
docker exec health_app_1 python manage.py showmigrations

# 4. Tester un redémarrage complet
docker-compose down
docker volume ls | grep postgres  # Vérifier que postgres_data EXISTE
docker-compose up -d

# 5. Vérifier que les données sont toujours là
docker exec health_app_1 python manage.py shell -c "from analytics.models import ClinicalData; print(f'Rows: {ClinicalData.objects.count()}')"
```

---

## ❓ FAQ

**Q: Où se trouvent physiquement les données?**
- A: Dans le volume Docker `health_postgres_data`, généralement à `/var/lib/docker/volumes/health_postgres_data/_data/`

**Q: Puis-je exporter les données avant de redémarrer?**
- A: Oui! `docker exec health_db_1 pg_dump -U health health_db > backup.sql`

**Q: Les différents tiers (Tier1, Tier2, Tier3) vont-ils être conservés?**
- A: Oui! Ils sont stockés dans `Prediction.influence_factors` (JSON en base de données)

**Q: Puis-je lancer plusieurs entraînements simultanément?**
- A: Non, le système empêche les exécutions concurrentes pour éviter les conflits. Mais chaque tâche est mise en queue.

---

## 📝 Prochaines étapes recommandées

1. ✅ Exécuter les migrations
2. ✅ Tester les boutons d'entraînement/ingestion
3. ✅ Vérifier la persistance après un redémarrage
4. 📌 Ajouter les tâches planifiées (entraînement automatique quand il y a de nouvelles données) - *Optionnel*
5. 📌 Ajouter des emails/notifications quand une tâche est complétée - *Optionnel*

---

## 🎯 Résumé des modifications

| Fichier | Changement |
|---------|-----------|
| `analytics/models.py` | ✅ Ajout du modèle `TaskRun` |
| `analytics/task_runner.py` | ✅ Nouveau fichier pour exécuter les tâches |
| `analytics/views.py` | ✅ 3 nouvelles vues pour les tâches |
| `analytics/urls.py` | ✅ 4 nouveaux endpoints |
| `analytics/migrations/0002_add_taskrun.py` | ✅ Migration pour `TaskRun` |
| `templates/analytics/task_list.html` | ✅ Nouveau template pour afficher les tâches |
| `templates/analytics/dashboard.html` | ✅ Ajout section "⚙️ Gestion des Données & Modèles" |
| `requirements.txt` | ℹ️ Pas de changement nécessaire |

