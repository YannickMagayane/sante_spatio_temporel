#!/bin/bash
set -e

echo "🚀 Démarrage du Pilier HealthDRC..."
until pg_isready -h db -U health_user; do
  echo "Base de données non prête... en attente."
  sleep 2
done

echo "🚀 PostgreSQL est prêt ! Lancement des migrations..."

# 1. Préparation de la base de données
export PYTHONPATH=$PYTHONPATH:/app
python manage.py makemigrations analytics --noinput
python manage.py migrate --noinput

# 2. Synchronisation et Intelligence (HDX, Climat, Seed, Predict)
echo "🧠 Préparation de l'orchestrateur intelligent (démarrage différé)..."
(sleep 15; nice -n 15 python3 scripts/orchestrator.py) &

# 3. Lancement des tâches de fond (Veille sanitaire continue)
echo "🔍 Préparation de la veille sanitaire (démarrage différé)..."
(sleep 30; while true; do nice -n 15 python3 scripts/web_scraping.py --platform all --keywords "épidémie" --city "DRC"; sleep 86400; done) &

# 4. Lancement du serveur Django sur le Dashboard Premium
echo "💻 Dashboard disponible sur http://health.localhost:8000"
exec python manage.py runserver 0.0.0.0:8000
