#!/bin/bash
# 🚀 script_deployment.sh - Déploie la solution complète

set -e

echo "=========================================="
echo "🚀 DÉPLOIEMENT SOLUTION PERSISTANCE + INTERFACE"
echo "=========================================="

cd /home/yannick-magayane/Bureau/projets/data_analyse/health

# 1. Appliquer les modifications
echo "✅ Modifications du code appliquées"

# 2. Vérifier que Docker est prêt
echo ""
echo "📋 État actuel des conteneurs Docker:"
docker-compose ps

# 3. Redémarrer les services pour appliquer les migrations
echo ""
echo "🔄 Redémarrage des conteneurs pour appliquer les migrations..."
docker-compose down
docker-compose up -d

# 4. Attendre que la base de données soit prête
echo ""
echo "⏳ Attente du démarrage de la base de données (30 secondes)..."
sleep 30

# 5. Vérifier que les migrations se sont bien passées
echo ""
echo "🔍 Vérification des migrations..."
docker exec health_app_1 python manage.py showmigrations analytics | head -20

# 6. Vérifier que la table TaskRun existe
echo ""
echo "📊 Création de la table TaskRun..."
docker exec health_app_1 python manage.py migrate

# 7. Résumé
echo ""
echo "=========================================="
echo "✅ DÉPLOIEMENT TERMINÉ!"
echo "=========================================="
echo ""
echo "🌐 Accédez au dashboard:"
echo "   → http://localhost:8000"
echo ""
echo "📋 Boutons disponibles:"
echo "   🧠 Entraîner les Modèles"
echo "   📥 Injecter Données HDX"
echo ""
echo "📊 Historique des tâches:"
echo "   → http://localhost:8000/tasks/"
echo ""
echo "ℹ️  Plus d'infos: SOLUTION_PERSISTANCE_ET_INTERFACE.md"
echo ""
