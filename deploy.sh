#!/bin/bash

# ========================================
# 🚀 DÉPLOIEMENT COMPLET
# Dashboard + Notebooks + Training Data
# ========================================

echo "📦 Déploiement du système complet Health DRC..."
echo ""

# 1. Arrêter les services actuels
echo "1️⃣  Arrêt des services..."
docker-compose down --remove-orphans
sleep 2

# 2. Reconstruire les images
echo "2️⃣  Reconstruction des images Docker..."
docker-compose build --no-cache
sleep 2

# 3. Démarrer les services
echo "3️⃣  Démarrage des services..."
docker-compose up -d
sleep 30

# 4. Vérifier statut
echo "4️⃣  Vérification des services..."
docker-compose ps

echo ""
echo "✅ DÉPLOIEMENT COMPLET!"
echo ""
echo "📍 Accès aux Services:"
echo ""
echo "  🌐 Dashboard Principal"
echo "     http://localhost:8000/"
echo ""
echo "  🧬 Notebook Control Panel"
echo "     http://localhost:8000/analysis/notebooks/"
echo ""
echo "  📓 JupyterLab (Direct)"
echo "     http://localhost:8888"
echo ""
echo "  📊 API Training Data"
echo "     http://localhost:8000/analysis/api/training/metrics/"
echo ""
echo "========================================
"
echo "🎯 PROCHAINES ÉTAPES:"
echo ""
echo "1. Ouvrez http://localhost:8000/analysis/notebooks/"
echo "2. Click '▶️ Execute' sur 02_model_training.ipynb"
echo "3. Attendez ~60 minutes"
echo "4. Les graphiques du dashboard se remplissent automatiquement!"
echo ""
echo "========================================
"
