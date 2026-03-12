#!/bin/bash
# Script pour générer automatiquement le rapport Paludisme & Choléra
# Usage: ./generate_report.sh

set -e

PROJECT_DIR="/home/yannick-magayane/Bureau/projets/data_analyse/health"
cd "$PROJECT_DIR"

echo "🔵 ========================================="
echo "🔵 RAPPORT PALUDISME & CHOLÉRA — AUTOMATIQUE"
echo "🔵 ========================================="
echo ""

# Étape 1: Exécuter train_models.py
echo "📊 [1/3] Exécution de train_models.py..."
echo "⏱️  Ceci va prendre 15-30 minutes..."
echo ""

docker exec health_app_1 python scripts/train_models.py 2>&1 | tee train_output_$(date +%d%m%Y_%H%M%S).log

echo ""
echo "✅ train_models.py terminé"
echo ""

# Étape 2: Générer le rapport
echo "📝 [2/3] Génération du rapport Word..."

docker exec health_app_1 python /app/report_gen.py

# Étape 3: Copier le rapport
echo ""
echo "📄 [3/3] Copie du rapport vers le répertoire hôte..."

REPORT_FILE=$(docker exec health_app_1 ls -t /app/RAPPORT_*.docx | head -1)
docker cp health_app_1:$REPORT_FILE "$PROJECT_DIR/"

REPORT_NAME=$(basename $REPORT_FILE)
echo ""
echo "✅ ========================================="
echo "✅ RAPPORT GÉNÉRÉ AVEC SUCCÈS!"
echo "✅ ========================================="
echo ""
echo "📁 Fichier: $PROJECT_DIR/$REPORT_NAME"
echo "📊 Maladies: Paludisme & Choléra"
echo "📈 Métriques: R², MAE, RMSE, Distribution Tiers"
echo ""
echo "🎯 Prochaines étapes:"
echo "   1. Ouvrir le fichier Word"
echo "   2. Consulter les métriques Paludisme & Choléra"
echo "   3. Analyser les performances par province"
echo ""
