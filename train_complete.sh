#!/bin/bash

# 🏥 HealthDRC - Script d'entraînement complet
# Ce script exécute le pipeline complet d'entraînement

echo "============================================================"
echo "🏥 HealthDRC - Entraînement Complet"
echo "============================================================"
echo ""
echo "Ce programme va :"
echo "  1. Charger données maladies (ClinicalData)"
echo "  2. Charger données provinces et zones"
echo "  3. Charger données climatiques (fichiers .nc)"
echo "  4. Nettoyer et combler les trous"
echo "  5. Entraîner modèles multi-tier"
echo "  6. Générer prédictions pour toutes zones"
echo "  7. Sauvegarder dans la base"
echo "  8. Préparer dashboard"
echo ""
echo "Durée estimée: 1-2 heures selon données"
echo "============================================================"
echo ""

# Vérifier que Docker est en cours d'exécution
if ! docker-compose ps | grep -q "app.*running"; then
    echo "⚠️  Docker n'est pas en cours d'exécution."
    echo "   Démarrage avec: docker-compose up -d"
    docker-compose up -d
    echo "   Attente de 30 secondes..."
    sleep 30
fi

# Exécuter le script d'entraînement
echo "🚀 Démarrage de l'entraînement..."
echo ""

docker-compose exec app python train_complete.py

# Vérifier le résultat
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Entraînement terminé avec succès !"
    echo ""
    echo "📊 Voir les résultats :"
    echo "  • Dashboard: http://localhost:8000/analysis/dashboard/"
    echo "  • Prédictions: http://localhost:8000/analysis/disease/"
    echo ""
    echo "📁 Fichiers générés :"
    echo "  • Modèles: models/model_*.keras"
    echo "  • Métadonnées: models/dashboard_training_data.json"
    echo "  • Données traitées: data/processed/epi_data_processed.csv"
    echo ""
    echo "🔍 Logs détaillés : training.log"
else
    echo ""
    echo "❌ Erreur pendant l'entraînement."
    echo "   Consultez training.log pour plus de détails."
    exit 1
fi
