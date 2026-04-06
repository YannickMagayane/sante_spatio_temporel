#!/bin/bash

# 📈 HealthDRC - Mise à jour des taux de propagation
# Calcule et met à jour les taux de croissance des prédictions

echo "============================================================"
echo "📈 Calcul des taux de propagation"
echo "============================================================"
echo ""
echo "Cette opération va :"
echo "  • Parcourir toutes les prédictions récentes"
echo "  • Comparer avec l'historique (3 derniers mois)"
echo "  • Calculer le taux de croissance"
echo "  • Déterminer le niveau d'alerte (ROUGE/ORANGE/STABLE)"
echo ""

# Vérifier Docker
if ! docker-compose ps | grep -q "app.*running"; then
    echo "⚠️  Docker n'est pas en cours d'exécution."
    docker-compose up -d
    sleep 10
fi

# Exécuter
echo "🚀 Démarrage..."
docker-compose exec app python update_spread_rates.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Propagation rates mis à jour !"
    echo ""
    echo "🔄 Rafraîchissez le dashboard:"
    echo "   http://localhost:8000/analysis/dashboard/"
else
    echo ""
    echo "❌ Erreur lors du calcul"
    exit 1
fi
