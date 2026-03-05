#!/bin/bash
# Script de gestion du projet Health Predict RDC

set -e

# Couleurs pour l'affichage
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANAGE_PY="$PROJECT_DIR/manage.py"

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}  PROJECT: Health Predict RDC${NC}"
echo -e "${YELLOW}========================================${NC}\n"

case "${1:-help}" in
    migrate)
        echo -e "${YELLOW}📋 Exécution des migrations...${NC}"
        python "$MANAGE_PY" migrate
        echo -e "${GREEN}✓ Migrations terminées${NC}\n"
        ;;
    
    seed)
        echo -e "${YELLOW}📍 Injection des données...${NC}"
        python "$MANAGE_PY" shell < "$PROJECT_DIR/scripts/inject_data.py"
        echo -e "${GREEN}✓ Données injectées${NC}\n"
        ;;
    
    makemigrations)
        echo -e "${YELLOW}🔨 Création des migrations...${NC}"
        python "$MANAGE_PY" makemigrations
        echo -e "${GREEN}✓ Migrations créées${NC}\n"
        ;;
    
    reset)
        echo -e "${RED}⚠️  RESET: Suppression de toutes les données et migrations...${NC}"
        read -p "Êtes-vous sûr? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            python "$MANAGE_PY" migrate analytics zero
            rm -f "$PROJECT_DIR/analytics/migrations/000*.py" 2>/dev/null || true
            echo -e "${GREEN}✓ Base de données réinitialisée${NC}\n"
        else
            echo -e "${YELLOW}Annulé${NC}\n"
        fi
        ;;
    
    superuser)
        echo -e "${YELLOW}👤 Création d'un superutilisateur...${NC}"
        python "$MANAGE_PY" createsuperuser
        ;;
    
    check)
        echo -e "${YELLOW}✓ Vérification du projet...${NC}"
        python "$MANAGE_PY" check
        echo -e "${GREEN}✓ Projet est valide${NC}\n"
        ;;
    
    runserver)
        echo -e "${YELLOW}🚀 Lancement du serveur de développement...${NC}"
        python "$MANAGE_PY" runserver 0.0.0.0:8000
        ;;
    
    shell)
        echo -e "${YELLOW}🐚 Démarrage du shell Django...${NC}"
        python "$MANAGE_PY" shell
        ;;
    
    stats)
        echo -e "${YELLOW}📊 Statistiques de la base de données...${NC}"
        python "$MANAGE_PY" shell -c "
from analytics.models import Province, Territory, HealthZone
print(f'Provinces: {Province.objects.count()}')
print(f'Territoires: {Territory.objects.count()}')
print(f'Zones de Santé: {HealthZone.objects.count()}')
"
        ;;
    
    *)
        cat << EOF
${YELLOW}Utilisation: $0 <commande>${NC}

Commandes disponibles:
  ${GREEN}migrate${NC}        - Exécuter les migrations Django
  ${GREEN}makemigrations${NC} - Créer les migrations Django
  ${GREEN}seed${NC}          - Injecter les données (26 prov, 145 terr, 519 zones)
  ${GREEN}reset${NC}         - Réinitialiser la BD (DANGER!)
  ${GREEN}superuser${NC}     - Créer un superutilisateur
  ${GREEN}check${NC}         - Vérifier la configuration du projet
  ${GREEN}runserver${NC}     - Lancer le serveur de développement
  ${GREEN}shell${NC}         - Démarrer le shell Django
  ${GREEN}stats${NC}         - Afficher les statistiques de la BD
  ${GREEN}help${NC}          - Afficher cette aide

Exemples:
  $0 makemigrations
  $0 migrate
  $0 seed
  $0 stats

EOF
        ;;
esac
