#!/bin/bash
# 🔍 VERIFY_TRAINING_ENVIRONMENT.sh
# Vérifie que tout est prêt pour lancer l'entraînement

set -e

echo "🔍 VÉRIFICATION ENVIRONNEMENT D'ENTRAÎNEMENT..."
echo "================================================"
echo ""

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Tracker
PASSED=0
FAILED=0

# FONCTION: Test et affiche résultat
test_requirement() {
    local test_name=$1
    local command=$2
    local expected=$3
    
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✅${NC} $test_name"
        ((PASSED++))
    else
        echo -e "${RED}❌${NC} $test_name"
        ((FAILED++))
    fi
}

# FONCTION: Afficher info
show_info() {
    local test_name=$1
    local command=$2
    
    echo ""
    echo "📍 $test_name:"
    eval "$command"
    echo ""
}

# ========== 1. Docker Services ==========
echo "1️⃣  SERVICES DOCKER"
echo "-------------------"

test_requirement "Docker daemon actif" "docker ps > /dev/null 2>&1"
test_requirement "Docker Compose installé" "docker-compose --version > /dev/null 2>&1"

# Vérifier services
echo ""
echo "📋 Statut des services:"
docker-compose ps --services | while read service; do
    if docker-compose ps $service | grep -q "Up"; then
        echo -e "  ${GREEN}✅${NC} $service (running)"
    else
        echo -e "  ${RED}❌${NC} $service (NOT running)"
        ((FAILED++))
    fi
done

# ========== 2. Réseau ==========
echo ""
echo "2️⃣  RÉSEAU & PORTS"
echo "------------------"

test_requirement "JupyterLab (port 8888)" "curl -s http://localhost:8888 > /dev/null || [ \$? -eq 7 ]"
test_requirement "Django App (port 8000)" "curl -s http://localhost:8000 > /dev/null || [ \$? -eq 7 ]"
test_requirement "Database (port 5432)" "nc -z localhost 5432 > /dev/null 2>&1 || true"

# ========== 3. Fichiers ==========
echo ""
echo "3️⃣  FICHIERS ESSENTIELS"
echo "-----------------------"

test_requirement "Notebook 01_data_preprocessing.ipynb existe" "test -f /home/yannick-magayane/Bureau/projets/data_analyse/health/notebooks/01_data_preprocessing.ipynb"
test_requirement "Notebook 02_model_training.ipynb existe" "test -f /home/yannick-magayane/Bureau/projets/data_analyse/health/notebooks/02_model_training.ipynb"
test_requirement "Notebook 03_generate_predictions.ipynb existe" "test -f /home/yannick-magayane/Bureau/projets/data_analyse/health/notebooks/03_generate_predictions.ipynb"

test_requirement "Dossier models existe" "test -d /home/yannick-magayane/Bureau/projets/data_analyse/health/models"
test_requirement "Dossier data existe" "test -d /home/yannick-magayane/Bureau/projets/data_analyse/health/data"

# ========== 4. Données ==========
echo ""
echo "4️⃣  DONNÉES PRÉTRAITÉES"
echo "-----------------------"

if test -f /home/yannick-magayane/Bureau/projets/data_analyse/health/data/processed/epi_data.csv; then
    echo -e "  ${GREEN}✅${NC} Données épidémiologiques existantes"
    ((PASSED++))
else
    echo -e "  ${YELLOW}⚠️${NC} Données épidémiologiques NON TROUVÉES"
    echo "     → VOUS DEVEZ EXÉCUTER: 01_data_preprocessing.ipynb"
    ((FAILED++))
fi

if test -f /home/yannick-magayane/Bureau/projets/data_analyse/health/data/processed/climate_data.csv; then
    echo -e "  ${GREEN}✅${NC} Données climatiques existantes"
    ((PASSED++))
else
    echo -e "  ${YELLOW}⚠️${NC} Données climatiques NON TROUVÉES"
    echo "     → VOUS DEVEZ EXÉCUTER: 01_data_preprocessing.ipynb"
    ((FAILED++))
fi

# ========== 5. Dépendances Python ==========
echo ""
echo "5️⃣  DÉPENDANCES PYTHON"
echo "----------------------"

# Vérifier dans Docker
check_module() {
    local module=$1
    docker-compose exec -T app python -c "import $module" > /dev/null 2>&1
}

test_requirement "TensorFlow/Keras installé" "check_module tensorflow"
test_requirement "Django installé" "check_module django"
test_requirement "Pandas installé" "check_module pandas"
test_requirement "NumPy installé" "check_module numpy"
test_requirement "JupyterLab installé" "check_module jupyter"

# ========== 6. Configuration ==========
echo ""
echo "6️⃣  CONFIGURATION DJANGO"
echo "------------------------"

test_requirement "Database migrations appliquées" "docker-compose exec -T app python manage.py showmigrations 2>/dev/null | grep -q 'X'"
test_requirement "Django settings valides" "docker-compose exec -T app python -c 'import django; django.setup()' > /dev/null 2>&1"

# ========== 7. API ==========
echo ""
echo "7️⃣  API ENDPOINTS"
echo "-----------------"

test_requirement "API /api/training/metrics/ accessible" "curl -s http://localhost:8000/analysis/api/training/metrics/ > /dev/null || curl -s http://localhost:8000/analysis/api/training/metrics/ | grep -q success"
test_requirement "Dashboard view accessible" "curl -s http://localhost:8000/analysis/dashboard/ > /dev/null || [ \$? -eq 7 ]"
test_requirement "Notebook control panel accessible" "curl -s http://localhost:8000/analysis/notebooks/ > /dev/null || [ \$? -eq 7 ]"

# ========== RÉSUMÉ ==========
echo ""
echo "================================================"
echo "📊 RÉSUMÉ"
echo "================================================"
echo -e "  ${GREEN}✅ Passé: $PASSED${NC}"

if [ $FAILED -gt 0 ]; then
    echo -e "  ${RED}❌ Échoué: $FAILED${NC}"
    echo ""
    echo "⚠️  ACTIONS REQUISES:"
    echo "   1. Exécuter 01_data_preprocessing.ipynb d'abord"
    echo "   2. Redémarrer Docker: docker-compose restart"
    echo "   3. Relancer ce script"
else
    echo ""
    echo -e "${GREEN}🎉 TOUT EST PRÊT!${NC}"
    echo ""
    echo "Vous pouvez maintenant lancer l'entraînement:"
    echo "  1. Allez à: http://localhost:8888"
    echo "  2. Ouvrez: 02_model_training.ipynb"
    echo "  3. Click: Run All Cells"
fi

echo ""
echo "================================================"
echo ""

exit $FAILED
