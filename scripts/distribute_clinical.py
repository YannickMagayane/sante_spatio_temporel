import os
import django
import random
import logging
from datetime import date

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from analytics.models import HealthZone, ClinicalData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def distribute():
    logger.info("⚠️ Script inactif — utiliser ingest_hdx.py pour injecter les données réelles de HDX")
    logger.info("📊 Données cliniques: source unique = ingest_hdx.py")

if __name__ == "__main__":
    distribute()
