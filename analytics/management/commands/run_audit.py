from django.core.management.base import BaseCommand
from analytics.models import ClinicalData, ClimateData, DiseaseAlert, HealthZone
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Audit intelligent de cohérence des données via Isolation Forest'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("🛡️ Démarrage de l'audit de cohérence..."))
        
        # 1. Préparation des données pour l'audit
        # On croise les données sanitaires avec les données climatiques
        clinical = ClinicalData.objects.all().values('health_zone_id', 'disease_name', 'cases_count', 'month')
        if not clinical.exists():
            self.stdout.write("⚠️ Pas de données cliniques à auditer.")
            return

        df_clinical = pd.DataFrame(list(clinical))
        
        # 2. Détection d'anomalies univariée (ex: pics de cas impossibles)
        # On peut utiliser Isolation Forest sur le count
        model = IsolationForest(contamination=0.05, random_state=42)
        df_clinical['anomaly'] = model.fit_predict(df_clinical[['cases_count']])
        
        anomalies = df_clinical[df_clinical['anomaly'] == -1]
        
        for _, row in anomalies.iterrows():
            # "Si le systeme voit une suite logique dans les zone peripherique alors il approuve"
            # Ici on simule cette logique : on vérifie les ZS voisines
            zs = HealthZone.objects.get(id=row['health_zone_id'])
            self.stdout.write(self.style.WARNING(f"🚩 Incohérence détectée : {row['cases_count']} cas de {row['disease_name']} à {zs.name}"))
            
            # TODO: Logique de propagation spatiale pour validation automatique
            # if check_peripheral_logic(zs, row['disease_name']): ...
            
        self.stdout.write(self.style.SUCCESS(f"🏁 Audit terminé. {len(anomalies)} points suspects identifiés."))
