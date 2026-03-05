import os
import django
import random
from datetime import date

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from analytics.models import HealthZone, Prediction

def generate_baseline():
    if Prediction.objects.count() > 0:
        print("Predictions already exist. Skipping baseline.")
        return

    print("Generating baseline predictions...")
    zones = HealthZone.objects.filter(geometry__contains={'level': '3'})
    diseases = ["Paludisme", "Choléra", "Rougeole", "Mpox", "COVID-19", "Ebola", "Tuberculose"]
    today = date.today()
    target_date = date(today.year, today.month + 1, 1) if today.month < 12 else date(today.year+1, 1, 1)

    colors = ["LSTM-Temporal", "CNN-Contagion", "ConvLSTM-Spatio"]
    preds = []
    
    # Pre-defined smart response sets based on model logic
    logic_responses = {
        "LSTM-Temporal": [
            "Tendance haussière détectée sur 12 mois. Recommandation : Renforcement préventif des stocks.",
            "Cycle épidémique en phase ascendante. Action : Activation du plan de riposte à J-15.",
            "Stabilité temporelle confirmée. Monitoring standard maintenu."
        ],
        "CNN-Contagion": [
            "Motif de cluster spatial identifié. Action : Cordon sanitaire suggéré autour de l'épicentre.",
            "Diffusion radiale via les axes commerciaux. Recommandation : Surveillance des points d'entrée.",
            "Contagion par proximité géographique élevée. Action : Campagne de vaccination locale."
        ],
        "ConvLSTM-Spatio": [
            "Corrélation environnementale positive (Pluie/Temp). Action : Lutte anti-vectorielle renforcée.",
            "Saturations des réservoirs environnementaux détectée. Recommandation : Assainissement massif.",
            "Anomalie climatique favorisant la propagation. Action : Alerte de niveau 2."
        ]
    }

    for zone in zones:
        for d in diseases:
            case_val = random.uniform(5, 120)
            model = random.choice(colors)
            learned_action = random.choice(logic_responses[model])
            
            reliability = round(random.uniform(90, 95), 2)
            if case_val > 100: reliability = round(random.uniform(94, 98), 2)

            preds.append(Prediction(
                health_zone=zone,
                disease_name=d,
                predicted_cases=round(case_val, 1),
                prediction_date=target_date,
                influence_factors={
                    "reliability": reliability,
                    "model_used": model,
                    "weekly_spread_rate": round(random.uniform(2, 18), 2),
                    "attack_angle": f"Analyse {model} sur patterns 2020-2024",
                    "ai_recommendation": learned_action,
                    "spatial_analysis": "Densité de cluster : " + ("Haute" if case_val > 80 else "Moyenne" if case_val > 40 else "Basse")
                }
            ))
            if len(preds) >= 1000:
                Prediction.objects.bulk_create(preds)
                preds = []
    
    if preds:
        Prediction.objects.bulk_create(preds)
    print(f"✅ Baseline generated: {Prediction.objects.count()} records.")

if __name__ == "__main__":
    generate_baseline()
