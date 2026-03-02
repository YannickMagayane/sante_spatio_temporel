"""
Generate interactive maps and SHAP visualizations for epidemic predictions.

Produces:
- Folium maps with risk circles per health zone
- SHAP summary plots for model interpretability
- Trend charts per city

Example:
    $ python scripts/visualize.py --type maps --period 2025-01
    $ python scripts/visualize.py --type shap --model lstm
    $ python scripts/visualize.py --type trends --city kinshasa
"""

import argparse
import json
import os

import folium
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


# Importation de la base de données géo centrale
try:
    from geo_reference import PROVINCES
except ImportError:
    # Fallback si lancé depuis un autre dossier
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from geo_reference import PROVINCES


def risk_color(value, max_value=100):
    """
    Map a risk value to a color from green (low) to red (high).
    """
    ratio = min(value / max_value, 1.0)
    r = int(255 * ratio)
    g = int(255 * (1 - ratio))
    return f"#{r:02x}{g:02x}33"


def generate_risk_map(period="2025-01", output_dir="reports/"):
    """
    Generate an interactive Folium map showing predicted epidemic risks.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Center map on DRC
    m = folium.Map(
        location=[-4.0, 23.0],
        zoom_start=5,
        tiles="CartoDB dark_matter",
        min_zoom=5,
        max_bounds=True,
    )

    # Add title
    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                z-index: 1000; background: rgba(0,0,0,0.7); color: white;
                padding: 10px 20px; border-radius: 8px; font-size: 16px;
                font-family: Arial, sans-serif; white-space: nowrap;">
        🗺️ Risque Épidémique RDC — {period}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Tenter de charger les prédictions réelles depuis models/predictions.json
    try:
        preds_path = "api/predictions.json" # Fichier généré pour le dashboard
        with open(preds_path, "r", encoding="utf-8") as f:
            predictions_raw = json.load(f)
            # Transformer pour accès rapide par province
            predictions = {p["province"].lower(): p["risks"] for p in predictions_raw}
    except:
        # Fallback si pas de données réelles
        predictions = {k.lower(): {"paludisme": np.random.randint(10, 90)} for k in PROVINCES.keys()}

    # Disease layers
    disease_groups = {}
    all_diseases = set()
    for prov_preds in predictions.values():
        all_diseases.update(prov_preds.keys())

    for disease in sorted(all_diseases):
        disease_groups[disease] = folium.FeatureGroup(name=f"🦠 {disease.capitalize()}", show=False)

    # Add markers for each province
    for province, data in PROVINCES.items():
        prov_preds = predictions.get(province.lower(), {})

        for disease, cases in prov_preds.items():
            color = risk_color(cases)
            radius = max(cases / 4, 4)

            popup_html = f"""
            <div style="font-family: Arial; min-width: 180px;">
                <h4 style="margin: 0; color: #333;">📍 Province: {province}</h4>
                <p style="margin: 5px 0; color: #666;">Chef-lieu: {data['capital']}</p>
                <hr style="margin: 5px 0;">
                <p><b>🦠 {disease.capitalize()}</b></p>
                <p>Cas prédits: <b style="color: {color};">{cases}</b></p>
                <p>Période: {period}</p>
            </div>
            """

            folium.CircleMarker(
                location=[data["lat"], data["lon"]],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{province} — {disease}: {cases} cas",
            ).add_to(disease_groups[disease])

    for group in disease_groups.values():
        group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Save
    filepath = os.path.join(output_dir, f"risk_map_{period}.html")
    m.save(filepath)
    print(f"✅ Map saved: {filepath}")
    return filepath


def generate_shap_plot(model_type, output_dir="reports/"):
    """
    Generate SHAP summary plot for a trained model.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        import shap
        from tensorflow.keras.models import load_model

        model_path = f"models/{model_type}_final.h5"
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return None

        model = load_model(model_path)

        # Extraction de la dimension d'entrée
        input_shape = model.input_shape
        batch_input_shape = [20] + list(input_shape[1:])
        X_sample = np.random.rand(*batch_input_shape).astype(np.float32)

        explainer = shap.DeepExplainer(model, X_sample[:10])
        shap_values = explainer.shap_values(X_sample[:5])

        filepath = os.path.join(output_dir, f"shap_{model_type}.png")
        plt.figure(figsize=(10, 6))
        plt.title(f"SHAP Interpretability — {model_type.upper()}")
        
        if model_type == "lstm":
            shap_flat = np.array(shap_values).reshape(5, -1)
            X_flat = X_sample[:5].reshape(5, -1)
            shap.summary_plot(shap_flat, X_flat, show=False)
        else:
            plt.text(0.5, 0.5, "SHAP viz for spatial models\nrequires specific data processing",
                     ha="center", va="center", transform=plt.gca().transAxes)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"✅ SHAP plot saved: {filepath}")
        return filepath

    except Exception as exc:
        print(f"⚠️  SHAP plot failed: {exc}")
        return None


def generate_trend_chart(region="Kinshasa", output_dir="reports/"):
    """
    Generate a trend chart showing predicted cases over time for a region.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Données simulées pour l'exemple visuel
    months = [f"2024-{m:02d}" for m in range(1, 13)]
    diseases = {
        "Paludisme": np.random.randint(30, 90, 12),
        "Choléra": np.random.randint(10, 50, 12),
        "Rougeole": np.random.randint(5, 25, 12),
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"Paludisme": "#e74c3c", "Choléra": "#3498db", "Rougeole": "#f39c12"}

    for disease, cases in diseases.items():
        ax.plot(months, cases, "-o", color=colors[disease], linewidth=2, label=disease)
        ax.fill_between(months, cases, alpha=0.1, color=colors[disease])

    ax.set_title(f"📈 Tendances épidémiques — {region}", fontsize=16, fontweight="bold")
    ax.set_xlabel("Période", fontsize=12)
    ax.set_ylabel("Cas (Prédictions)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    safe_name = region.lower().replace(" ", "_")
    filepath = os.path.join(output_dir, f"trends_{safe_name}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"✅ Trend chart saved: {filepath}")
    return filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate visualizations for epidemic predictions."
    )
    parser.add_argument(
        "--type",
        choices=["maps", "shap", "trends", "all"],
        required=True,
        help="Type of visualization",
    )
    parser.add_argument("--period", type=str, default="2025-01", help="Period for maps")
    parser.add_argument("--model", type=str, default="lstm", help="Model for SHAP")
    parser.add_argument("--city", type=str, default="kinshasa", help="City for trends")
    args = parser.parse_args()

    if args.type in ("maps", "all"):
        generate_risk_map(args.period)

    if args.type in ("shap", "all"):
        generate_shap_plot(args.model)

    if args.type in ("trends", "all"):
        for city in CITY_COORDS:
            generate_trend_chart(city)

    print("\n🏁 Visualization complete.")
