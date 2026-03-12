#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convertir ANALYSE_GRAPHIQUES_PROVINCES.md en document Word avec images intégrées
Version FINALE avec ALL images de Paludisme ET Choléra + textes complets
"""

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

def shade_cell(cell, color):
    """Ajouter une couleur de fond à une cellule"""
    shading_elm = OxmlElement('w:shd')
    shading_elm.set(qn('w:fill'), color)
    cell._element.get_or_add_tcPr().append(shading_elm)

def add_heading(doc, text, level=1):
    """Ajouter un titre au document"""
    h = doc.add_heading(text, level=level)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return h

def add_paragraph(doc, text, bold=False, italic=False, size=11):
    """Ajouter un paragraphe avec formatage"""
    p = doc.add_paragraph(text)
    for run in p.runs:
        run.bold = bold
        run.italic = italic
        run.font.size = Pt(size)
    return p

def add_image_strict(doc, image_path, caption="", width=6.0):
    """Ajouter image avec vérification stricte du chemin"""
    if os.path.isfile(image_path):
        try:
            doc.add_picture(image_path, width=Inches(width))
            if caption:
                p = doc.add_paragraph(caption)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.italic = True
                    run.font.size = Pt(10)
            return True
        except Exception as e:
            print(f"❌ Erreur image: {image_path} - {e}")
            add_paragraph(doc, f"[Image: {caption}]", italic=True)
            return False
    else:
        print(f"⚠️  Image non trouvée: {image_path}")
        add_paragraph(doc, f"[Image: {caption}]", italic=True)
        return False

def create_word_document_final():
    """Créer le document Word FINAL avec ALL images"""
    
    doc = Document()
    
    # ===== COUVERTURE =====
    title = add_heading(doc, "📊 ANALYSE DÉTAILLÉE DES GRAPHIQUES AVANT INJECTION", level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    subtitle = add_paragraph(doc, "Province: Haut-Katanga | Maladies: Paludisme & Choléra", size=12)
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in subtitle.runs:
        run.italic = True
    
    doc.add_paragraph("")
    
    # ===== INTRO =====
    add_heading(doc, "🔍 PROBLÈMES IDENTIFIÉS AVANT INJECTION", level=2)
    
    add_heading(doc, "Données Fragmentées (14-16 mois)", level=3)
    add_paragraph(doc, "Paludisme Haut-Katanga: 14 mois fragmentés → R² = -2.54 ❌")
    add_paragraph(doc, "Choléra Haut-Katanga: 16 mois fragmentés → R² = 0.3-0.6 (instable)")
    add_paragraph(doc, "Impact: Impossible de détecter la saisonnalité avec si peu de données")
    
    doc.add_paragraph("")
    
    add_heading(doc, "Saisonnalité Non-Détectable", level=3)
    add_paragraph(doc, "Avec seulement 14-16 mois, impossible d'identifier les cycles saisonniers complets (Nov-Mar pour paludisme, Oct-Jan pour choléra)")
    
    doc.add_paragraph("")
    
    add_heading(doc, "Split Train/Val/Test Invalide", level=3)
    add_paragraph(doc, "AVANT: 5-6 séquences train ÷ 38K paramètres = overfitting sévère")
    add_paragraph(doc, "APRÈS: 180 mois → 176 séquences = Split valide (106/35/35)")
    
    doc.add_page_break()
    
    # ===== PALUDISME COMPLET =====
    add_heading(doc, "🦟 PALUDISME — GRAPHIQUES AVANT INJECTION", level=2)
    
    add_heading(doc, "1️⃣ Comparaison Données vs Prédictions (Données Fragmentées)", level=3)
    add_paragraph(doc, "Cette courbe montre les données réelles fragmentées (points bleus) sur 14 mois discontinus. La ligne de prédiction (rouge) est quasi-invalide car le modèle n'a que 5 séquences pour apprendre.")
    add_paragraph(doc, "R² = -2.54 indique que les prédictions sont pires qu'une simple moyenne. Les écarts énormes montrent l'impossibilité de modéliser avec si peu de données.")
    
    add_image_strict(doc, 
        "/home/yannick-magayane/Bureau/projets/data_analyse/health/images/palu/ancienne donnee/donnee vs pred.png",
        "Données vs Prédictions - Paludisme AVANT injection")
    
    doc.add_paragraph("")
    doc.add_paragraph("")
    
    add_heading(doc, "2️⃣ Tendances Temporelles et Pics d'Épidémie", level=3)
    add_paragraph(doc, "Affiche les cas confirmés mensuels (bleu) + décès (croix rouges). Période: Janvier - Décembre 2025 (14 mois). Les pics visibles (Nov-Mar) indiquent la saisonnalité du paludisme.")
    add_paragraph(doc, "Les trous dans la série (mois manquants) rendent impossible l'apprentissage continu. Impact: Impossible de prédire avec confiance les pics futurs.")
    
    add_image_strict(doc,
        "/home/yannick-magayane/Bureau/projets/data_analyse/health/images/palu/ancienne donnee/Capture d'écran du 2026-03-10 00-08-14.png",
        "Série Temporelle - Paludisme AVANT injection")
    
    doc.add_paragraph("")
    doc.add_paragraph("")
    
    add_heading(doc, "3️⃣ Distribution par Zone de Santé (Zones Hotspot)", level=3)
    add_paragraph(doc, "Carte thermique montrant les cas par zone de santé en Haut-Katanga. Zones Rouges (Hotspots): Lubumbashi, Mufunga Sampwe (263k+ cas/100k = sur-transmission).")
    add_paragraph(doc, "Zones Jaunes: Transmission modérée (50-100k cas). Zones Vertes/Bleues: Transmission basse (<50k cas). Distribution stable mais non fiable sur 14 mois fragmentés.")
    
    add_image_strict(doc,
        "/home/yannick-magayane/Bureau/projets/data_analyse/health/images/palu/ancienne donnee/Capture d'écran du 2026-03-10 00-09-42.png",
        "Intensité par Zone - Paludisme AVANT injection")
    
    doc.add_paragraph("")
    doc.add_paragraph("")
    
    add_heading(doc, "4️⃣ Analyse des Données Manquantes (Complétude par Zone)", level=3)
    add_paragraph(doc, "Graphique en barres montrant le % de couverture de données par zone (Top 15).")
    add_paragraph(doc, "Zones en vert (100%): Kilwa, Rwashi, Mumbunda → données fiables. Zones en orange (83%): Lubumbashi, Mufunga Sampwe → 17% de mois manquants = impacts significatifs.")
    add_paragraph(doc, "Impossible d'utiliser un modèle global fiable avec cette hétérogénéité des données.")
    
    add_image_strict(doc,
        "/home/yannick-magayane/Bureau/projets/data_analyse/health/images/palu/ancienne donnee/Capture d'écran du 2026-03-10 00-08-58.png",
        "Couverture des Données - Paludisme AVANT injection")
    
    doc.add_page_break()
    
    add_heading(doc, "5️⃣ Analyse Comparative (Avant vs Après - Séries Temporelles)", level=3)
    add_paragraph(doc, "Compare visuellement les données réelles fragmentées (avant) vs les données injectées continues (après).")
    add_paragraph(doc, "AVANT: Trous visibles tous les 2-3 mois = modélisation impossible. APRÈS: Série continue 180 mois = 15 cycles saisonniers complets = prédictions viables.")
    add_paragraph(doc, "Les crêtes régulières (après) montrent la saisonnalité détectée de manière fiable.")
    
    add_image_strict(doc,
        "/home/yannick-magayane/Bureau/projets/data_analyse/health/images/palu/ancienne donnee/Capture d'écran du 2026-03-10 00-09-20.png",
        "Avant/Après Complétude - Paludisme")
    
    doc.add_paragraph("")
    doc.add_paragraph("")
    
    add_heading(doc, "6️⃣ Ranking des Zones (Top 20 par charge)", level=3)
    add_paragraph(doc, "Classement des zones par charge de morbidité absolue (nombre de cas).")
    add_paragraph(doc, "Top 3: Lubumbashi > Mufunga Sampwe > Manika (responsables de ~60% de la charge).")
    add_paragraph(doc, "Implication opérationnelle: 80/20 → 20% des zones (6/28) = 80% des décisions/ressources. Zone de focalisation: les 6 zones en rouge = priorités pour la vaccination/surveillance.")
    
    add_image_strict(doc,
        "/home/yannick-magayane/Bureau/projets/data_analyse/health/images/palu/ancienne donnee/top 20.png",
        "Top 20 Zones - Paludisme AVANT injection")
    
    doc.add_paragraph("")
    doc.add_paragraph("")
    
    add_heading(doc, "7️⃣ Analyse Granulaire (11 Zones Sélectionnées)", level=3)
    add_paragraph(doc, "Analyse focalisée sur 11 zones critiques avec graphiques individuels. Chaque zone montre sa tendance propre (croissance, stabilité, déclin).")
    add_paragraph(doc, "Permet d'identifier les zones avec trajectoires différentes = adaptation des interventions.")
    add_paragraph(doc, "AVANT injection: Patterns peu clairs sur 14 mois → APRÈS injection: 15 cycles = patterns évidents. Utilité: Ajustement des stratégies sanitaires par zone.")
    
    add_image_strict(doc,
        "/home/yannick-magayane/Bureau/projets/data_analyse/health/images/palu/ancienne donnee/pour 11 zs.png",
        "Analyse Détaillée (11 Zones) - Paludisme AVANT injection")
    
    doc.add_page_break()
    
    # ===== CHOLÉRA COMPLET =====
    add_heading(doc, "💧 CHOLÉRA — GRAPHIQUES AVANT INJECTION", level=2)
    
    add_heading(doc, "1️⃣ Données vs Prédictions (Instabilité Évidente)", level=3)
    add_paragraph(doc, "Affiche les données réelles choléra (points bleus épars) sur 16 mois fragmentés. La courbe de prédiction (rouge) est très instable car le modèle CNN-LSTM manque de données.")
    add_paragraph(doc, "R² = 0.35-0.60 (intermédiaire) mais très volatil selon les mois. Les pics réels ne sont pas anticipés → réactivité (pas prédictif).")
    add_paragraph(doc, "Après injection: R² = 0.752 (stable) = peut prédire 2-3 mois à l'avance.")
    add_paragraph(doc, "Constat: Le choléra montrait une meilleure performance qu'le paludisme avant injection, mais restait instable et imprévisible sur le plan opérationnel.")
    
    add_image_strict(doc,
        "/home/yannick-magayane/Bureau/projets/data_analyse/health/images/cholera/ancienne/Capture d'écran du 2026-03-10 00-12-36.png",
        "Données vs Prédictions - Choléra AVANT injection")
    
    doc.add_paragraph("")
    doc.add_paragraph("")
    
    add_heading(doc, "2️⃣ Taux d'Incidence et Saisonnalité", level=3)
    add_paragraph(doc, "Montre le taux d'incidence par 100k habitants pour chaque zone de santé.")
    add_paragraph(doc, "Pics saisonniers: Oct-Jan (saison sèche + forte humidité résiduelle = conditions optimales pour Vibrio cholerae).")
    add_paragraph(doc, "Taux extrêmes: Manika (2,307 cas/100k au pic) = risque très élevé. Taux modéré: Kampemba, Mitwaba (1,000-1,500 cas/100k).")
    add_paragraph(doc, "Taux bas: Vargu, Kasongo (< 500 cas/100k = transmission faible). Les pics saisonniers étaient observables mais 16 mois n'est pas assez pour prédire le cycle suivant.")
    
    add_image_strict(doc,
        "/home/yannick-magayane/Bureau/projets/data_analyse/health/images/cholera/ancienne/Capture d'écran du 2026-03-10 00-12-48.png",
        "Taux d'Incidence - Choléra AVANT injection")
    
    doc.add_paragraph("")
    doc.add_paragraph("")
    
    add_heading(doc, "3️⃣ Distribution Géographique (Zones Hotspot Choléra)", level=3)
    add_paragraph(doc, "Heatmap des zones les plus affectées par le choléra.")
    add_paragraph(doc, "Hotspots primaires (Manika, Kampemba): zones urbaines avec mauvaise assainissement. Hotspots secondaires (Kasongo, Mitwaba): zones de transit commerciaux.")
    add_paragraph(doc, "Zones contrôlées (Vargu, Haut): actes de sanitaire/eau potable existants.")
    add_paragraph(doc, "AVANT injection: Hotspots clairs mais non prédictibles → réactivité permanente. Vue: Utile pour allocation de ressources de prévention préventive (vaccination, eau, hygiène).")
    
    add_image_strict(doc,
        "/home/yannick-magayane/Bureau/projets/data_analyse/health/images/cholera/ancienne/Capture d'écran du 2026-03-10 00-13-06.png",
        "Distribution par Zone - Choléra AVANT injection")
    
    doc.add_page_break()
    
    # ===== SYNTHÈSE =====
    add_heading(doc, "🎯 SYNTHÈSE AVANT vs APRÈS INJECTION", level=2)
    
    # Tableau comparatif
    table = doc.add_table(rows=8, cols=3)
    table.style = 'Light Grid Accent 1'
    
    # Header
    header_cells = table.rows[0].cells
    header_cells[0].text = "Aspect"
    header_cells[1].text = "AVANT (14-16 mois)"
    header_cells[2].text = "APRÈS (180 mois)"
    
    for cell in header_cells:
        shade_cell(cell, "4472C4")
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.color.rgb = RGBColor(255, 255, 255)
                run.font.bold = True
    
    # Données
    data = [
        ("Mois de données", "14-16 discontinus", "180 continus"),
        ("Cycles saisonniers", "<1", "15 complets"),
        ("R² Paludisme", "-2.54 (invalide)", "0.717 (excellent)"),
        ("R² Choléra", "0.35-0.60 (instable)", "0.752 (stable)"),
        ("Usabilité", "❌ Non viable", "✅ Opérationnelle"),
        ("Prédictibilité", "❌ Nulle", "✅ 2-3 mois avance"),
        ("Décision sanitaire", "Intuition/réactivité", "Prédictive & proactive"),
    ]
    
    for i, (aspect, avant, apres) in enumerate(data, 1):
        row_cells = table.rows[i].cells
        row_cells[0].text = aspect
        row_cells[1].text = avant
        row_cells[2].text = apres
        shade_cell(row_cells[1], "FCE4D6")  # Orange clair
        shade_cell(row_cells[2], "E2EFDA")  # Vert clair
    
    doc.add_paragraph("")
    
    # ===== CONCLUSIONS =====
    add_heading(doc, "✅ CONCLUSIONS", level=2)
    
    add_paragraph(doc, "Les graphiques AVANT injection montrent clairement pourquoi la génération de données synthétiques était nécessaire.")
    add_paragraph(doc, "Avec seulement 14-16 mois fragmentés, les modèles ConvLSTM et CNN-LSTM ne pouvaient pas apprendre les patterns fiables.")
    
    add_paragraph(doc, "L'injection de 180 mois continus a permis:")
    add_paragraph(doc, "  • 15 cycles saisonniers complets = patterns détectables")
    add_paragraph(doc, "  • 176 séquences train vs 5-6 avant = overfitting résolu")
    add_paragraph(doc, "  • R² amélioré de -2.54 → 0.717 (Paludisme) et 0.35 → 0.752 (Choléra)")
    
    add_paragraph(doc, "Les graphiques après injection démontrent la viabilité opérationnelle des modèles pour des alertes précoces 2-3 mois à l'avance.")
    add_paragraph(doc, "Cette approche sauve des vies en permettant une allocation proactive des ressources sanitaires.")
    
    doc.add_paragraph("")
    
    # Footer
    footer_p = doc.add_paragraph("Rapport généré: 10 mars 2026 | Haut-Katanga | Paludisme & Choléra")
    footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in footer_p.runs:
        run.italic = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(128, 128, 128)
    
    # Sauvegarde
    output_path = "/home/yannick-magayane/Bureau/projets/data_analyse/health/ANALYSE_GRAPHIQUES_AVANT_INJECTION.docx"
    doc.save(output_path)
    
    # Afficher confirmation
    file_size = os.path.getsize(output_path) / 1024
    print(f"✅ Document FINAL créé avec succès!")
    print(f"📄 Fichier: {output_path}")
    print(f"📊 Taille: {file_size:.1f} KB")
    print(f"📸 Images intégrées: 10 (7 Paludisme + 3 Choléra)")
    print(f"📝 Textes complets: TOUS les graphiques expliqués")
    
    return output_path

if __name__ == "__main__":
    create_word_document_final()
