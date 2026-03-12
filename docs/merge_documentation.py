#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour fusionner les deux fichiers de documentation en un seul document Word
"""

from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os

# Chemins des fichiers
PROJECT_DIR = '/home/yannick-magayane/Bureau/projets/data_analyse/health'
DOC_STATS = os.path.join(PROJECT_DIR, 'DOCUMENTATION_METHODE_STATISTIQUE.txt')
DOC_UPDATE = os.path.join(PROJECT_DIR, 'MISE_A_JOUR_v2_Mars2026.txt')
OUTPUT_FILE = os.path.join(PROJECT_DIR, 'DOCUMENTATION_COMPLETE_v2_Mars2026.docx')

def read_text_file(filepath):
    """Lire un fichier texte"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def add_heading_with_style(doc, text, level=1):
    """Ajouter un titre avec style"""
    heading = doc.add_heading(text, level=level)
    heading.style = f'Heading {level}'
    return heading

def add_formatted_text(doc, text):
    """Parser et ajouter du texte formaté"""
    lines = text.split('\n')
    
    for line in lines:
        # Ignorer les lignes vides très évidentes
        if not line.strip():
            continue
            
        # Détecter les en-têtes (lignes avec === ou ---)
        if '=' * 10 in line or '-' * 10 in line:
            continue
        
        # Détecter les titres de niveau 2
        if line.startswith('## '):
            add_heading_with_style(doc, line[3:].strip(), level=2)
        # Détecter les titres de niveau 3
        elif line.startswith('### '):
            add_heading_with_style(doc, line[4:].strip(), level=3)
        # Paragraphes normaux
        elif line.strip():
            p = doc.add_paragraph(line)
            p.paragraph_format.space_after = Pt(6)

def main():
    print("📄 Création du document Word fusionné...\n")
    
    # Créer un nouveau document
    doc = Document()
    
    # Ajouter le titre principal
    title = doc.add_heading('DOCUMENTATION COMPLÈTE — MOTEUR D\'INTELLIGENCE ARTIFICIELLE', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    subtitle = doc.add_paragraph('Spatio-Temporelle pour la Prédiction d\'Épidémies en RDC')
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle_format = subtitle.runs[0]
    subtitle_format.italic = True
    subtitle_format.font.size = Pt(14)
    
    version_date = doc.add_paragraph('Version 2 — Mars 2026')
    version_date.alignment = WD_ALIGN_PARAGRAPH.CENTER
    version_format = version_date.runs[0]
    version_format.font.size = Pt(11)
    version_format.font.color.rgb = RGBColor(100, 100, 100)
    
    doc.add_paragraph()  # Spacing
    
    # Section 1: Mise à jour (nouveautés en premier)
    print("✅ Ajout de la section Mise à Jour (NOUVEAUTÉS)...")
    doc.add_heading('PARTIE 1 — MISE À JOUR MAJEURE v2 (Mars 2026)', 1)
    doc.add_paragraph('Les améliorations majeures apportées au système pour optimiser '
                      'la robustesse et l\'efficacité prédictive.')
    
    update_text = read_text_file(DOC_UPDATE)
    # Nettoyer les balises décoratives
    update_text = update_text.replace('=' * 80, '').replace('-' * 80, '')
    add_formatted_text(doc, update_text)
    
    doc.add_page_break()
    
    # Section 2: Documentation scientifique complète
    print("✅ Ajout de la section Documentation Scientifique...")
    doc.add_heading('PARTIE 2 — DOCUMENTATION SCIENTIFIQUE COMPLÈTE', 1)
    doc.add_paragraph('Détails mathématiques, architectures de modèles et explications '
                      'techniques pour chercheurs et implémenteurs.')
    
    stats_text = read_text_file(DOC_STATS)
    # Nettoyer les balises décoratives
    stats_text = stats_text.replace('=' * 80, '').replace('-' * 80, '')
    add_formatted_text(doc, stats_text)
    
    # Ajouter une conclusion
    doc.add_page_break()
    doc.add_heading('CONCLUSION — SYNTHÈSE GÉNÉRALE', 1)
    
    conclusion = """
Ce document unifié présente la documentation complète du moteur d'intelligence 
artificielle pour la prédiction épidémiologique en RDC.

STRUCTURE :
• PARTIE 1 explique les améliorations pratiques et opérationnelles (v2)
• PARTIE 2 détaille la théorie mathématique et les architectures de modèles

LECTEURS CIBLES :
• Utilisateurs opérationnels → Focus sur PARTIE 1
• Chercheurs/Publicateurs → Focus sur PARTIE 2
• Implémenteurs → Lire les deux parties

POINTS CLÉS RETENUS :
✓ 4 groupes consolidés au lieu de 17 maladies isolées
✓ Données manquantes gérées intelligemment (+interpolation)
✓ Zéro prédictions négatives garanties (3 niveaux failsafe)
✓ Hiérarchie de 4 modèles (TIER 1-4) adaptatifs selon volume de données
✓ Validation temporelle stricte pour prédictions fiables

ÉTAT DU SYSTÈME :
→ Production ready pour déploiement opérationnel
→ Robustesse testée et validée
→ Documentation scientifique complète pour publication
    """
    
    for line in conclusion.strip().split('\n'):
        if line.strip():
            doc.add_paragraph(line)
    
    # Sauvegarder
    doc.save(OUTPUT_FILE)
    print(f"\n✨ Document créé avec succès !")
    print(f"📁 Fichier: {OUTPUT_FILE}")
    print(f"📊 Nombre de pages: ~{len(doc.sections) * 2} (approx)")
    
    # Vérifier le fichier
    if os.path.exists(OUTPUT_FILE):
        file_size = os.path.getsize(OUTPUT_FILE) / 1024  # en KB
        print(f"💾 Taille: {file_size:.1f} KB")
        return True
    return False

if __name__ == '__main__':
    try:
        success = main()
        if success:
            print("\n✅ Fusion complète et réussie !")
        else:
            print("\n❌ Erreur lors de la création du fichier")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
