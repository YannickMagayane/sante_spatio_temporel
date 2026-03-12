#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour fusionner les deux fichiers de documentation en un seul document Word
avec correction des sources de données et informations Docker Compose
"""

from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
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

def add_colored_box(doc, title, content_lines, bg_color=(230, 245, 250)):
    """Ajouter un encadré coloré pour les sections importantes"""
    table = doc.add_table(rows=1, cols=1)
    table.autofit = False
    table.allow_autofit = False
    
    # Style la cellule
    cell = table.rows[0].cells[0]
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), f'{bg_color[0]:02x}{bg_color[1]:02x}{bg_color[2]:02x}')
    tcPr.append(shd)
    
    # Contenu
    title_para = cell.paragraphs[0]
    title_run = title_para.add_run(f"📌 {title}")
    title_run.bold = True
    title_run.font.size = Pt(12)
    
    for line in content_lines:
        if line.strip():
            p = cell.add_paragraph(line, style='List Bullet')
            p.paragraph_format.left_indent = Inches(0.2)

def add_formatted_text(doc, text):
    """Parser et ajouter du texte formaté"""
    lines = text.split('\n')
    
    for line in lines:
        if not line.strip():
            continue
            
        # Détecter les en-têtes
        if '=' * 10 in line or '-' * 10 in line:
            continue
        
        if line.startswith('## '):
            add_heading_with_style(doc, line[3:].strip(), level=2)
        elif line.startswith('### '):
            add_heading_with_style(doc, line[4:].strip(), level=3)
        elif line.strip():
            p = doc.add_paragraph(line)
            p.paragraph_format.space_after = Pt(6)

def add_data_sources_section(doc):
    """Ajouter une section sur les sources de données"""
    doc.add_heading('SOURCES DE DONNÉES ET ARCHITECTURE', 1)
    
    doc.add_heading('Sources de Données', level=2)
    
    sources = [
        ('🏥 Données Sanitaires (Sanitaires)', 'HDX (Humanitarian Data Exchange) - OCHA | Chemin: data/sanitaires | Format: CSV/XLSX'),
        ('🗺️ Données Géospatiales', 'HDX - Geoboundaries | Chemin: data/raw_hdx | Format: GeoJSON'),
        ('🌍 Données Climatiques', 'Copernicus Climate Data Store (CDS) | Chemin: data/climat | Format: NetCDF/CSV'),
        ('📊 Données Traitées', 'Post-traitement local | Chemin: data/processed | Format: CSV'),
    ]
    
    for title, desc in sources:
        p = doc.add_paragraph(title, style='List Bullet')
        p.paragraph_format.left_indent = Inches(0.3)
        sub = doc.add_paragraph(desc, style='List Bullet 2')
        sub.paragraph_format.left_indent = Inches(0.6)
    
    doc.add_heading('Architecture Docker Compose', level=2)
    
    architecture_text = """Le système déploie 4 services principaux via Docker Compose:

1. **Traefik** (Reverse Proxy)
   - Port 8000: Application Web
   - Port 8081: Dashboard Traefik
   - Routage automatique des requêtes

2. **PostgreSQL** (Base de données)
   - Version 16-alpine
   - Stockage persistant: postgres_data volume
   - Configuration via .env

3. **App** (Django + ML)
   - Build: ./compose/app/Dockerfile
   - Volumes montés:
     - .:/app (code source)
     - ./compose/cle_api/.cdsapirc:/root/.cdsapirc (Copernicus API key)
   - Exécute scripts (train_models.py, auto_sync.py, etc.)

4. **Réseau** (web)
   - Communication interne entre services
   - Isolation de sécurité
"""
    
    for line in architecture_text.split('\n'):
        if line.strip():
            if line.startswith('**'):
                p = doc.add_paragraph(line, style='List Number')
            else:
                p = doc.add_paragraph(line)
            p.paragraph_format.space_after = Pt(4)
    
    doc.add_heading('Workflow de Synchronisation des Données', level=2)
    
    workflow = [
        'auto_sync.py vérifie les mises à jour HDX toutes les X heures',
        'Compare hash fichiers locaux vs HDX (économise bandwidth)',
        'Télécharge seulement les fichiers GeoJSON et CSV modifiés',
        'Alimente data/sanitaires et data/raw_hdx',
        'Métadonnées timestampées dans sync_metadata.json',
        'Logs détaillés pour monitoring',
    ]
    
    for step in workflow:
        p = doc.add_paragraph(step, style='List Bullet')
        p.paragraph_format.space_after = Pt(4)

def main():
    print("📄 Création du document Word fusionné (v2 améliorée)...\n")
    
    # Créer un nouveau document
    doc = Document()
    
    # Titre principal
    title = doc.add_heading('DOCUMENTATION COMPLÈTE — MOTEUR D\'INTELLIGENCE ARTIFICIELLE', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    subtitle = doc.add_paragraph('Spatio-Temporelle pour la Prédiction d\'Épidémies en RDC')
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle_format = subtitle.runs[0]
    subtitle_format.italic = True
    subtitle_format.font.size = Pt(14)
    
    version_date = doc.add_paragraph('Version 2 — Mars 2026 | Docker Compose Architecture')
    version_date.alignment = WD_ALIGN_PARAGRAPH.CENTER
    version_format = version_date.runs[0]
    version_format.font.size = Pt(11)
    version_format.font.color.rgb = RGBColor(100, 100, 100)
    
    doc.add_paragraph()  # Spacing
    
    # Table des matières
    doc.add_heading('TABLE DES MATIÈRES', level=2)
    toc_items = [
        '1. Sources de Données et Architecture',
        '2. Mise à Jour Majeure v2 (Nouveautés)',
        '3. Documentation Scientifique Complète',
        '4. Conclusion et Prochaines Étapes',
    ]
    for item in toc_items:
        p = doc.add_paragraph(item, style='List Bullet')
    
    doc.add_page_break()
    
    # Section 1: Sources de données
    print("✅ Ajout de la section Sources de Données...")
    add_data_sources_section(doc)
    
    doc.add_page_break()
    
    # Section 2: Mise à jour v2
    print("✅ Ajout de la section Mise à Jour v2...")
    doc.add_heading('PARTIE 1 — MISE À JOUR MAJEURE v2 (Mars 2026)', 1)
    doc.add_paragraph('Les améliorations majeures apportées au système pour optimiser '
                      'la robustesse et l\'efficacité prédictive dans un environnement Docker.')
    
    update_text = read_text_file(DOC_UPDATE)
    update_text = update_text.replace('=' * 80, '').replace('-' * 80, '')
    add_formatted_text(doc, update_text)
    
    doc.add_page_break()
    
    # Section 3: Documentation scientifique
    print("✅ Ajout de la section Documentation Scientifique...")
    doc.add_heading('PARTIE 2 — DOCUMENTATION SCIENTIFIQUE COMPLÈTE', 1)
    doc.add_paragraph('Détails mathématiques, architectures de modèles et explications '
                      'techniques pour chercheurs et implémenteurs.')
    
    stats_text = read_text_file(DOC_STATS)
    stats_text = stats_text.replace('=' * 80, '').replace('-' * 80, '')
    add_formatted_text(doc, stats_text)
    
    # Conclusion finale
    doc.add_page_break()
    doc.add_heading('CONCLUSION — SYNTHÈSE GÉNÉRALE', 1)
    
    conclusion_sections = {
        'VISION GLOBALE': [
            'Système entièrement conteneurisé avec Docker Compose',
            'Données synchronisées automatiquement depuis HDX (OCHA) et Copernicus',
            'Architecture microservices: Traefik, PostgreSQL, Django App',
            '4 groupes de maladies consolidés pour prédictions robustes',
            'Zéro prédictions négatives garanties (3 niveaux de failsafe)',
        ],
        'SOURCES DE DONNÉES': [
            'Données Sanitaires: HDX/OCHA → data/sanitaires/',
            'Données Géospatiales: HDX Geoboundaries → data/raw_hdx/',
            'Données Climatiques: Copernicus CDS (credentials .cdsapirc) → data/climat/',
            'Synchronisation intelligente: auto_sync.py (hash-based)',
        ],
        'ARCHITECTURE DÉPLOIEMENT': [
            'Orchestration: docker-compose.yml (services interconnectés)',
            'Reverse Proxy: Traefik (port 8000 web, 8081 dashboard)',
            'Base de données: PostgreSQL 16-alpine avec persistence',
            'Application: Django + TensorFlow dans conteneur',
        ],
        'ÉTAT DE PRODUCTION': [
            '✅ Code refactorisé pour 4 groupes consolidés',
            '✅ Documentation complète (scientifique + opérationnelle)',
            '✅ Containerisé et prêt pour déploiement cloud',
            '✅ Données synchronisées depuis sources officielles',
        ],
    }
    
    for section_title, bullets in conclusion_sections.items():
        doc.add_heading(section_title, level=2)
        for bullet in bullets:
            p = doc.add_paragraph(bullet, style='List Bullet')
            p.paragraph_format.space_after = Pt(4)
    
    # Sauvegarder
    doc.save(OUTPUT_FILE)
    print(f"\n✨ Document créé avec succès !")
    print(f"📁 Fichier: {OUTPUT_FILE}")
    
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
