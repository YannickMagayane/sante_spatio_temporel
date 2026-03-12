from django.db import models
from django.utils.translation import gettext_lazy as _

class Province(models.Model):
    """
    Modèle pour les Provinces de la RDC (26 provinces).
    """
    code = models.CharField(_("Code Province"), max_length=10, unique=True)
    name = models.CharField(_("Nom de la Province"), max_length=255)
    
    class Meta:
        verbose_name = _("Province")
        verbose_name_plural = _("Provinces")
        ordering = ['name']

    def __str__(self):
        return self.name


class Territory(models.Model):
    """
    Modèle pour les Territoires (145 territoires).
    Un territoire appartient à une province et contient plusieurs zones de santé.
    """
    code = models.CharField(_("Code Territoire"), max_length=20, unique=True)
    name = models.CharField(_("Nom du Territoire"), max_length=255)
    province = models.ForeignKey(Province, on_delete=models.CASCADE, related_name='territories')
    latitude = models.FloatField(_("Latitude"), null=True, blank=True)
    longitude = models.FloatField(_("Longitude"), null=True, blank=True)
    geometry = models.JSONField(_("Géométrie (GeoJSON)"), null=True, blank=True)
    population = models.IntegerField(_("Population estimée"), default=0)
    
    class Meta:
        verbose_name = _("Territoire")
        verbose_name_plural = _("Territoires")
        ordering = ['province', 'name']
        unique_together = ['name', 'province']

    def __str__(self):
        return f"{self.name} ({self.province.name})"


class HealthZone(models.Model):
    """
    Modèle pour les Zones de Santé de la RDC (519 zones de santé - données HDX).
    Une zone de santé appartient à un territoire qui lui-même appartient à une province.
    """
    code = models.CharField(_("Code Pcode"), max_length=50, unique=True)
    name = models.CharField(_("Nom"), max_length=255)
    territory = models.ForeignKey(Territory, on_delete=models.CASCADE, related_name='health_zones', null=True, blank=True)
    
    latitude = models.FloatField(_("Latitude"), null=True, blank=True)
    longitude = models.FloatField(_("Longitude"), null=True, blank=True)
    geometry = models.JSONField(_("Géométrie (GeoJSON)"), null=True, blank=True)
    population = models.IntegerField(_("Population estimée"), default=0)

    class Meta:
        verbose_name = _("Zone de Santé")
        verbose_name_plural = _("Zones de Santé")
        ordering = ['territory__province', 'territory', 'name']

    def __str__(self):
        return f"{self.name}"
    
    @property
    def province(self):
        """Accès rapide à la province parente"""
        return self.territory.province if self.territory else None

class DiseaseAlert(models.Model):
    """
    Alertes récupérées via Scraping (OMS, ReliefWeb, Twitter).
    """
    PLATFORM_CHOICES = [
        ('oms', 'OMS'),
        ('reliefweb', 'ReliefWeb'),
        ('twitter', 'Twitter (Nitter)'),
    ]

    title = models.TextField(_("Titre/Texte"))
    content = models.TextField(_("Contenu complet"), blank=True)
    source_url = models.URLField(_("URL source"), max_length=500)
    platform = models.CharField(_("Plateforme"), max_length=20, choices=PLATFORM_CHOICES)
    detected_at = models.DateTimeField(_("Date du signalement"))
    scraped_at = models.DateTimeField(auto_now_add=True)
    
    # Géolocalisation
    health_zone = models.ForeignKey(HealthZone, on_delete=models.SET_NULL, null=True, blank=True)
    confidence_score = models.FloatField(_("Score de confiance"), default=0.0)
    
    is_approved = models.BooleanField(_("Approuvé par l'IA/Expert"), default=False)

    class Meta:
        verbose_name = _("Alerte Maladie")
        verbose_name_plural = _("Alertes Maladies")

class Prediction(models.Model):
    """
    Résultats des modèles LSTM/CNN.
    """
    THREAT_TYPE_CHOICES = [
        ('epidemic', 'Épidémie (Contagieuse)'),
        ('nutritional_crisis', 'Crise Nutritionnelle'),
        ('coinfection', 'Comorbidité (Épidémie + Malnutrition)'),
    ]
    
    health_zone = models.ForeignKey(HealthZone, on_delete=models.CASCADE)
    disease_name = models.CharField(_("Maladie"), max_length=100)
    threat_type = models.CharField(_("Type de Menace"), max_length=20, choices=THREAT_TYPE_CHOICES, default='epidemic')
    predicted_cases = models.FloatField(_("Cas prédits"))
    prediction_date = models.DateField(_("Date de prédiction"))
    influence_factors = models.JSONField(_("Facteurs SHAP"), null=True)
    propagation_rate = models.FloatField(_("Vitesse de Propagation (%)"), default=0.0)
    contextual_risk = models.JSONField(_("Risques Contextuels"), null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('health_zone', 'disease_name', 'prediction_date')

class ClinicalData(models.Model):
    """
    Données sanitaires officielles (DPS, PNLP, INRB).
    """
    health_zone = models.ForeignKey(HealthZone, on_delete=models.CASCADE)
    disease_name = models.CharField(_("Maladie"), max_length=100)
    cases_count = models.IntegerField(_("Nombre de cas"))
    deaths_count = models.IntegerField(_("Nombre de décès"), default=0)
    month = models.DateField(_("Mois des données"))
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('health_zone', 'disease_name', 'month')

class ClimateData(models.Model):
    """
    Données ERA5-Land par zone de santé.
    """
    health_zone = models.ForeignKey(HealthZone, on_delete=models.CASCADE)
    month = models.DateField()
    temperature = models.FloatField(_("Température moyenne"))
    precipitation = models.FloatField(_("Précipitations totales"))
    humidity = models.FloatField(_("Humidité relative"), null=True)
    
    class Meta:
        unique_together = ('health_zone', 'month')


class TaskRun(models.Model):
    """
    Modèle pour tracker l'exécution des tâches longues (entraînement, ingestion HDX).
    """
    TASK_TYPE_CHOICES = [
        ('train', 'Entraînement des modèles'),
        ('ingest', 'Ingestion de données HDX'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'En attente'),
        ('running', 'En cours'),
        ('completed', 'Complété'),
        ('failed', 'Échoué'),
    ]
    
    task_type = models.CharField(_("Type de tâche"), max_length=20, choices=TASK_TYPE_CHOICES)
    status = models.CharField(_("Statut"), max_length=20, choices=STATUS_CHOICES, default='pending')
    progress = models.IntegerField(_("Progrès (%)"), default=0)
    
    started_at = models.DateTimeField(_("Démarrage"), null=True, blank=True)
    completed_at = models.DateTimeField(_("Fin"), null=True, blank=True)
    
    output_log = models.TextField(_("Journal de sortie"), blank=True)
    error_log = models.TextField(_("Journal d'erreur"), blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = _("Exécution de tâche")
        verbose_name_plural = _("Exécutions de tâches")
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.get_task_type_display()} - {self.get_status_display()}"
