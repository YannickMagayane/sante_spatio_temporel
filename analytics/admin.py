from django.contrib import admin
from .models import Province, Territory, HealthZone, DiseaseAlert, Prediction, ClinicalData, ClimateData


@admin.register(Province)
class ProvinceAdmin(admin.ModelAdmin):
    """Admin view for Province model."""
    list_display = ('code', 'name', 'get_territory_count', 'get_health_zone_count')
    search_fields = ('name', 'code')
    ordering = ('name',)
    
    def get_territory_count(self, obj):
        return obj.territories.count()
    get_territory_count.short_description = "Territoires"
    
    def get_health_zone_count(self, obj):
        return HealthZone.objects.filter(territory__province=obj).count()
    get_health_zone_count.short_description = "Zones de Santé"


@admin.register(Territory)
class TerritoryAdmin(admin.ModelAdmin):
    """Admin view for Territory model."""
    list_display = ('code', 'name', 'province', 'population', 'get_health_zone_count')
    list_filter = ('province',)
    search_fields = ('name', 'code', 'province__name')
    ordering = ('province', 'name')
    
    def get_health_zone_count(self, obj):
        return obj.health_zones.count()
    get_health_zone_count.short_description = "Zones de Santé"


@admin.register(HealthZone)
class HealthZoneAdmin(admin.ModelAdmin):
    """Admin view for HealthZone model."""
    list_display = ('code', 'name', 'territory', 'province_display', 'population')
    list_filter = ('territory__province', 'territory')
    search_fields = ('name', 'code', 'territory__name')
    ordering = ('territory__province', 'territory', 'name')
    
    def province_display(self, obj):
        return obj.province
    province_display.short_description = "Province"


@admin.register(ClinicalData)
class ClinicalDataAdmin(admin.ModelAdmin):
    """Admin view for ClinicalData model."""
    list_display = ('health_zone', 'disease_name', 'cases_count', 'deaths_count', 'month')
    list_filter = ('disease_name', 'month')
    search_fields = ('health_zone__name', 'disease_name')
    ordering = ('-month', 'disease_name')


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    """Admin view for Prediction model."""
    list_display = ('health_zone', 'disease_name', 'predicted_cases', 'prediction_date', 'created_at')
    list_filter = ('disease_name', 'prediction_date')
    search_fields = ('health_zone__name', 'disease_name')
    ordering = ('-prediction_date',)


@admin.register(DiseaseAlert)
class DiseaseAlertAdmin(admin.ModelAdmin):
    """Admin view for DiseaseAlert model."""
    list_display = ('title', 'platform', 'health_zone', 'detected_at', 'is_approved')
    list_filter = ('platform', 'is_approved')
    search_fields = ('title', 'content')
    ordering = ('-detected_at',)


@admin.register(ClimateData)
class ClimateDataAdmin(admin.ModelAdmin):
    """Admin view for ClimateData model."""
    list_display = ('health_zone', 'month', 'temperature', 'precipitation', 'humidity')
    list_filter = ('month',)
    search_fields = ('health_zone__name',)
    ordering = ('-month',)
