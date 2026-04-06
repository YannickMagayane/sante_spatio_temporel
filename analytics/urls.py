from django.urls import path
from .views import (
    DashboardView, api_stats, api_predictions, api_alerts,
    api_zone_detail, api_zone_propagation_path,
    TaskRunListView, launch_training_view, launch_ingest_view, launch_ingest_climate_view, task_status_api,
    training_performance_api, zone_recurrence_api, pressure_virale_api, stop_task_view
)
from .graph_views import ProvinceAnalysisView, DiseaseDetailView
from .disease_interactive_view import DiseaseInteractiveView
from .enhanced_disease_view import EnhancedDiseaseAnalysisView
from .notebook_executor import execute_notebook, list_notebooks, get_notebook_status
from .notebook_views import NotebookControlView, JupyterProxyView
from .training_api import training_metrics_api, model_tier_distribution_api, model_performance_api
from .climate_views import (
    province_climate_api,
    zone_propagation_heatmap_api,
    provincial_predictions_api,
    disease_propagation_map_api
)
from .province_data_api import province_dashboard_api, zone_list_api

app_name = 'analytics'

urlpatterns = [
    path('', DashboardView.as_view(), name='dashboard'),
    path('dashboard/', DashboardView.as_view(), name='dashboard_alt'),
    path('api/stats.json', api_stats, name='api_stats'),
    path('api/predictions.json', api_predictions, name='api_predictions'),
    path('api/alerts.json', api_alerts, name='api_alerts'),
    path('api/zone/<int:zone_id>/', api_zone_detail, name='api_zone_detail'),
    path('api/zone/<int:zone_id>/propagation/', api_zone_propagation_path, name='api_zone_propagation'),
    
    # Training API
    path('api/training/metrics/', training_metrics_api, name='training_metrics'),
    path('api/training/tier-distribution/', model_tier_distribution_api, name='tier_distribution'),
    path('api/training/performance/', model_performance_api, name='model_performance'),
    
    # Task management
    path('tasks/', TaskRunListView.as_view(), name='task_list'),
    path('tasks/launch/training/', launch_training_view, name='launch_training'),
    path('tasks/launch/ingest/', launch_ingest_view, name='launch_ingest'),
    path('tasks/launch/ingest_climate/', launch_ingest_climate_view, name='launch_ingest_climate'),
    path('tasks/api/<int:task_id>/status/', task_status_api, name='task_status_api'),
    path('tasks/<int:task_id>/stop/', stop_task_view, name='stop_task'),
    
    # Analysis and graphs
    path('analysis/province/', ProvinceAnalysisView.as_view(), name='province_analysis'),
    path('analysis/disease/', DiseaseDetailView.as_view(), name='disease_detail'),
    path('analysis/disease/interactive/', EnhancedDiseaseAnalysisView.as_view(), name='disease_interactive'),
    
    # Jupyter Notebook API
    path('api/notebooks/', list_notebooks, name='list_notebooks'),
    path('api/notebooks/execute/', execute_notebook, name='execute_notebook'),
    path('api/notebooks/status/<int:task_id>/', get_notebook_status, name='notebook_status'),
    
    # Notebook Control Interface
    path('notebooks/', NotebookControlView.as_view(), name='notebook_control'),
    path('jupyter/', JupyterProxyView.as_view(), name='jupyter_embed'),

    # Climate & Provincial Analysis APIs
    path('api/climate/province/', province_climate_api, name='province_climate'),
    path('api/heatmap/propagation/', zone_propagation_heatmap_api, name='zone_heatmap'),
    path('api/province/predictions/', provincial_predictions_api, name='province_predictions'),
    path('api/map/propagation/', disease_propagation_map_api, name='disease_propagation_map'),

    # Province Dashboard Data
    path('api/province/dashboard/', province_dashboard_api, name='province_dashboard'),
    path('api/province/zones/', zone_list_api, name='zone_list'),

    # Training Performance API (Dynamic Graphs)
    path('api/training/performance/dynamic/', training_performance_api, name='training_performance_dynamic'),
    path('api/zones/recurrence/', zone_recurrence_api, name='zone_recurrence'),

    # Pressure Virale Graph API (Top 10 zones, robust models only)
    path('api/pressure/virale/', pressure_virale_api, name='pressure_virale'),
]
