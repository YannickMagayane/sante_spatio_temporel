from django.urls import path
from .views import (
    DashboardView, api_stats, api_predictions, api_alerts,
    TaskRunListView, launch_training_view, launch_ingest_view, launch_ingest_climate_view, task_status_api
)
from .graph_views import ProvinceAnalysisView, DiseaseDetailView
from .disease_interactive_view import DiseaseInteractiveView
from .enhanced_disease_view import EnhancedDiseaseAnalysisView

app_name = 'analytics'

urlpatterns = [
    path('', DashboardView.as_view(), name='dashboard'),
    path('dashboard/', DashboardView.as_view(), name='dashboard_alt'),
    path('api/stats.json', api_stats, name='api_stats'),
    path('api/predictions.json', api_predictions, name='api_predictions'),
    path('api/alerts.json', api_alerts, name='api_alerts'),
    
    # Task management
    path('tasks/', TaskRunListView.as_view(), name='task_list'),
    path('tasks/launch/training/', launch_training_view, name='launch_training'),
    path('tasks/launch/ingest/', launch_ingest_view, name='launch_ingest'),
    path('tasks/launch/ingest_climate/', launch_ingest_climate_view, name='launch_ingest_climate'),
    path('tasks/api/<int:task_id>/status/', task_status_api, name='task_status_api'),
    
    # Analysis and graphs
    path('analysis/province/', ProvinceAnalysisView.as_view(), name='province_analysis'),
    path('analysis/disease/', DiseaseDetailView.as_view(), name='disease_detail'),
    path('analysis/disease/interactive/', EnhancedDiseaseAnalysisView.as_view(), name='disease_interactive'),
]
