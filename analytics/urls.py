from django.urls import path
from .views import DashboardView, api_stats, api_predictions, api_alerts

app_name = 'analytics'

urlpatterns = [
    path('', DashboardView.as_view(), name='dashboard'),
    path('dashboard/', DashboardView.as_view(), name='dashboard_alt'),
    path('api/stats.json', api_stats, name='api_stats'),
    path('api/predictions.json', api_predictions, name='api_predictions'),
    path('api/alerts.json', api_alerts, name='api_alerts'),
]
