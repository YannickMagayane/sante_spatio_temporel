"""
Training Data API - Retrieve model training metrics for dashboard
"""

import json
from pathlib import Path
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

@require_http_methods(["GET"])
def training_metrics_api(request):
    """Get training metrics from last training run"""
    try:
        dashboard_data_file = Path("/app/models/dashboard_training_data.json")
        
        if dashboard_data_file.exists():
            with open(dashboard_data_file, 'r') as f:
                data = json.load(f)
            
            return JsonResponse({
                'success': True,
                'data': data
            })
        else:
            return JsonResponse({
                'success': False,
                'message': 'No training data available. Run training notebook first.',
                'data': None
            })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
            'data': None
        }, status=500)


@require_http_methods(["GET"])
def model_tier_distribution_api(request):
    """Get distribution of model tiers used"""
    try:
        dashboard_data_file = Path("/app/models/dashboard_training_data.json")
        
        if dashboard_data_file.exists():
            with open(dashboard_data_file, 'r') as f:
                data = json.load(f)
            
            tier_dist = data.get('tier_distribution', {})
            
            return JsonResponse({
                'success': True,
                'labels': list(tier_dist.keys()),
                'values': list(tier_dist.values()),
                'total': sum(tier_dist.values())
            })
        else:
            return JsonResponse({
                'success': False,
                'message': 'No training data available'
            })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def model_performance_api(request):
    """Get individual model performance metrics"""
    try:
        dashboard_data_file = Path("/app/models/dashboard_training_data.json")
        
        if dashboard_data_file.exists():
            with open(dashboard_data_file, 'r') as f:
                data = json.load(f)
            
            performance = data.get('model_performance', {})
            
            return JsonResponse({
                'success': True,
                'models': performance,
                'count': len(performance)
            })
        else:
            return JsonResponse({
                'success': False,
                'message': 'No training data available'
            })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
