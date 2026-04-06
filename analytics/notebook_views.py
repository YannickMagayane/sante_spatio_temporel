"""
Notebook Management Views - Control Jupyter Notebooks from Django Dashboard
"""

from django.shortcuts import render
from django.views.generic import TemplateView
from django.http import JsonResponse
from analytics.models import TaskRun
from pathlib import Path
import json

class NotebookControlView(TemplateView):
    """Main view for controlling Jupyter notebooks"""
    template_name = "analytics/notebook_control.html"
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Available notebooks
        notebooks_dir = Path("/app/notebooks")
        notebooks = []
        
        for nb_file in notebooks_dir.glob('*.ipynb'):
            # Skip output notebooks
            if '.output_' not in nb_file.name:
                try:
                    with open(nb_file, 'r') as f:
                        data = json.load(f)
                        title = ""
                        description = ""
                        if data['cells'] and data['cells'][0]['cell_type'] == 'markdown':
                            content = ''.join(data['cells'][0]['source']) if data['cells'][0]['source'] else ""
                            lines = content.split('\n')
                            title = lines[0].replace('# ', '').strip()
                            if len(lines) > 1:
                                description = '\n'.join(lines[1:]).strip()
                        
                        notebooks.append({
                            'id': nb_file.stem,
                            'name': nb_file.name,
                            'title': title,
                            'description': description,
                            'path': str(nb_file)
                        })
                except Exception as e:
                    print(f"Error loading {nb_file}: {e}")
        
        context['notebooks'] = notebooks
        
        # Recent task runs
        context['recent_tasks'] = TaskRun.objects.filter(
            task_type='notebook_execution'
        ).order_by('-created_at')[:10]
        
        # Statistics
        context['total_executions'] = TaskRun.objects.filter(
            task_type='notebook_execution'
        ).count()
        
        context['successful_executions'] = TaskRun.objects.filter(
            task_type='notebook_execution',
            status='completed'
        ).count()
        
        context['failed_executions'] = TaskRun.objects.filter(
            task_type='notebook_execution',
            status='failed'
        ).count()
        
        return context


class JupyterProxyView(TemplateView):
    """Embed JupyterLab in an iframe"""
    template_name = "analytics/jupyter_embed.html"
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['jupyter_url'] = 'http://jupyter:8888'
        return context
