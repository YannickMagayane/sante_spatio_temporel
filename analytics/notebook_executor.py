"""
Notebook Execution API - Integration with Django
Allows running Jupyter notebooks from the dashboard
"""

import os
import subprocess
import json
from pathlib import Path
from datetime import datetime
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from analytics.models import TaskRun

NOTEBOOKS_DIR = Path("/app/notebooks")

@csrf_exempt
@require_POST
def execute_notebook(request):
    """Execute a Jupyter notebook and return results"""
    try:
        data = json.loads(request.body)
        notebook_name = data.get('notebook')
        
        if not notebook_name:
            return JsonResponse({'error': 'Missing notebook name'}, status=400)
        
        notebook_path = NOTEBOOKS_DIR / f"{notebook_name}.ipynb"
        output_path = NOTEBOOKS_DIR / f".output_{notebook_name}_{datetime.now().timestamp()}.ipynb"
        
        if not notebook_path.exists():
            return JsonResponse({'error': f'Notebook not found: {notebook_name}'}, status=404)
        
        # Create task record
        task = TaskRun.objects.create(
            task_name=f"Execute {notebook_name}",
            task_type='notebook_execution',
            status='running'
        )
        
        # Execute notebook
        cmd = [
            'jupyter', 'nbconvert',
            '--to', 'notebook',
            '--execute',
            f'--output={output_path}',
            str(notebook_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                task.status = 'completed'
                task.result = {'output': str(output_path), 'timestamp': datetime.now().isoformat()}
                task.save()
                
                return JsonResponse({
                    'success': True,
                    'message': f'Notebook {notebook_name} executed successfully',
                    'task_id': task.id,
                    'output': str(output_path)
                })
            else:
                task.status = 'failed'
                task.result = {'error': result.stderr, 'stdout': result.stdout}
                task.save()
                
                return JsonResponse({
                    'success': False,
                    'error': result.stderr,
                    'task_id': task.id
                }, status=500)
        
        except subprocess.TimeoutExpired:
            task.status = 'failed'
            task.result = {'error': 'Execution timeout (1 hour)'}
            task.save()
            return JsonResponse({
                'success': False,
                'error': 'Notebook execution timeout',
                'task_id': task.id
            }, status=500)
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@require_POST
def list_notebooks(request):
    """List all available notebooks"""
    try:
        notebooks = []
        for nb_file in NOTEBOOKS_DIR.glob('*.ipynb'):
            # Skip output notebooks
            if '.output_' not in nb_file.name:
                with open(nb_file, 'r') as f:
                    data = json.load(f)
                    title = ""
                    if data['cells'] and data['cells'][0]['cell_type'] == 'markdown':
                        title = data['cells'][0]['source'][0] if data['cells'][0]['source'] else nb_file.stem
                    
                    notebooks.append({
                        'id': nb_file.stem,
                        'name': nb_file.name,
                        'title': title.replace('# ', '').strip(),
                        'path': str(nb_file)
                    })
        
        return JsonResponse({'notebooks': notebooks})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def get_notebook_status(request, task_id):
    """Get execution status of a notebook"""
    try:
        task = TaskRun.objects.get(id=task_id)
        return JsonResponse({
            'task_id': task.id,
            'status': task.status,
            'task_name': task.task_name,
            'result': task.result,
            'created_at': task.created_at.isoformat(),
            'updated_at': task.updated_at.isoformat()
        })
    except TaskRun.DoesNotExist:
        return JsonResponse({'error': 'Task not found'}, status=404)
