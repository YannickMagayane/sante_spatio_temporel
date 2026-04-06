"""
Gestionnaire de tâches asynchrones pour entraînement et ingestion.
Exécute les scripts en arrière-plan sans bloquer l'interface Django.
"""

import subprocess
import threading
import logging
import os
from datetime import datetime
from pathlib import Path
from django.utils import timezone

from .models import TaskRun

logger = logging.getLogger(__name__)

class TaskRunner:
    """Exécute les tâches en arrière-plan via threading."""
    
    SCRIPTS_DIR = Path("/app/scripts")
    
    @staticmethod
    def run_task_async(task_type: str):
        """Lancer une tâche en arrière-plan."""
        # Créer un TaskRun
        task_run = TaskRun.objects.create(
            task_type=task_type,
            status='pending'
        )
        
        # Lancer la tâche dans un thread daemon
        thread = threading.Thread(
            target=TaskRunner._execute_task,
            args=(task_run.id, task_type),
            daemon=True
        )
        thread.start()
        
        return task_run
    
    @staticmethod
    def _execute_task(task_id: int, task_type: str):
        """Exécuter la tâche et mettre à jour le statut."""
        try:
            task_run = TaskRun.objects.get(id=task_id)
            task_run.status = 'running'
            task_run.started_at = timezone.now()
            task_run.save()
            
            if task_type == 'train':
                TaskRunner._train_models(task_run)
            elif task_type == 'ingest':
                TaskRunner._ingest_hdx_data(task_run)
            elif task_type == 'ingest_climate':
                TaskRunner._ingest_climate_data(task_run)
            
            # Succès
            task_run.status = 'completed'
            task_run.progress = 100
            task_run.completed_at = timezone.now()
            
        except Exception as e:
            logger.exception(f"Erreur lors de l'exécution de la tâche {task_id}")
            task_run.status = 'failed'
            task_run.error_log = str(e)
            task_run.completed_at = timezone.now()
        
        finally:
            task_run.save()
    
    @staticmethod
    def _train_models(task_run: TaskRun):
        """Entraîner les modèles."""
        logger.info(f"[Task {task_run.id}] Démarrage de l'entraînement des modèles...")

        script_path = TaskRunner.SCRIPTS_DIR / "train_models.py"

        try:
            # Utiliser subprocess pour exécuter le script
            process = subprocess.Popen(
                ["python", str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combiner stderr dans stdout
                text=True,
                cwd="/app"
            )

            # Enregistrer le PID pour pouvoir arrêter le processus
            task_run.process_id = process.pid
            task_run.save()

            # Capturer la sortie
            stdout, _ = process.communicate()

            # Effacer le PID car le processus est terminé
            task_run.process_id = None
            task_run.save()

            task_run.output_log = stdout

            # Vérifier si des erreurs critiques sont présentes dans les logs
            is_error = (
                process.returncode != 0 and
                stdout and
                ('ERROR' in stdout or 'Traceback' in stdout or 'Exception' in stdout)
            )

            if is_error:
                raise Exception(f"Script retourné code {process.returncode}\n{stdout[-1000:]}")

            logger.info(f"[Task {task_run.id}] Entraînement complété")

        except Exception as e:
            logger.error(f"[Task {task_run.id}] Erreur entraînement: {e}")
            raise
    
    @staticmethod
    def _ingest_hdx_data(task_run: TaskRun):
        """Injecter les données HDX."""
        logger.info(f"[Task {task_run.id}] Démarrage de l'ingestion HDX...")

        script_path = TaskRunner.SCRIPTS_DIR / "ingest_hdx.py"

        try:
            # Utiliser subprocess pour exécuter le script
            process = subprocess.Popen(
                ["python", str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combiner stderr dans stdout
                text=True,
                cwd="/app"
            )

            # Enregistrer le PID pour pouvoir arrêter le processus
            task_run.process_id = process.pid
            task_run.save()

            # Capturer la sortie
            stdout, _ = process.communicate()

            # Effacer le PID car le processus est terminé
            task_run.process_id = None
            task_run.save()

            task_run.output_log = stdout

            # Vérifier si des erreurs critiques sont présentes dans les logs
            is_error = (
                process.returncode != 0 and
                stdout and
                ('ERROR' in stdout or 'Traceback' in stdout or 'Exception' in stdout)
            )

            if is_error:
                raise Exception(f"Script retourné code {process.returncode}\n{stdout[-1000:]}")

            logger.info(f"[Task {task_run.id}] Ingestion complétée")

        except Exception as e:
            logger.error(f"[Task {task_run.id}] Erreur ingestion: {e}")
            raise
    
    @staticmethod
    def _ingest_climate_data(task_run: TaskRun):
        """Injecter les données climatiques ERA5-Land."""
        logger.info(f"[Task {task_run.id}] Démarrage de l'ingestion climatique...")

        script_path = TaskRunner.SCRIPTS_DIR / "ingest_hdx.py"

        try:
            # Utiliser subprocess avec l'argument --climate-only
            process = subprocess.Popen(
                ["python", str(script_path), "--climate-only"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combiner stderr dans stdout
                text=True,
                cwd="/app"
            )

            # Enregistrer le PID pour pouvoir arrêter le processus
            task_run.process_id = process.pid
            task_run.save()

            # Capturer la sortie
            stdout, _ = process.communicate()

            # Effacer le PID car le processus est terminé
            task_run.process_id = None
            task_run.save()

            task_run.output_log = stdout

            # Vérifier si des erreurs critiques sont présentes dans les logs
            is_error = (
                process.returncode != 0 and
                stdout and
                ('ERROR' in stdout or 'Traceback' in stdout or 'Exception' in stdout)
            )

            if is_error:
                raise Exception(f"Script retourné code {process.returncode}\n{stdout[-1000:]}")

            logger.info(f"[Task {task_run.id}] Ingestion climatique complétée")

        except Exception as e:
            logger.error(f"[Task {task_run.id}] Erreur ingestion climatique: {e}")
            raise
